"""GPU job helpers: VRAM fit check and local vLLM server lifecycle (M0b).

Stdlib only, so it runs in any env (the server itself lives in $SERVE_CONDA_ENV, see Q21).
Entry points (gen/eval/judge) use ``local_server(model)``: it starts the server through
``scripts/serve.sh`` and always stops it again, so the dispatcher can demand a free GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

MIB = 1024 * 1024
OVERHEAD_MIB = 1536  # [inferred] CUDA context + activations/cudagraph buffers vLLM needs beyond weights + KV
DEFAULT_MAX_MODEL_LEN = 4096  # [inferred] NPC prompts are short; the 32k model default would not fit beside 32B
DEFAULT_GPU_UTIL = 0.90  # [inferred] vLLM --gpu-memory-utilization
KV_DTYPE_BYTES = 2  # fp16/bf16 KV cache

# Documented figures (RTX 5090, 32 GB) — see docs/serving.md. Computed by `fit-check`, not promised.
SERVE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "serve.sh"


class FitError(RuntimeError):
    """The model does not fit into the free VRAM (or the free VRAM is unknown)."""


def weights_bytes(model_dir: Path) -> int:
    files = list(model_dir.glob("*.safetensors")) or list(model_dir.glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"no weight files (*.safetensors / *.bin) in {model_dir}")
    return sum(f.stat().st_size for f in files)


def kv_cache_bytes(config: dict, max_model_len: int, max_seqs: int = 1) -> int:
    """KV cache for ``max_seqs`` sequences of ``max_model_len`` tokens."""
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim") or config["hidden_size"] // heads
    per_token = 2 * config["num_hidden_layers"] * kv_heads * head_dim * KV_DTYPE_BYTES
    return per_token * max_model_len * max_seqs


def free_vram_mib() -> int | None:
    """Free VRAM of GPU 0 in MiB, or None when it cannot be determined."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout.split()
            return int(out[0])
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            pass
    try:
        import torch  # type: ignore[import-not-found]

        free, _total = torch.cuda.mem_get_info()
        return int(free // MIB)
    except Exception:  # noqa: BLE001 - no torch, no CUDA, driver errors: all mean "unknown"
        return None


def fit_check(
    model_dir: str | Path,
    free_mib: int | None = None,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_seqs: int = 4,
    gpu_util: float = DEFAULT_GPU_UTIL,
) -> dict:
    """Return a report; raise FitError when weights + KV cache + overhead exceed the usable free VRAM."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    w = weights_bytes(model_dir) // MIB
    kv = kv_cache_bytes(config, max_model_len, max_seqs) // MIB
    need = w + kv + OVERHEAD_MIB
    free = free_mib if free_mib is not None else free_vram_mib()
    report = {
        "model": str(model_dir), "weights_mib": w, "kv_cache_mib": kv, "overhead_mib": OVERHEAD_MIB,
        "need_mib": need, "free_mib": free, "gpu_util": gpu_util, "max_model_len": max_model_len,
        "max_seqs": max_seqs,
    }
    if free is None:
        raise FitError(f"cannot determine free VRAM (no nvidia-smi / CUDA); need {need} MiB: {report}")
    budget = int(free * gpu_util)
    report["budget_mib"] = budget
    if need > budget:
        raise FitError(f"{model_dir.name} needs {need} MiB but only {budget} MiB usable of {free} free: {report}")
    return report


def server_url() -> str:
    return os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8000/v1")


def healthy(url: str | None = None, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen((url or server_url()).rstrip("/") + "/models", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _serve(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SERVE_SCRIPT), *args], check=False)


@contextmanager
def local_server(model: str | Path) -> Iterator[str]:
    """Start the vLLM server for ``model``; stop it on exit, whatever happens. Yields the base URL."""
    try:
        if _serve("--model", str(model), "--start").returncode != 0:
            raise RuntimeError(f"serve.sh --start failed for {model}")
        yield server_url()
    finally:
        _serve("--stop")


def wait_healthy(url: str | None = None, timeout_sec: float = 600.0, interval: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if healthy(url):
            return True
        time.sleep(interval)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fc = sub.add_parser("fit-check")
    fc.add_argument("--model", required=True)
    fc.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    fc.add_argument("--max-seqs", type=int, default=4)
    fc.add_argument("--gpu-util", type=float, default=DEFAULT_GPU_UTIL)
    fc.add_argument("--free-mib", type=int, default=None, help="override detected free VRAM (docs/tests)")
    sub.add_parser("health")
    ns = ap.parse_args(argv)
    if ns.cmd == "health":
        ok = healthy()
        print("healthy" if ok else "not healthy")
        return 0 if ok else 1
    try:
        print(json.dumps(fit_check(ns.model, ns.free_mib, ns.max_model_len, ns.max_seqs, ns.gpu_util)))
    except (FitError, FileNotFoundError) as e:
        print(f"fit-check: REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
