"""Unit tests for src/rpgnpc/jobs.py and scripts/serve.sh — offline, no GPU, no vLLM."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_latency

from rpgnpc import jobs

QWEN7B = {"num_hidden_layers": 28, "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584}


def make_model(tmp_path: Path, weight_mib: int, cfg: dict = QWEN7B) -> Path:
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    with open(d / "model-00001.safetensors", "wb") as f:
        f.truncate(weight_mib * jobs.MIB)  # sparse file
    return d


def test_kv_cache_qwen7b():
    # 2 * 28 layers * 4 kv heads * 128 head_dim * 2 B = 57344 B per token
    assert jobs.kv_cache_bytes(QWEN7B, 1000) == 57344 * 1000


def test_weights_bytes_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        jobs.weights_bytes(tmp_path)


def test_fit_ok(tmp_path):
    m = make_model(tmp_path, 15000)
    rep = jobs.fit_check(m, free_mib=30000)
    assert rep["weights_mib"] == 15000 and rep["need_mib"] < rep["budget_mib"]


def test_fit_refuses_when_too_big(tmp_path):
    m = make_model(tmp_path, 15000)
    with pytest.raises(jobs.FitError):
        jobs.fit_check(m, free_mib=16000)


def test_fit_refuses_when_free_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "free_vram_mib", lambda: None)
    with pytest.raises(jobs.FitError):
        jobs.fit_check(make_model(tmp_path, 100))


def test_cli_exit_codes(tmp_path):
    m = make_model(tmp_path, 15000)
    assert jobs.main(["fit-check", "--model", str(m), "--free-mib", "30000"]) == 0
    assert jobs.main(["fit-check", "--model", str(m), "--free-mib", "1000"]) == 2


def test_local_server_always_stops(monkeypatch):
    calls = []

    def fake(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(jobs, "_serve", fake)
    with pytest.raises(ValueError), jobs.local_server("/m"):
        raise ValueError("job failed")
    assert calls[0][-1] == "--start" and calls[-1] == ("--stop",)


def test_local_server_stops_when_start_fails(monkeypatch):
    calls = []

    def fake(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1 if args[-1] == "--start" else 0)

    monkeypatch.setattr(jobs, "_serve", fake)
    with pytest.raises(RuntimeError), jobs.local_server("/m"):
        pass
    assert calls[-1] == ("--stop",)


def test_serve_sh_usage_errors():
    r = subprocess.run(["bash", str(ROOT / "scripts" / "serve.sh")], capture_output=True, text=True, check=False)
    assert r.returncode == 64
    r = subprocess.run(["bash", str(ROOT / "scripts" / "serve.sh"), "--bogus"], capture_output=True, text=True, check=False)
    assert r.returncode == 64


def test_serve_sh_fit_check_refuses_too_big_model(tmp_path):
    m = make_model(tmp_path, 15000)
    env = {**os.environ, "SERVE_GPU_UTIL": "0.01"}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "serve.sh"), "--model", str(m), "--fit-check"],
                       capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 2
    assert "REFUSED" in r.stderr


def test_percentile():
    assert bench_latency.percentile([1.0, 2.0, 3.0], 0.5) == 2.0
