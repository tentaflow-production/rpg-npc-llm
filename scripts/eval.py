#!/usr/bin/env python3
"""Evaluation harness (M1): ``eval.py --config <yaml> --job-dir <dir> [--limit N] [--no-judge] [--no-server]``.

Reads a frozen eval config (see ``experiments/eval_smoke.yaml``), runs the fixed eval sets through the model and
writes ``<job-dir>/eval.json`` (schema: supervisor ``experiments/PROTOCOL.md`` "Metrics"). Exit 0 and ``eval.json``
on success; non-zero and no ``eval.json`` otherwise.

Backends: ``server`` (default; starts the local vLLM server through ``rpgnpc.jobs.local_server`` and stops it again)
and ``reference`` (``--no-server``: replays each row's reference completion; offline, no GPU, smoke tests only).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rpgnpc import evalkit, matcher

os.environ.setdefault("ML_REPO", str(ROOT))  # job_dispatch's run.sh does not export it; configs reference ${ML_REPO}
WORKERS = int(os.environ.get("EVAL_WORKERS", "8"))
REQUEST_TIMEOUT = 300


def _expand(v):
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def check_sha256(path: str | Path) -> None:
    """Refuse an eval file whose sha256 differs from the committed ``data/<name>.sha256``."""
    path = Path(path)
    sidecar = ROOT / "data" / f"{path.name}.sha256"
    if not sidecar.exists():
        raise ValueError(f"no committed hash for {path.name} ({sidecar})")
    want = sidecar.read_text().split()[0]
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise ValueError(f"{path} sha256 {got[:12]} != committed {want[:12]}: the eval set is fixed")


def _post(url: str, payload: dict) -> dict:
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:  # local server only (H8)
        return json.loads(r.read())


def _served_model(base_url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def _chat(base_url: str, model: str, messages: list[dict], dec: dict, n: int = 1) -> list[str]:
    body = {"model": model, "messages": messages, "temperature": dec["temperature"], "top_p": dec["top_p"],
            "max_tokens": dec["max_new_tokens"], "n": n}
    return [c["message"]["content"] for c in _post(base_url.rstrip("/") + "/chat/completions", body)["choices"]]


def _eval_loss(base_url: str, model: str, model_dir: str, rows: list[dict]) -> float:
    """Mean per-token NLL of the reference completion given the prompt (server ``prompt_logprobs``). [inferred, Q33]"""
    from transformers import AutoTokenizer  # local files only (HF_HUB_OFFLINE=1)

    tok = AutoTokenizer.from_pretrained(model_dir)
    total, count = 0.0, 0

    def one(row: dict) -> tuple[float, int]:
        prefix = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        n_prefix = len(tok(prefix)["input_ids"])
        ids = tok(prefix + row["reference"] + "<|im_end|>")["input_ids"]
        res = _post(base_url.rstrip("/") + "/completions",
                    {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0})
        plp = res["choices"][0]["prompt_logprobs"]
        lps = []
        for tid, entry in zip(ids[n_prefix:], plp[n_prefix:]):
            lps.append(entry[str(tid)]["logprob"])
        return -sum(lps), len(lps)

    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for nll, k in ex.map(one, rows):
            total, count = total + nll, count + k
    if not count or not math.isfinite(total):
        raise RuntimeError("eval_loss could not be computed")
    return total / count


def _judge_all(base_url: str, model: str, rows: list[dict], style: dict, version: str) -> dict:
    import judge

    n = int(style.get("samples_per_prompt", 1))

    def one(row: dict) -> tuple[str, dict | None]:
        lines = [evalkit.parse_envelope(o)["line"] for o in _chat(base_url, model, row["messages"], style, n)]
        ics, oks = [], []
        for line in lines:
            reply = _chat(base_url, model, judge.build_judge_messages(row, line),
                          {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 40})[0]
            try:
                r = judge.parse_judge_reply(reply)
            except ValueError:
                continue  # an unparsable judge reply is dropped, not guessed
            ics.append(r["in_character"])
            oks.append(1.0 if r["silence_ok"] else 0.0)
        if not ics:
            return row["id"], None
        return row["id"], {"in_character": sum(ics) / len(ics), "silence_ok": sum(oks) / len(oks)}

    assert version == judge.PROMPT_VERSION, f"config judge_prompt_version {version} != judge.py {judge.PROMPT_VERSION}"
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        return {rid: j for rid, j in ex.map(one, rows) if j}


def run(ns: argparse.Namespace) -> dict:
    cfg = _expand(yaml.safe_load(Path(ns.config).read_text(encoding="utf-8")))
    ecfg = cfg["eval"]
    limit = ns.limit if ns.limit is not None else int(ecfg.get("limit", 0) or 0)
    job_dir = Path(ns.job_dir)
    evalkit.refuse_baseline_dir(job_dir, limit)
    if ns.no_server and job_dir.name.startswith("exp_"):
        raise ValueError("an offline (--no-server) run must not write into an experiment directory")
    leak_dec, style_dec = ecfg["leak_decoding"], ecfg["style_decoding"]
    if float(leak_dec["temperature"]) != 0.0:
        raise ValueError("eval.leak_decoding.temperature must be 0 (greedy leak scoring)")
    d = cfg["data"]
    for key in ("eval", "eval_adversarial"):
        check_sha256(d[key])
    secrets = matcher.load_secrets(d["secrets"])
    ordinary, adversarial = evalkit.load_rows(d["eval"]), evalkit.load_rows(d["eval_adversarial"])
    set_hash = evalkit.eval_set_hash(d["eval"], d["eval_adversarial"])
    if limit:
        ordinary, adversarial = evalkit.select_limit(ordinary, adversarial, limit)
    rows = ordinary + adversarial
    version = ecfg["judge_prompt_version"]
    decoding = {"leak": leak_dec, "style": style_dec}
    extra: dict = {"smoke": bool(limit) or ns.no_server, "limit": limit}

    if ns.no_server:
        outputs = {r["id"]: r["reference"] for r in rows}
        loss = None
        extra["backend"] = "reference"
    else:
        from rpgnpc import jobs

        if cfg["model"].get("adapter"):
            raise NotImplementedError("adapter evaluation needs a merged model dir (serve.sh serves one model); see handoff")
        model_dir = cfg["model"]["base"]
        extra["backend"] = "server"
        with jobs.local_server(model_dir) as url:
            model = _served_model(url)

            def gen(row: dict) -> tuple[str, str]:
                return row["id"], _chat(url, model, row["messages"], leak_dec)[0]

            with cf.ThreadPoolExecutor(WORKERS) as ex:
                outputs = dict(ex.map(gen, rows))
            loss = _eval_loss(url, model, model_dir, rows)
            if not ns.no_judge:
                extra["judgments"] = _judge_all(url, model, rows, style_dec, version)
    return evalkit.build_eval_json(cfg["exp_name"], ordinary, adversarial, outputs, secrets, set_hash, decoding, loss,
                                   version, extra)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke subset: N/2 ordinary + N/2 adversarial rows")
    ap.add_argument("--no-judge", action="store_true", help="skip in_character / silence_compliance")
    ap.add_argument("--no-server", action="store_true", help="offline reference backend (smoke only)")
    ns = ap.parse_args(argv)
    try:
        ev = run(ns)
    except Exception as e:  # noqa: BLE001 - any failure: no eval.json, non-zero exit
        print(f"eval.py: FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    job = Path(ns.job_dir)
    job.mkdir(parents=True, exist_ok=True)
    tmp = job / "eval.json.tmp"
    tmp.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(job / "eval.json")
    print(f"eval.py: wrote {job / 'eval.json'} (n={ev['n']}, leak_adv={ev['leak_rate_adversarial']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
