"""Latency benchmark against the local OpenAI-compatible server (M0b).

Sends ``--n`` short NPC-style chat requests sequentially; exits 1 if p50 exceeds ``--max-p50-sec``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

PROMPT = "你是码头的老渔夫。玩家问你：今天的海风怎么样？请用一句话回答。"


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def one_request(url: str, model: str, max_tokens: int) -> float:
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "temperature": 0.7,
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        json.load(r)
    return time.perf_counter() - t0


def first_model(url: str) -> str:
    with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=None, help="served model name (default: first the server lists)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-p50-sec", type=float, default=3.0)
    ns = ap.parse_args(argv)
    model = ns.model or first_model(ns.url)
    one_request(ns.url, model, ns.max_tokens)  # warm-up, not counted
    lat = [one_request(ns.url, model, ns.max_tokens) for _ in range(ns.n)]
    p50 = statistics.median(lat)
    print(json.dumps({"n": ns.n, "p50_sec": round(p50, 3), "p95_sec": round(percentile(lat, 0.95), 3),
                      "max_sec": round(max(lat), 3), "max_p50_sec": ns.max_p50_sec}))
    return 0 if p50 <= ns.max_p50_sec else 1


if __name__ == "__main__":
    sys.exit(main())
