# Local inference server (vLLM)

`scripts/serve.sh` starts/stops one OpenAI-compatible vLLM server on `$LOCAL_LLM_URL` (default `http://127.0.0.1:8000/v1`).
vLLM runs in its own conda env `$SERVE_CONDA_ENV` (default `serve`, created by the human; Q21) so it never replaces the torch of `$CONDA_ENV`.

```
bash scripts/serve.sh --model <dir> --fit-check   # VRAM check only
bash scripts/serve.sh --model <dir> --start       # fit check, start detached, wait for /models
bash scripts/serve.sh --check-only                # exit 0 iff the server answers
bash scripts/serve.sh --stop
python scripts/bench_latency.py --url "$LOCAL_LLM_URL" --n 5 --max-p50-sec 3.0
```

Env knobs: `SERVE_MAX_MODEL_LEN` (4096), `SERVE_GPU_UTIL` (0.90), `SERVE_START_TIMEOUT` (600 s), `SERVE_PIDFILE`.

## Fit check

`need = weights (sum of *.safetensors) + KV cache (4 seqs x max_model_len) + 1.5 GiB overhead`; refused when `need > free VRAM x SERVE_GPU_UTIL`,
or when free VRAM cannot be determined (no `nvidia-smi` and no CUDA torch). Computed on the RTX 5090 (≈30.9 GiB free, desktop idle), max_model_len 4096:

| Model | weights | KV (4 seqs) | need | usable (0.90) |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct bf16 | 14.2 GiB | 0.9 GiB | 16.6 GiB | 27.2 GiB |
| Qwen2.5-32B-Instruct-AWQ | 18.0 GiB | 4.0 GiB | 23.5 GiB | 27.2 GiB |

Neither fits beside training (PROTOCOL "VRAM budget"). Jobs use `rpgnpc.jobs.local_server(model)`, a context manager that always stops the server.
