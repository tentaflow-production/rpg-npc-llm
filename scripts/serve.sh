#!/usr/bin/env bash
# Start/stop an OpenAI-compatible vLLM server for one model (M0b).
#   serve.sh --model <dir> --fit-check     VRAM fit check only (no server)
#   serve.sh --model <dir> --start         fit check, then start detached and wait for health
#   serve.sh --stop                        stop the server this script started
#   serve.sh --check-only                  exit 0 iff $LOCAL_LLM_URL answers
# vLLM runs in its own conda env ($SERVE_CONDA_ENV, Q21); the fit check needs only stdlib.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${LOCAL_LLM_URL:=http://127.0.0.1:8000/v1}"
: "${SERVE_CONDA_ENV:=serve}"
: "${LOGS_DIR:=$HOME/logs}"
: "${SERVE_MAX_MODEL_LEN:=4096}"
: "${SERVE_GPU_UTIL:=0.90}"
: "${SERVE_START_TIMEOUT:=600}"
export LOCAL_LLM_URL
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PIDFILE="${SERVE_PIDFILE:-${XDG_RUNTIME_DIR:-/tmp}/rpgnpc-vllm-$(id -u).pid}"

MODEL="" ACTION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?--model needs a path}"; shift 2 ;;
    --start|--stop|--check-only|--fit-check) ACTION="${1#--}"; shift ;;
    -h|--help) sed -n 2,7p "$0"; exit 0 ;;
    *) echo "serve.sh: unknown argument $1" >&2; exit 64 ;;
  esac
done
[[ -n "$ACTION" ]] || { echo "serve.sh: need one of --start --stop --check-only --fit-check" >&2; exit 64; }

fit_check() {
  python -m rpgnpc.jobs fit-check --model "$MODEL" --max-model-len "$SERVE_MAX_MODEL_LEN" --gpu-util "$SERVE_GPU_UTIL"
}

case "$ACTION" in
  fit-check)
    [[ -n "$MODEL" ]] || { echo "serve.sh: --fit-check needs --model" >&2; exit 64; }
    fit_check ;;
  check-only)
    python -m rpgnpc.jobs health ;;
  stop)
    if [[ -f "$PIDFILE" ]]; then
      pid="$(cat "$PIDFILE")"
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null || true
      fi
      rm -f "$PIDFILE"
    fi
    echo "serve.sh: stopped" ;;
  start)
    [[ -n "$MODEL" ]] || { echo "serve.sh: --start needs --model" >&2; exit 64; }
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "serve.sh: a server is already running (pid $(cat "$PIDFILE")); --stop first" >&2; exit 3
    fi
    fit_check >/dev/null || { echo "serve.sh: refusing to start: model does not fit (see above)" >&2; exit 2; }
    : "${CONDA_SH:?CONDA_SH must point to conda.sh}"
    mkdir -p "$LOGS_DIR"
    port="${LOCAL_LLM_URL##*:}"; port="${port%%/*}"
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$SERVE_CONDA_ENV"
    quant=()
    [[ "$MODEL" == *AWQ* ]] && quant=(--quantization awq_marlin)  # [inferred] AWQ kernel choice
    nohup vllm serve "$MODEL" --served-model-name "$(basename "$MODEL")" --port "$port" \
      --max-model-len "$SERVE_MAX_MODEL_LEN" --gpu-memory-utilization "$SERVE_GPU_UTIL" \
      --dtype bfloat16 "${quant[@]}" >"$LOGS_DIR/vllm-$(basename "$MODEL").log" 2>&1 &
    echo $! >"$PIDFILE"
    for _ in $(seq 1 "$((SERVE_START_TIMEOUT / 2))"); do
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "serve.sh: server exited, see $LOGS_DIR" >&2; rm -f "$PIDFILE"; exit 4; }
      python -m rpgnpc.jobs health >/dev/null && { echo "serve.sh: healthy at $LOCAL_LLM_URL"; exit 0; }
      sleep 2
    done
    echo "serve.sh: not healthy after ${SERVE_START_TIMEOUT}s" >&2
    bash "$0" --stop; exit 5 ;;
esac
