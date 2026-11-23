#!/usr/bin/env bash
# Regression: the done-when commands of every merged item. CI runs this.
# Never edit a line you did not add in your own item. Append a block per item.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${EXPERIMENTS_DIR:=$HOME/experiments}"
export EXPERIMENTS_DIR

# --- M0: config-driven, resumable training ---------------------------------
python -m pytest tests -q
grep -q "safetensors" .gitignore && grep -q "jsonl" .gitignore && test -x .githooks/pre-commit
# The GPU smoke needs the workstation (models + datasets); skipped where they are absent (e.g. CI).
if [[ -d "${MODELS_DIR:-$HOME/models}/Qwen2.5-7B-Instruct" ]]; then
  python scripts/train_qwen_sft.py --config experiments/smoke.yaml --job-dir "$EXPERIMENTS_DIR/smoke" --dry-run
  python -c "import json,os;r=json.load(open(os.environ['EXPERIMENTS_DIR']+'/smoke/results.json'));assert r['steps']>=5 and all(k in r for k in ('git_sha','config_sha256','train_hash','val_hash','eval_hash'))"
  python scripts/train_qwen_sft.py --config experiments/smoke.yaml --job-dir "$EXPERIMENTS_DIR/smoke" --dry-run --resume-test
else
  echo "regression: *** SKIPPED *** GPU smoke, results.json check and resume test (no local models; M0 done-when NOT fully exercised)"
fi
if [[ -n "${SUPERVISOR_REPO:-}" && -f "$SUPERVISOR_REPO/docs/target-repo-workflows/on-merge.yml" ]]; then
  cmp -s .github/workflows/on-merge.yml "$SUPERVISOR_REPO/docs/target-repo-workflows/on-merge.yml"
else
  echo "regression: *** SKIPPED *** on-merge.yml comparison (SUPERVISOR_REPO not available)"
fi
