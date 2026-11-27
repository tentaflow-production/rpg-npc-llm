"""Candidate generator (M2), dispatched via ``job_dispatch.sh --type gen``. SPEC-PHASE STUB.

Contract (pinned by tests/test_dataset.py):
  gen_dataset.py --config CFG.yaml --job-dir DIR [--limit N] [--backend server|fake|fake_leaky]
Writes ``DIR/build/candidates/candidates.jsonl`` (+ ``DIR/build/rejected.jsonl``); NEVER a promoted dataset (Q20 option 1).
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("M2 spec stub")


if __name__ == "__main__":
    sys.exit(main())
