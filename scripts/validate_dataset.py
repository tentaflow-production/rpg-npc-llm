"""Dataset validator CLI (M2). SPEC-PHASE STUB: exits non-zero without a verdict.

Contract (pinned by tests/test_dataset.py): prints ONE JSON object on stdout
``{"ok": bool, "n": int, "metrics": {...}, "violations": [{"code": str, ...}]}`` and exits 0 iff ok, else 1.
Usage: validate_dataset.py DIR --min-n N --max-leak-rate F --max-imbalance F --contamination-against FILE... --allowlist FILE
       --require-manifest --require-field NAME --language L --min-offscript F [--secrets PATH] [--templates DIR]
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("M2 spec stub")


if __name__ == "__main__":
    sys.exit(main())
