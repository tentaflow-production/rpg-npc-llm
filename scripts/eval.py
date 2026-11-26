#!/usr/bin/env python3
"""Evaluation harness (M1): ``eval.py --config <yaml> --job-dir <dir> [--limit N] [--no-judge] [--no-server]``.

STUB: the spec phase only fixes the interface; tests/test_eval.py drives ``main``.
Exit 0 and ``<job-dir>/eval.json`` on success; non-zero and no ``eval.json`` otherwise.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
