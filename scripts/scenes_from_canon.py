"""canon 04 presence matrix + case bible -> scenes.yaml (M2). SPEC-PHASE STUB.

Contract (pinned by tests/test_scenes_from_canon.py):
  scenes_from_canon.py --canon DIR [--out FILE]      write the generated scenes.yaml to FILE (default stdout)
  scenes_from_canon.py --canon DIR --check FILE      exit 0 iff FILE equals the generated document, else 1 with a diff-like
                                                     message naming the first differing keys on stderr
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def build_scenes(canon_dir: str | Path) -> dict[str, Any]:
    """Parse ``04-scenes-and-names.md`` (presence matrix) and apply the conflict defaults C1-C13; returns the scenes document."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("M2 spec stub")


if __name__ == "__main__":
    sys.exit(main())
