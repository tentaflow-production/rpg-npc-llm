#!/usr/bin/env python3
"""LLM judge for in_character / silence_compliance (M1). STUB: the spec phase only fixes the interface.

Untrusted until M4 calibrates it (kappa >= 0.6); the prompt is versioned and the version is written to eval.json.
"""

from __future__ import annotations

PROMPT_VERSION = "v1"


def build_judge_messages(row: dict, line: str) -> list[dict]:
    """Chat messages asking the judge: is ``line`` in character for ``row['npc']`` and does it respect the silence rule?"""
    raise NotImplementedError


def parse_judge_reply(text: str) -> dict:
    """``{"in_character": 0|1, "silence_ok": bool}`` from the judge's JSON reply; ``ValueError`` on anything else."""
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
