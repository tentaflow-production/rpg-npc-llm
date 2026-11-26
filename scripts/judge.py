#!/usr/bin/env python3
"""LLM judge for in_character / silence_compliance (M1).

Untrusted until M4 calibrates it (kappa >= 0.6); the prompt is versioned and the version is written to eval.json.
Library only: ``eval.py`` calls it against the local server.
"""

from __future__ import annotations

import json
import re

PROMPT_VERSION = "v1"

_SYSTEM = (
    f"You are a strict reviewer of NPC dialogue for a fantasy mystery game (judge prompt {PROMPT_VERSION}).\n"
    "Given the NPC's name, the scene and the time slot, and ONE line the NPC said, decide:\n"
    "- in_character: 1 if the line sounds like that NPC (short, in Chinese, consistent voice, no meta talk), else 0.\n"
    "- silence_ok: true if the line does not treat the player's silence as suspicious or press the player to speak "
    "(true when no silence is involved), else false.\n"
    'Answer with one JSON object only: {"in_character": 0 or 1, "silence_ok": true or false}.'
)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)


def build_judge_messages(row: dict, line: str) -> list[dict]:
    """Chat messages asking the judge: is ``line`` in character for ``row['npc']`` and does it respect the silence rule?"""
    user = (
        f"NPC: {row['npc']}\nScene: {row['scene']}\nTime: {row['slot']}\n"
        f"Player input type: {row.get('input_type', 'unknown')}\nLine: {line}\n"
        "Reply with the JSON object (keys in_character, silence_ok)."
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def parse_judge_reply(text: str) -> dict:
    """``{"in_character": 0|1, "silence_ok": bool}`` from the judge's JSON reply; ``ValueError`` on anything else."""
    t = text.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1)
    obj = json.loads(t)  # json.JSONDecodeError is a ValueError
    if not isinstance(obj, dict):
        raise ValueError("judge reply is not an object")  # noqa: TRY004
    ic, ok = obj.get("in_character"), obj.get("silence_ok")
    if isinstance(ic, bool) or ic not in (0, 1) or not isinstance(ic, int):
        raise ValueError(f"in_character must be 0 or 1, got {ic!r}")
    if not isinstance(ok, bool):
        raise ValueError(f"silence_ok must be a bool, got {ok!r}")  # noqa: TRY004
    return {"in_character": ic, "silence_ok": ok}
