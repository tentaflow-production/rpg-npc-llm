"""NPC prompt-contract renderer (M2). SPEC-PHASE STUB: every function raises NotImplementedError.

Contract: ``$SUPERVISOR_REPO/docs/prompt_contract.md`` (Q18 envelope, Q19 language, D19 ``scenario_id``). Wording and the
closed action set live in ``data/templates/*.yaml`` (frozen oracles); tests: ``tests/test_prompt_contract.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "data" / "templates"


class UnsupportedLanguage(ValueError):
    """``case['language']`` is not in ``render.yaml: languages``."""


class PromptTooLong(ValueError):
    """System message > ``budget.system_tokens`` or whole prompt > ``budget.total_tokens`` (invariant 5)."""


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: number of non-ASCII characters + ceil(number of ASCII characters / 3)."""
    raise NotImplementedError


def render_messages(case: Mapping[str, Any], templates_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Render the chat messages (system, then history user/assistant pairs, then the current user action) for a fixture-shaped case.

    ``case`` keys: npc, scene, slot ("D2E"), scenario_id, language, belief_bucket, clue_ids_permitted, withheld_topics,
    stance_set, history ([{"action": {...}, "line": str}]), action ({"type": ..., args}). The final assistant turn is NOT included.
    """
    raise NotImplementedError


def render_envelope(head: Mapping[str, Any]) -> str:
    """Serialise ``{"stance", "reveal", "line"}`` as the assistant turn: ``json.dumps(..., ensure_ascii=False)``, keys in that order."""
    raise NotImplementedError


def validate_head(head: object, *, stance_set: Iterable[str], permitted_ids: Iterable[str]) -> bool:
    """True iff ``head`` is a dict with exactly the keys stance/reveal/line, stance in the set, reveal a list of permitted ids, non-empty line."""
    raise NotImplementedError
