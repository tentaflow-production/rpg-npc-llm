"""Leak matcher over ``data/secrets.yaml`` (M1). STUB: the spec phase only fixes the interface.

A *leak* is a hit of a secret whose id is not in the prompt's ``secret_ids_permitted``.
Until M4 reports recall/precision a hit means "matcher-detected leak".
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def load_secrets(path: str | Path) -> list[dict]:
    """Return the ``secrets`` list of ``secrets.yaml`` (each: id, npc, topic, patterns); validates the schema."""
    raise NotImplementedError


def normalize(text: str) -> str:
    """NFKC-normalise and drop all whitespace, so spaced-out or full-width spellings still match."""
    raise NotImplementedError


def find_leaks(text: str, secrets: list[dict], permitted: Iterable[str] = ()) -> list[str]:
    """Sorted ids of secrets whose pattern occurs in ``text`` and that are not in ``permitted``.

    A pattern is a plain substring, or a regex when it starts with ``re:``. The topic label alone is not a leak.
    """
    raise NotImplementedError
