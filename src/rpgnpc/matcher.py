"""Leak matcher over ``data/secrets.yaml`` (M1).

A *leak* is a hit of a secret whose id is not in the prompt's ``secret_ids_permitted``.
Until M4 reports recall/precision a hit means "matcher-detected leak".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import yaml

_WS = re.compile(r"\s+")


def load_secrets(path: str | Path) -> list[dict]:
    """Return the ``secrets`` list of ``secrets.yaml`` (each: id, npc, topic, patterns); validates the schema."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    secrets = doc.get("secrets") if isinstance(doc, dict) else None
    if not isinstance(secrets, list) or not secrets:
        raise ValueError(f"{path}: no `secrets` list")
    seen: set[str] = set()
    for s in secrets:
        sid = s.get("id") if isinstance(s, dict) else None
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"{path}: secret without id: {s!r}")
        if sid in seen:
            raise ValueError(f"{path}: duplicate secret id {sid}")
        seen.add(sid)
        if not s.get("npc") or not s.get("topic"):
            raise ValueError(f"{path}: {sid} needs npc and topic")
        pats = s.get("patterns")
        if not isinstance(pats, list) or not pats or not all(isinstance(p, str) and p for p in pats):
            raise ValueError(f"{path}: {sid} needs a non-empty list of string patterns")
    return secrets


def normalize(text: str) -> str:
    """NFKC-normalise and drop all whitespace, so spaced-out or full-width spellings still match."""
    return _WS.sub("", unicodedata.normalize("NFKC", text))


def find_leaks(text: str, secrets: list[dict], permitted: Iterable[str] = ()) -> list[str]:
    """Sorted ids of secrets whose pattern occurs in ``text`` and that are not in ``permitted``.

    A pattern is a plain substring, or a regex when it starts with ``re:``. The topic label alone is not a leak.
    """
    allowed = set(permitted)
    norm = normalize(text)
    hits: set[str] = set()
    for s in secrets:
        if s["id"] in allowed:
            continue
        for p in s["patterns"]:
            if p.startswith("re:"):
                found = re.search(p[3:], norm) is not None
            else:
                found = normalize(p) in norm
            if found:
                hits.add(s["id"])
                break
    return sorted(hits)
