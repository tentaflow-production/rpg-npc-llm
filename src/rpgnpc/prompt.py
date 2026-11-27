"""NPC prompt-contract renderer (M2).

Contract: ``$SUPERVISOR_REPO/docs/prompt_contract.md`` (Q18 envelope, Q19 language, D19 ``scenario_id``). Wording and the
closed action set live in ``data/templates/*.yaml`` (frozen oracles); tests: ``tests/test_prompt_contract.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "data" / "templates"
HEAD_KEYS = ("stance", "reveal", "line")


class UnsupportedLanguage(ValueError):
    """``case['language']`` is not in ``render.yaml: languages``."""


class PromptTooLong(ValueError):
    """System message > ``budget.system_tokens`` or whole prompt > ``budget.total_tokens`` (invariant 5)."""


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: number of non-ASCII characters + ceil(number of ASCII characters / 3)."""
    ascii_n = sum(1 for ch in text if ord(ch) < 128)
    return (len(text) - ascii_n) + math.ceil(ascii_n / 3)


def _load(templates_dir: Path, name: str) -> dict[str, Any]:
    return yaml.safe_load((templates_dir / name).read_text(encoding="utf-8"))


def render_action(action: object, render: Mapping[str, Any], language: str) -> str:
    """Render a player action from the closed set; unknown type, missing argument or non-dict raises ValueError."""
    if not isinstance(action, Mapping) or "type" not in action:
        raise ValueError(f"action must be a dict with a type: {action!r}")
    spec = render["actions"].get(action["type"])
    if spec is None:
        raise ValueError(f"unknown action type: {action['type']!r}")
    missing = [a for a in spec["args"] if not isinstance(action.get(a), str) or not action[a]]
    if missing:
        raise ValueError(f"action {action['type']} lacks {missing}")
    return spec[language].format(**{a: action[a] for a in spec["args"]})


def render_messages(case: Mapping[str, Any], templates_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Render the chat messages (system, then history user/assistant pairs, then the current user action) for a fixture-shaped case.

    ``case`` keys: npc, scene, slot ("D2E"), scenario_id, language, belief_bucket, clue_ids_permitted, withheld_topics,
    stance_set, history ([{"action": {...}, "line": str}]), action ({"type": ..., args}). The final assistant turn is NOT included.
    """
    tdir = Path(templates_dir) if templates_dir else DEFAULT_TEMPLATES_DIR
    render = _load(tdir, "render.yaml")
    lang = case["language"]
    if lang not in render["languages"]:
        raise UnsupportedLanguage(f"language {lang!r} not in {render['languages']}")
    t = render[lang]
    npc = case["npc"]

    cards = _load(tdir, "cards.yaml")["cards"]
    buckets = _load(tdir, "beliefs.yaml")["buckets"]
    clues = _load(tdir, "clues.yaml")["clues"]
    stances = _load(tdir, "stances.yaml")["stances"]
    if npc not in cards:
        raise ValueError(f"unknown npc {npc!r}")
    if case["belief_bucket"] not in buckets:
        raise ValueError(f"unknown belief bucket {case['belief_bucket']!r}")
    unknown = [c for c in case["clue_ids_permitted"] if c not in clues]
    if unknown:
        raise ValueError(f"unknown clue ids {unknown}")
    stance_set = list(case["stance_set"])
    if not stance_set or not set(stance_set) <= set(stances.get(npc, [])):
        raise ValueError(f"stance set {stance_set} does not fit {npc}")
    scene_name = render["scene_names"].get(case["scene"])
    if scene_name is None:
        raise ValueError(f"unknown scene {case['scene']!r}")
    slot = case["slot"]
    slot_label = render["slot_labels"][lang].get(slot[-1])
    if slot_label is None or not slot[1:-1].isdigit():
        raise ValueError(f"bad slot {slot!r}")
    if not case["scenario_id"]:
        raise ValueError("empty scenario_id")

    sections = [
        cards[npc],
        t["scene_line"].format(scene_name=scene_name, day=int(slot[1:-1]), slot_label=slot_label,
                               scenario_id=case["scenario_id"]),
        t["belief_line"].format(belief_text=buckets[case["belief_bucket"]]),
    ]
    sections += [t["knowledge_line"].format(clue_id=c, text=clues[c]) for c in case["clue_ids_permitted"]]
    if case["withheld_topics"]:
        sections.append(t["withheld_line"].format(topics=t["withheld_joiner"].join(case["withheld_topics"])))
    sections.append(t["stance_line"].format(stances=t["stance_joiner"].join(stance_set)))
    sections.append(t["style_rules"])
    system = render["section_joiner"].join(s for s in sections if s)

    messages = [{"role": "system", "content": system}]
    for turn in list(case["history"])[-render["max_history_turns"]:]:
        messages.append({"role": "user", "content": render_action(turn["action"], render, lang)})
        messages.append({"role": "assistant", "content": turn["line"]})
    messages.append({"role": "user", "content": render_action(case["action"], render, lang)})

    budget = render["budget"]
    if estimate_tokens(system) > budget["system_tokens"]:
        raise PromptTooLong(f"system message {estimate_tokens(system)} > {budget['system_tokens']}")
    total = sum(estimate_tokens(m["content"]) for m in messages)
    if total > budget["total_tokens"]:
        raise PromptTooLong(f"prompt {total} > {budget['total_tokens']}")
    return messages


def render_envelope(head: Mapping[str, Any]) -> str:
    """Serialise ``{"stance", "reveal", "line"}`` as the assistant turn: ``json.dumps(..., ensure_ascii=False)``, keys in that order."""
    return json.dumps({k: head[k] for k in HEAD_KEYS}, ensure_ascii=False)


def validate_head(head: object, *, stance_set: Iterable[str], permitted_ids: Iterable[str]) -> bool:
    """True iff ``head`` is a dict with exactly the keys stance/reveal/line, stance in the set, reveal a list of permitted ids, non-empty line."""
    if not isinstance(head, dict) or set(head) != set(HEAD_KEYS):
        return False
    reveal = head["reveal"]
    line = head["line"]
    return (
        head["stance"] in set(stance_set)
        and isinstance(reveal, list)
        and all(isinstance(r, str) and r in set(permitted_ids) for r in reveal)
        and isinstance(line, str)
        and bool(line.strip())
    )
