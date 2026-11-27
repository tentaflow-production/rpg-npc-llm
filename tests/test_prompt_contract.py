"""M2 spec: rpgnpc.prompt against docs/prompt_contract.md (fixtures in tests/fixtures/prompt_contract, oracles in data/templates).

Each docstring names the contract line / roadmap line it checks. These tests are RED until M2 is built.
"""

from __future__ import annotations

import copy
import json
import re
import sys
import unicodedata
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rpgnpc import prompt

FIX = ROOT / "tests" / "fixtures" / "prompt_contract"
TPL = ROOT / "data" / "templates"
FIXTURES = sorted(FIX.glob("*.json"))
CASE_KEYS = ["npc", "scene", "slot", "scenario_id", "language", "belief_bucket", "clue_ids_permitted",
             "withheld_topics", "stance_set", "history", "action"]


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def case_of(doc: dict) -> dict:
    return {k: doc[k] for k in CASE_KEYS}


def norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def secrets() -> list[dict]:
    return yaml.safe_load((ROOT / "data" / "secrets.yaml").read_text(encoding="utf-8"))["secrets"]


def hits(text: str, secret: dict) -> bool:
    n = norm(text)
    return any((re.search(p[3:], n) if p.startswith("re:") else p in n) for p in secret["patterns"])


# ---------- oracle self-consistency (these pass without any implementation; they guard the frozen data) ----------

def test_fixture_set_is_complete() -> None:
    """Contract 'Fixture': files carry every contract key plus valid_heads/invalid_heads/expected_messages; all five NPCs appear."""
    assert len(FIXTURES) >= 6
    for p in FIXTURES:
        doc = load(p)
        for k in CASE_KEYS + ["expected_messages", "valid_heads", "invalid_heads"]:
            assert k in doc, f"{p.name} lacks {k}"
        assert doc["valid_heads"] and doc["invalid_heads"]
    assert {load(p)["npc"] for p in FIXTURES} == {"Aurelia", "Bohan", "Cecily", "Dorian", "Elwin"}


def test_oracle_texts_never_match_a_secret_pattern() -> None:
    """Invariant 1/2 + secrets.yaml header: clue texts, belief texts and cards must not contain a leak pattern, or a permitted line would count as a leak."""
    texts = list(yaml.safe_load((TPL / "clues.yaml").read_text(encoding="utf-8"))["clues"].values())
    texts += list(yaml.safe_load((TPL / "beliefs.yaml").read_text(encoding="utf-8"))["buckets"].values())
    texts += list(yaml.safe_load((TPL / "cards.yaml").read_text(encoding="utf-8"))["cards"].values())
    for s in secrets():
        assert not hits(s["topic"], s), f"topic label of {s['id']} is itself a pattern hit"
        for t in texts:
            assert not hits(t, s), f"{s['id']} matches oracle text {t!r}"


def test_fixture_topics_exist_in_secrets_and_belong_to_the_npc() -> None:
    """Invariant 1: <withheld topics> are labels from data/secrets.yaml.topic of a secret that NPC holds."""
    by_topic = {s["topic"]: s for s in secrets()}
    for p in FIXTURES:
        doc = load(p)
        for label in doc["withheld_topics"]:
            assert label in by_topic, f"{p.name}: unknown topic label {label}"
            assert by_topic[label]["npc"] == doc["npc"], f"{p.name}: {label} is not held by {doc['npc']}"


def test_stances_oracle_matches_contract_text() -> None:
    """Invariant 6: 'Cecily has no `hostile`, Dorian has `charming | probing | cold`'; fixtures' stance sets are subsets."""
    st = yaml.safe_load((TPL / "stances.yaml").read_text(encoding="utf-8"))["stances"]
    assert "hostile" not in st["Cecily"]
    assert set(st["Dorian"]) == {"charming", "probing", "cold"}
    for p in FIXTURES:
        doc = load(p)
        assert doc["stance_set"] and set(doc["stance_set"]) <= set(st[doc["npc"]])


# ---------- the renderer (red until built) ----------

@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_render_matches_fixture(path: Path) -> None:
    """Contract 'Fixture' + 'Message layout': render_messages(case) == expected_messages exactly (system, history pairs, current action)."""
    doc = load(path)
    assert prompt.render_messages(case_of(doc), TPL) == doc["expected_messages"]


def test_history_is_cut_to_the_last_six_turns() -> None:
    """Contract layout 'up to the last 6 turns' / invariant 5 'history <= 6 turns' (render.yaml max_history_turns)."""
    doc = load(FIX / "aurelia_rooftop_d3e_long_history.json")
    assert len(doc["history"]) == 7
    msgs = prompt.render_messages(case_of(doc), TPL)
    assert len(msgs) == 1 + 2 * 6 + 1
    assert "第1句回答" not in json.dumps(msgs, ensure_ascii=False)
    assert "第7句回答" in json.dumps(msgs, ensure_ascii=False)


def test_withheld_content_is_never_in_the_prompt() -> None:
    """Invariant 1: 'its content is never in the prompt': for every withheld topic the secret's patterns do not occur in the system message."""
    by_topic = {s["topic"]: s for s in secrets()}
    for p in FIXTURES:
        doc = load(p)
        system = prompt.render_messages(case_of(doc), TPL)[0]["content"]
        for label in doc["withheld_topics"]:
            assert label in system
            assert not hits(system, by_topic[label]), f"{p.name}: content of {label} leaked into the prompt"


def test_knowledge_holds_exactly_the_permitted_ids() -> None:
    """Invariant 1: <knowledge> holds exactly clue_ids_permitted with canonical text; no other clue text appears."""
    clues = yaml.safe_load((TPL / "clues.yaml").read_text(encoding="utf-8"))["clues"]
    for p in FIXTURES:
        doc = load(p)
        system = prompt.render_messages(case_of(doc), TPL)[0]["content"]
        for cid, text in clues.items():
            assert (f"[{cid}] {text}" in system) == (cid in doc["clue_ids_permitted"]), f"{p.name}: {cid}"


def test_empty_sections_are_omitted_not_blank() -> None:
    """Layout: no permitted clue -> no <knowledge> line; no withheld topic -> no <withheld topics> line; never an empty line."""
    doc = load(FIX / "dorian_street_d3m.json")
    c = case_of(doc)
    c["withheld_topics"] = []
    system = prompt.render_messages(c, TPL)[0]["content"]
    assert "你会回避这些话题" not in system
    assert "\n\n" not in system and not system.endswith("\n")
    assert not re.search(r"^\[\w+\] ", system, re.MULTILINE)


def test_scenario_id_changes_only_the_scene_card() -> None:
    """D19 / roadmap goal (2): 'alternative scenarios are different prompts': the two Bohan fixtures differ in exactly the scene-card line."""
    a = prompt.render_messages(case_of(load(FIX / "bohan_office_d4m.json")), TPL)
    b = prompt.render_messages(case_of(load(FIX / "bohan_office_d4m_alt_scenario.json")), TPL)
    assert a != b
    la, lb = a[0]["content"].split("\n"), b[0]["content"].split("\n")
    diff = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    assert len(la) == len(lb) and diff == [1]
    assert "scenario_id: canon_default" in la[1] and "scenario_id: alt_c3_one_scene_per_slot" in lb[1]
    assert a[1:] == b[1:]


def test_scene_slot_and_scenario_are_always_present() -> None:
    """Invariants 4 + D19: scene name, 'Day N <slot label>' and scenario_id are in every system message; M/A/E = 早上/下午/晚上."""
    labels = {"M": "早上", "A": "下午", "E": "晚上"}
    for p in FIXTURES:
        doc = load(p)
        system = prompt.render_messages(case_of(doc), TPL)[0]["content"]
        day, slot = int(doc["slot"][1:-1]), doc["slot"][-1]
        assert f"Day {day} {labels[slot]}" in system
        assert f"scenario_id: {doc['scenario_id']}" in system


def test_belief_line_uses_the_bucket_text_never_the_bare_label() -> None:
    """Invariant 2: the belief line carries the bucket's canonical text; the bucket id itself never appears in the prompt."""
    buckets = yaml.safe_load((TPL / "beliefs.yaml").read_text(encoding="utf-8"))["buckets"]
    for p in FIXTURES:
        doc = load(p)
        system = prompt.render_messages(case_of(doc), TPL)[0]["content"]
        assert buckets[doc["belief_bucket"]] in system
        assert doc["belief_bucket"] not in system


def test_action_set_is_closed() -> None:
    """Invariant 3: 'there is no free-text player line, ever': an unknown action type, a missing argument or a bare string is rejected."""
    base = case_of(load(FIX / "dorian_street_d3m.json"))
    for bad in [{"type": "say", "text": "你好"}, {"type": "small_talk"}, {"type": "buy"}, "你好"]:
        c = copy.deepcopy(base)
        c["action"] = bad
        with pytest.raises((ValueError, KeyError, TypeError)):
            prompt.render_messages(c, TPL)


def test_stance_set_must_fit_the_npc() -> None:
    """Invariant 6: Cecily with `hostile`, Dorian with `warm`, or an empty/unknown stance set cannot be rendered."""
    cec = case_of(load(FIX / "cecily_bistro_d2e.json"))
    dor = case_of(load(FIX / "dorian_street_d3m.json"))
    for npc_case, bad in [(cec, ["warm", "hostile"]), (dor, ["charming", "warm"]), (dor, []), (cec, ["furious"])]:
        c = copy.deepcopy(npc_case)
        c["stance_set"] = bad
        with pytest.raises(ValueError):
            prompt.render_messages(c, TPL)


def test_unknown_clue_id_or_belief_bucket_is_rejected() -> None:
    """Invariant 7: clue ids are the case bible's nodes; an id with no canonical text cannot be shown (no invented facts)."""
    base = case_of(load(FIX / "bohan_office_d4m.json"))
    c1 = copy.deepcopy(base)
    c1["clue_ids_permitted"] = ["g1", "zz9"]
    c2 = copy.deepcopy(base)
    c2["belief_bucket"] = "made_up"
    for c in (c1, c2):
        with pytest.raises((ValueError, KeyError)):
            prompt.render_messages(c, TPL)


def test_language_other_than_zh_is_refused_until_templates_exist() -> None:
    """Q19 answer (bilingual) + Q36 default: render.yaml lists only `zh`; `en` raises UnsupportedLanguage rather than rendering Chinese text under an English tag."""
    c = case_of(load(FIX / "dorian_street_d3m.json"))
    c["language"] = "en"
    with pytest.raises(prompt.UnsupportedLanguage):
        prompt.render_messages(c, TPL)


def test_token_estimate_formula() -> None:
    """Invariant 5 helper: estimate = non-ASCII chars + ceil(ASCII chars / 3) (conservative for Qwen)."""
    assert prompt.estimate_tokens("") == 0
    assert prompt.estimate_tokens("你好") == 2
    assert prompt.estimate_tokens("a") == 1
    assert prompt.estimate_tokens("abcdef") == 2
    assert prompt.estimate_tokens("你好abcd") == 2 + 2


def test_system_budget_is_enforced() -> None:
    """Invariant 5: 'system <= 600 tokens': a system message over budget raises PromptTooLong."""
    c = case_of(load(FIX / "dorian_street_d3m.json"))
    c["withheld_topics"] = ["月之石的下落"] * 150          # 150 * 6 chars + separators >> 600 tokens
    with pytest.raises(prompt.PromptTooLong):
        prompt.render_messages(c, TPL)


def test_total_budget_is_enforced() -> None:
    """Invariant 5: 'total <= 1024': six long history turns push the prompt over budget although the system message fits."""
    c = case_of(load(FIX / "dorian_street_d3m.json"))
    c["history"] = [{"action": {"type": "silent"}, "line": "好" * 160} for _ in range(6)]   # 6 * (14 + 160) > 1024 alone
    with pytest.raises(prompt.PromptTooLong):
        prompt.render_messages(c, TPL)


def test_every_fixture_fits_the_budget() -> None:
    """Invariant 5: the pinned fixtures are within 600 / 1024 by the pinned estimate."""
    for p in FIXTURES:
        msgs = load(p)["expected_messages"]
        assert prompt.estimate_tokens(msgs[0]["content"]) <= 600
        assert sum(prompt.estimate_tokens(m["content"]) for m in msgs) <= 1024


# ---------- the envelope (Q18 option 2, invariant 6) ----------

@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_valid_and_invalid_heads(path: Path) -> None:
    """Fixture 'valid_heads / invalid_heads': the validator accepts every valid head and rejects every invalid one
    (bad stance, reveal id outside <knowledge>, empty line, non-list reveal, missing/extra key, Cecily `hostile`, Dorian `warm`)."""
    doc = load(path)
    for h in doc["valid_heads"]:
        assert prompt.validate_head(h, stance_set=doc["stance_set"], permitted_ids=doc["clue_ids_permitted"]) is True, h
    for h in doc["invalid_heads"]:
        assert prompt.validate_head(h, stance_set=doc["stance_set"], permitted_ids=doc["clue_ids_permitted"]) is False, h


def test_validate_head_rejects_non_dicts() -> None:
    """Invariant 6: a bare line, a list or None is not an envelope."""
    for h in ["好的。", ["warm"], None, 3]:
        assert prompt.validate_head(h, stance_set=["warm"], permitted_ids=[]) is False


def test_envelope_serialisation_is_pinned() -> None:
    """Contract last line: assistant turn is one JSON object {"stance", "reveal", "line"}, Chinese kept unescaped, key order fixed."""
    head = {"line": "你好。", "reveal": ["g1"], "stance": "warm"}
    assert prompt.render_envelope(head) == '{"stance": "warm", "reveal": ["g1"], "line": "你好。"}'
    assert json.loads(prompt.render_envelope(head)) == head
