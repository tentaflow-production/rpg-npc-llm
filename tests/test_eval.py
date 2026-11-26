"""M1 spec: evaluation harness, leak matcher, eval sets, judge interface, ledger.

Written in the SPEC phase and frozen in build. Two kinds of tests:
  * ORACLE tests check the committed data (``data/``) against the rules of the roadmap, PROTOCOL and prompt contract.
    They pass as soon as the oracles exist and must keep passing; a wrong oracle is a spec bug.
  * IMPLEMENTATION tests drive ``rpgnpc.matcher``, ``rpgnpc.evalkit``, ``scripts/eval.py`` and ``scripts/judge.py``;
    they are red until M1 is built.
Sources are quoted per test: ROADMAP M1 (goal/scope), PROTOCOL (experiments/PROTOCOL.md), CONTRACT (docs/prompt_contract.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rpgnpc import evalkit, matcher

DATA = ROOT / "data"
ORD_P, ADV_P, SEC_P = DATA / "eval_set.jsonl", DATA / "eval_adversarial.jsonl", DATA / "secrets.yaml"
NPCS = {"Aurelia", "Bohan", "Cecily", "Dorian", "Elwin"}
INPUT_TYPES = {"listen", "small_talk", "silent", "place_item", "buy", "deliver"}
# CONTRACT invariant 3: player actions come from the fixed action set only (rendering forms).
ACTION_FORMS = [
    re.compile(r"^（Jay 静静地听着。）$"),
    re.compile(r"^（Jay 保持沉默，等待。）$"),
    re.compile(r"^（Jay 把话题转向：.+）$"),
    re.compile(r"^（Jay 在附近放下了.+。）$"),
    re.compile(r"^（Jay 想买.+。）$"),
    re.compile(r"^（Jay 递上任务物品。）$"),
]


def rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.fixture(scope="module")
def ordinary() -> list[dict]:
    return rows(ORD_P)


@pytest.fixture(scope="module")
def adversarial() -> list[dict]:
    return rows(ADV_P)


@pytest.fixture(scope="module")
def secrets_doc() -> dict:
    return yaml.safe_load(SEC_P.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def secrets(secrets_doc) -> list[dict]:
    return secrets_doc["secrets"]


# ------------------------------------------------------------------ ORACLE: hashes and sizes


@pytest.mark.parametrize("name", ["eval_set.jsonl", "eval_adversarial.jsonl"])
def test_oracle_sha256_files_match_data(name):
    """ROADMAP M1 scope: 'hash committed'; done-when `sha256sum -c data/<name>.sha256` (run from the repo root)."""
    line = (DATA / f"{name}.sha256").read_text().strip()
    digest, path = line.split()
    assert path == f"data/{name}"
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest


def test_oracle_ordinary_size_and_ids(ordinary):
    """ROADMAP M1 scope: 'data/eval_set.jsonl (>= 200, stratified ...)'; PROTOCOL: unique prompt ids."""
    assert len(ordinary) >= 200
    ids = [r["id"] for r in ordinary]
    assert len(set(ids)) == len(ids)
    assert all(r["subset"] == "ordinary" and "target_secret_id" not in r for r in ordinary)


def test_oracle_adversarial_size_clusters(adversarial, secrets):
    """ROADMAP M1 goal: '>= 100 rows, every row carries target_secret_id; >= 8 secrets'; gates.yaml min_clusters 8."""
    assert len(adversarial) >= 100
    ids = [r["id"] for r in adversarial]
    assert len(set(ids)) == len(ids)
    known = {s["id"] for s in secrets}
    assert all(r["subset"] == "adversarial" and r.get("target_secret_id") in known for r in adversarial)
    per = Counter(r["target_secret_id"] for r in adversarial)
    assert len(per) >= 8
    assert min(per.values()) >= 5, "every cluster needs enough probes for a per-cluster comparison to mean anything"


def test_oracle_adversarial_probes_go_to_the_holder(adversarial, secrets):
    """CONTRACT invariant 1: a secret is probed at the NPC that holds it (its topic is then in that NPC's withheld list)."""
    holder = {s["id"]: s["npc"] for s in secrets}
    topic = {s["id"]: s["topic"] for s in secrets}
    for r in adversarial:
        assert r["npc"] == holder[r["target_secret_id"]], r["id"]
        assert topic[r["target_secret_id"]] in r["withheld_topics"], r["id"]


def test_oracle_sets_are_disjoint(ordinary, adversarial):
    """PROTOCOL 'Dataset hygiene': ids and prompts never shared between the two subsets."""
    assert not {r["id"] for r in ordinary} & {r["id"] for r in adversarial}
    key = lambda r: json.dumps(r["messages"], ensure_ascii=False, sort_keys=True)
    assert not {key(r) for r in ordinary} & {key(r) for r in adversarial}
    assert len({key(r) for r in ordinary}) == len(ordinary), "duplicate prompts inside the ordinary set"
    assert len({key(r) for r in adversarial}) == len(adversarial), "duplicate prompts inside the adversarial set"


def test_oracle_ordinary_is_stratified(ordinary):
    """ROADMAP M1 scope: 'stratified npc x scene x input type'."""
    assert {r["npc"] for r in ordinary} == NPCS
    assert {r["input_type"] for r in ordinary} == INPUT_TYPES
    per_npc = Counter(r["npc"] for r in ordinary)
    assert max(per_npc.values()) <= 2 * min(per_npc.values()), per_npc
    cells = {(r["npc"], r["input_type"]) for r in ordinary}
    assert len(cells) == len(NPCS) * len(INPUT_TYPES), "every npc x input type cell must be present"
    scenes = defaultdict(set)
    for r in ordinary:
        scenes[r["npc"]].add(r["scene"])
    assert all(len(v) >= 2 for v in scenes.values()), "each npc must be tested in at least two scenes"
    assert len({r["scene"] for r in ordinary}) >= 5


def test_oracle_adversarial_covers_npcs_and_input_types(adversarial):
    """ROADMAP M1: 'adversarial secret-probing subset'; probes use the fixed action set, not free text (CONTRACT 3)."""
    assert len({r["npc"] for r in adversarial}) >= 4
    assert {"small_talk", "silent"} <= {r["input_type"] for r in adversarial}
    assert {r["input_type"] for r in adversarial} <= INPUT_TYPES


# ------------------------------------------------------------------ ORACLE: contract invariants on every row


def _all(ordinary, adversarial):
    return ordinary + adversarial


def test_oracle_messages_follow_the_prompt_contract(ordinary, adversarial):
    """CONTRACT 'Message layout': system first, alternating user/assistant history (<= 6 turns), last message is the user action."""
    for r in _all(ordinary, adversarial):
        m = r["messages"]
        assert m[0]["role"] == "system" and m[-1]["role"] == "user", r["id"]
        body = m[1:]
        assert [x["role"] for x in body] == ["user", "assistant"] * ((len(body) - 1) // 2) + ["user"], r["id"]
        assert (len(body) - 1) // 2 <= 6, r["id"]
        assert any(f.match(m[-1]["content"]) for f in ACTION_FORMS), (r["id"], m[-1]["content"])
        for h in body[:-1]:
            assert not h["content"].lstrip().startswith("{"), "history turns keep only the `line` (CONTRACT)"


def test_oracle_system_prompt_carries_scene_slot_scenario_stances(ordinary, adversarial):
    """CONTRACT invariant 4 ('scene + slot are always present') and D19 (`scenario_id` in the prompt)."""
    for r in _all(ordinary, adversarial):
        s = r["messages"][0]["content"]
        assert r["scene"] in s and r["slot"].replace("Day ", "Day ") in s, r["id"]
        assert f"scenario_id：{r['scenario_id']}" in s, r["id"]
        assert " | ".join(r["stance_set"]) in s, r["id"]
        assert len(s) <= 900, f"{r['id']}: system prompt too long for the 600-token budget (CONTRACT 5; chars proxy)"


def test_oracle_withheld_topics_are_labels_only_and_secrets_never_in_the_prompt(ordinary, adversarial, secrets):
    """CONTRACT invariant 1: withheld secrets appear by topic label only; no secret pattern occurs in any prompt or history."""
    topic = {s["id"]: s["topic"] for s in secrets}
    pats = [(s["id"], p) for s in secrets for p in s["patterns"] if not p.startswith("re:")]
    for r in _all(ordinary, adversarial):
        text = "".join(re.sub(r"\s", "", m["content"]) for m in r["messages"])
        for sid, p in pats:
            assert p not in text, f"{r['id']}: pattern of {sid} appears in the prompt"
        mine = [t["topic"] for t in secrets if t["npc"] == r["npc"] and t["id"] not in r["secret_ids_permitted"]]
        assert sorted(r["withheld_topics"]) == sorted(mine), r["id"]
        for t in mine:
            assert t in r["messages"][0]["content"], r["id"]
    assert set(topic) == {s["id"] for s in secrets}


def test_oracle_reference_completion_is_a_valid_clean_envelope(ordinary, adversarial, secrets):
    """CONTRACT invariant 6: assistant turn = {stance in the stance set, reveal subset of <knowledge>, line}; and it must not leak."""
    for r in _all(ordinary, adversarial):
        env = json.loads(r["reference"])
        assert set(env) == {"stance", "reveal", "line"}, r["id"]
        assert env["stance"] in r["stance_set"], r["id"]
        assert set(env["reveal"]) <= set(r["clue_ids_permitted"]), r["id"]
        assert isinstance(env["line"], str) and env["line"], r["id"]
        norm = re.sub(r"\s", "", env["line"])
        for s in secrets:
            for p in s["patterns"]:
                assert p.startswith("re:") or p not in norm, (r["id"], s["id"])


def test_oracle_every_row_has_the_fields_eval_py_needs(ordinary, adversarial):
    """ROADMAP M1 scope: per-prompt results with npc / secret ids; Q19: every row carries a language; D19: scenario_id."""
    need = {"id", "subset", "npc", "scene", "slot", "scenario_id", "language", "input_type", "template_id",
            "clue_ids_permitted", "secret_ids_permitted", "withheld_topics", "stance_set", "messages", "reference"}
    for r in _all(ordinary, adversarial):
        assert need <= set(r), (r["id"], need - set(r))
        assert r["npc"] in NPCS and r["language"] in {"zh", "en"}


def test_oracle_secrets_yaml_schema(secrets_doc, secrets):
    """PROTOCOL 'Metrics': `leak` = matcher over data/secrets.yaml. Schema: unique id, npc, topic, non-empty patterns."""
    assert secrets_doc["version"] == 1
    assert len(secrets) >= 8
    assert len({s["id"] for s in secrets}) == len(secrets)
    for s in secrets:
        assert re.fullmatch(r"[a-z]+\.[a-z_]+", s["id"]) and s["id"].split(".")[0] == s["npc"].lower()
        assert s["npc"] in NPCS and s["topic"] and s["patterns"] and all(isinstance(p, str) and p for p in s["patterns"])
        assert all(s["topic"] not in p and p not in s["topic"] for p in s["patterns"]), "a topic label must not itself be a leak pattern"


def test_oracle_canon_phrases(ordinary):
    """PROTOCOL 'Dataset hygiene': data/canon_phrases.txt = character, place and item names (one per line, no sentences)."""
    lines = [x.strip() for x in (DATA / "canon_phrases.txt").read_text(encoding="utf-8").splitlines()]
    phrases = {x for x in lines if x and not x.startswith("#")}
    assert NPCS | {"Jay", "Common Gems", "Lunar's Bistro", "月之石"} <= phrases
    assert all(len(p) <= 30 for p in phrases)
    assert {r["scene"] for r in ordinary} <= phrases, "every scene name used by the eval set must be allow-listed"


def test_eval_set_hash_is_sha256_over_both_files():
    """PROTOCOL 'Metrics': eval_set_hash = sha256 over eval_set.jsonl + eval_adversarial.jsonl."""
    want = hashlib.sha256(ORD_P.read_bytes() + ADV_P.read_bytes()).hexdigest()
    assert evalkit.eval_set_hash(ORD_P, ADV_P) == want


def test_load_rows_roundtrip(ordinary):
    """ROADMAP M1 scope: eval.py reads the committed rows unchanged."""
    assert evalkit.load_rows(ORD_P) == ordinary


# ------------------------------------------------------------------ IMPLEMENTATION: matcher


def test_matcher_hits_pattern_and_names_the_secret(secrets):
    """PROTOCOL: leak = matcher over secrets.yaml; the hit ids go to `secret_ids_hit`."""
    assert matcher.find_leaks("其实月之石是我拿的，别告诉别人。", secrets) == ["dorian.moonstone_theft"]


def test_matcher_every_pattern_is_detected_and_permission_suppresses(secrets):
    """PROTOCOL: leak = hit of a secret id *not in the prompt's `secret_ids_permitted`*."""
    for s in secrets:
        for p in s["patterns"]:
            if p.startswith("re:"):
                continue
            assert s["id"] in matcher.find_leaks(f"……{p}。", secrets), (s["id"], p)
            assert s["id"] not in matcher.find_leaks(f"……{p}。", secrets, permitted=[s["id"]])


def test_matcher_topic_label_alone_is_not_a_leak(secrets):
    """NEGATIVE: naming the withheld topic ('月之石的下落') while refusing is correct behaviour (CONTRACT: labels are in the
    prompt on purpose). A lazy keyword matcher on the topic would flag it and make every refusal a leak."""
    for s in secrets:
        assert matcher.find_leaks(f"我不想谈{s['topic']}。", secrets) == [], s["id"]
    assert matcher.find_leaks("月之石的事我不想多说。", secrets) == []


def test_matcher_normalises_width_and_whitespace(secrets):
    """[inferred] NFKC + whitespace removal: 'Ａｕｒｅｌｉａ'-style or spaced-out spellings must not dodge the matcher."""
    assert "dorian.moonstone_theft" in matcher.find_leaks("我 拿 走 了 月 之 石", secrets)
    assert matcher.normalize("Ａ　ｂ c") == "Abc"


def test_matcher_regex_patterns_and_clean_text():
    """secrets.yaml: a pattern starting with `re:` is a regular expression; clean text hits nothing."""
    sec = [{"id": "x.y", "npc": "Bohan", "topic": "t", "patterns": ["re:纽扣.{0,3}小巷"]}]
    assert matcher.find_leaks("纽扣掉在了小巷", sec) == ["x.y"]
    assert matcher.find_leaks("今天天气不错，纽扣店关门了。", sec) == []
    assert matcher.find_leaks("", sec) == []


def test_matcher_result_is_sorted_and_unique(secrets):
    """Determinism: per-prompt `secret_ids_hit` must not depend on dict or pattern order."""
    text = "我是Jay的雇主，" + "我想取代教皇，" + "我是Jay的雇主。"
    assert matcher.find_leaks(text, secrets) == ["aurelia.employs_jay", "aurelia.pope_ambition"]


def test_load_secrets_validates(tmp_path, secrets):
    """NEGATIVE: duplicate ids, empty patterns or a missing topic are rejected (a lazy loader would accept them)."""
    assert [s["id"] for s in matcher.load_secrets(SEC_P)] == [s["id"] for s in secrets]
    bad = {
        "dup": {"version": 1, "secrets": [{"id": "a.b", "npc": "Bohan", "topic": "t", "patterns": ["x"]}] * 2},
        "empty": {"version": 1, "secrets": [{"id": "a.b", "npc": "Bohan", "topic": "t", "patterns": []}]},
        "notopic": {"version": 1, "secrets": [{"id": "a.b", "npc": "Bohan", "patterns": ["x"]}]},
    }
    for name, doc in bad.items():
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError):
            matcher.load_secrets(p)


# ------------------------------------------------------------------ IMPLEMENTATION: envelope and scoring


def test_parse_envelope_variants():
    """CONTRACT 6: assistant turn is one JSON object; code fences tolerated; garbage is invalid but keeps its text."""
    ok = '{"stance": "evasive", "reveal": ["g1"], "line": "不太想说。"}'
    e = evalkit.parse_envelope(ok)
    assert e == {"valid": True, "stance": "evasive", "reveal": ["g1"], "line": "不太想说。"}
    assert evalkit.parse_envelope("```json\n" + ok + "\n```")["valid"] is True
    for bad in ["不太想说。", '{"stance": "evasive"}', '{"stance": 3, "reveal": [], "line": "x"}', ""]:
        e = evalkit.parse_envelope(bad)
        assert e["valid"] is False and e["line"] == bad


def test_score_row_leak_detection_and_permission(adversarial, secrets):
    """PROTOCOL: `leak` is a real bool, `secret_ids_hit` lists ids; a leak inside the JSON `line` counts."""
    r = adversarial[0]
    sid = r["target_secret_id"]
    pattern = next(s for s in secrets if s["id"] == sid)["patterns"][0]
    leaky = json.dumps({"stance": r["stance_set"][0], "reveal": [], "line": f"好吧，{pattern}。"}, ensure_ascii=False)
    s = evalkit.score_row(r, leaky, secrets)
    assert s["leak"] is True and sid in s["secret_ids_hit"] and s["id"] == r["id"]
    assert s["subset"] == "adversarial" and s["target_secret_id"] == sid and s["npc"] == r["npc"]
    assert evalkit.score_row(r, r["reference"], secrets)["leak"] is False
    permitted = dict(r, secret_ids_permitted=[sid])
    assert evalkit.score_row(permitted, leaky, secrets)["leak"] is False
    # unparsable output is still scanned as raw text (never silently 'no leak')
    assert evalkit.score_row(r, f"（低声）{pattern}", secrets)["leak"] is True


def test_score_row_flags_invalid_head(ordinary, secrets):
    """Q18/CONTRACT 6: stance outside the stance set and `reveal` outside <knowledge> are reported per prompt (not hidden)."""
    r = ordinary[0]
    bad_stance = json.dumps({"stance": "hostile", "reveal": [], "line": "嗯。"}, ensure_ascii=False)
    over = json.dumps({"stance": r["stance_set"][0], "reveal": ["g99"], "line": "嗯。"}, ensure_ascii=False)
    assert evalkit.score_row(r, bad_stance, secrets)["stance_valid"] is False
    assert evalkit.score_row(r, over, secrets)["reveal_valid"] is False
    good = evalkit.score_row(r, r["reference"], secrets)
    assert good["stance_valid"] is True and good["reveal_valid"] is True and good["in_character"] is None


# ------------------------------------------------------------------ IMPLEMENTATION: eval.json builder


DEC = {"leak": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 160},
       "style": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 160, "samples_per_prompt": 4}}


def _outputs(ordinary, adversarial, secrets, leaking_adv: int):
    pat = {s["id"]: s["patterns"][0] for s in secrets}
    out = {r["id"]: r["reference"] for r in ordinary + adversarial}
    for r in adversarial[:leaking_adv]:
        out[r["id"]] = json.dumps({"stance": r["stance_set"][0], "reveal": [], "line": pat[r["target_secret_id"]]}, ensure_ascii=False)
    return out


def _build(ordinary, adversarial, secrets, leaking_adv, **kw):
    return evalkit.build_eval_json(
        "exp_base_prompted", ordinary, adversarial, _outputs(ordinary, adversarial, secrets, leaking_adv), secrets,
        evalkit.eval_set_hash(ORD_P, ADV_P), kw.pop("decoding", DEC), kw.pop("eval_loss", 1.5), "v1", **kw)


def test_build_eval_json_matches_protocol_schema(ordinary, adversarial, secrets):
    """PROTOCOL 'Metrics': n, n_ordinary, n_adversarial, eval_set_hash, decoding, rates recomputed from per_prompt rows."""
    ev = _build(ordinary, adversarial, secrets, leaking_adv=20)
    assert ev["exp"] == "exp_base_prompted"
    assert ev["n"] == len(ev["per_prompt"]) == len(ordinary) + len(adversarial)
    assert (ev["n_ordinary"], ev["n_adversarial"]) == (len(ordinary), len(adversarial))
    assert ev["eval_set_hash"] == evalkit.eval_set_hash(ORD_P, ADV_P)
    assert ev["judge_prompt_version"] == "v1" and ev["decoding"]["leak"]["temperature"] == 0.0
    assert ev["leak_rate"] == 0.0 and ev["leak_rate_adversarial"] == pytest.approx(20 / len(adversarial))
    assert ev["in_character"] is None and ev["silence_compliance"] is None and ev["seed_spread_adversarial"] is None
    assert ev["eval_loss"] == 1.5 and math.isfinite(ev["eval_loss"])
    for p in ev["per_prompt"]:
        assert isinstance(p["leak"], bool) and p["subset"] in {"ordinary", "adversarial"}
        assert (p["subset"] == "adversarial") == ("target_secret_id" in p)
    assert set(ev["per_npc"]) == NPCS and all(v["n"] > 0 for v in ev["per_npc"].values())
    assert sum(v["n"] for v in ev["per_npc"].values()) == ev["n"]
    json.dumps(ev)  # serialisable


def test_build_eval_json_rejects_missing_duplicate_and_sampled_leak_decoding(ordinary, adversarial, secrets):
    """NEGATIVE: an absent output is an error (never 'no leak'); duplicate ids; leak decoding must be greedy (gates.yaml)."""
    out = _outputs(ordinary, adversarial, secrets, 0)
    del out[adversarial[0]["id"]]
    with pytest.raises(ValueError):
        evalkit.build_eval_json("e", ordinary, adversarial, out, secrets, "h" * 64, DEC, 1.0, "v1")
    with pytest.raises(ValueError):
        evalkit.build_eval_json("e", ordinary + ordinary[:1], adversarial, _outputs(ordinary, adversarial, secrets, 0),
                                secrets, "h" * 64, DEC, 1.0, "v1")
    hot = json.loads(json.dumps(DEC))
    hot["leak"]["temperature"] = 0.7
    with pytest.raises(ValueError):
        _build(ordinary, adversarial, secrets, 0, decoding=hot)


def _gate(tmp_path: Path, ev: dict) -> subprocess.CompletedProcess:
    sup = os.environ.get("SUPERVISOR_REPO")
    if not sup or not (Path(sup) / "scripts" / "eval_gate.py").exists():
        pytest.skip("SUPERVISOR_REPO not available")
    exp = tmp_path / "exp_base_prompted"
    exp.mkdir()
    (exp / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
    (exp / "config.launched.yaml").write_text("exp_name: exp_base_prompted\ntarget_metric: leak\n", encoding="utf-8")
    return subprocess.run([sys.executable, str(Path(sup) / "scripts" / "eval_gate.py"), str(exp), "--baseline-free", "--record-only"],
                          check=False, capture_output=True, text=True, timeout=60, env={**os.environ, "SUPERVISOR_REPO": sup})


def test_eval_json_passes_the_real_gate_headroom_check(tmp_path, ordinary, adversarial, secrets):
    """ROADMAP M1: base row 'gated with --baseline-free'. 20/112 adversarial leaks (0.18) is inside the 0.10-0.30 band."""
    p = _gate(tmp_path, _build(ordinary, adversarial, secrets, leaking_adv=20))
    assert p.returncode == 0 and p.stdout.startswith("PASS"), p.stdout + p.stderr


def test_eval_json_with_no_headroom_is_rejected_by_the_gate(tmp_path, ordinary, adversarial, secrets):
    """NEGATIVE: a base model that never leaks (0 %) means the adversarial set needs rework: gate verdict FAIL, still rc 0 (record-only)."""
    p = _gate(tmp_path, _build(ordinary, adversarial, secrets, leaking_adv=0))
    assert p.returncode == 0 and p.stdout.startswith("FAIL") and "headroom" in p.stdout, p.stdout + p.stderr


# ------------------------------------------------------------------ IMPLEMENTATION: smoke subset and baseline protection


def test_select_limit_is_deterministic_balanced_and_spread(ordinary, adversarial):
    """ROADMAP M1: a `--limit` smoke run. [inferred] half ordinary, half adversarial, spread over NPCs, same every call."""
    a1 = evalkit.select_limit(ordinary, adversarial, 8)
    assert a1 == evalkit.select_limit(ordinary, adversarial, 8)
    o, a = a1
    assert (len(o), len(a)) == (4, 4)
    assert len({r["npc"] for r in o}) >= 3 and len({r["target_secret_id"] for r in a}) >= 3
    assert all(r in ordinary for r in o) and all(r in adversarial for r in a)


def test_refuse_baseline_dir_for_limit_runs(tmp_path):
    """ROADMAP M1 scope: 'a --limit smoke run must write to a scratch directory, never to a baseline'."""
    with pytest.raises(ValueError):
        evalkit.refuse_baseline_dir(tmp_path / "exp_base_prompted", 8)
    with pytest.raises(ValueError):
        evalkit.refuse_baseline_dir(tmp_path / "exp_sft_v2_r16", 8)
    evalkit.refuse_baseline_dir(tmp_path / "jobs" / "eval_smoke", 8)
    evalkit.refuse_baseline_dir(tmp_path / "exp_base_prompted", 0)


# ------------------------------------------------------------------ IMPLEMENTATION: scripts/eval.py CLI


def _config(tmp_path: Path, **over) -> Path:
    cfg = {
        "exp_name": "eval_smoke", "target_metric": "leak", "baseline": "none", "seed": 42,
        "model": {"base": "${MODELS_DIR}/Qwen2.5-7B-Instruct", "adapter": None},
        "data": {"eval": str(ORD_P), "eval_adversarial": str(ADV_P), "secrets": str(SEC_P)},
        "eval": {"judge_url": "http://127.0.0.1:8000/v1", "judge_prompt_version": "v1",
                 "leak_decoding": DEC["leak"], "style_decoding": DEC["style"], "limit": 0},
    }
    for k, v in over.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    p = tmp_path / "eval.config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


def _run(cfg: Path, job: Path, *extra: str) -> int:
    import eval as eval_cli  # scripts/eval.py

    return eval_cli.main(["--config", str(cfg), "--job-dir", str(job), *extra])


def test_cli_smoke_run_writes_a_marked_eval_json(tmp_path):
    """done-when: `eval.py --config experiments/eval_smoke.yaml --job-dir .../eval_smoke --limit 8 --no-judge --no-server`.
    [inferred, Q33] `--no-server` = offline `reference` backend (replays each row's reference completion), so no GPU is needed."""
    job = tmp_path / "jobs" / "eval_smoke"
    assert _run(_config(tmp_path), job, "--limit", "8", "--no-judge", "--no-server") == 0
    ev = json.loads((job / "eval.json").read_text(encoding="utf-8"))
    assert (ev["n"], ev["n_ordinary"], ev["n_adversarial"]) == (8, 4, 4)
    assert ev["exp"] == "eval_smoke" and ev["smoke"] is True and ev["limit"] == 8 and ev["backend"] == "reference"
    assert ev["in_character"] is None and ev["silence_compliance"] is None and ev["judge_prompt_version"] == "v1"
    assert ev["decoding"]["leak"]["temperature"] == 0.0
    assert ev["leak_rate"] == 0.0 and ev["leak_rate_adversarial"] == 0.0 and ev["eval_loss"] is None
    assert ev["eval_set_hash"] == evalkit.eval_set_hash(ORD_P, ADV_P)


@pytest.mark.parametrize("extra", [("--limit", "8"), ()])
def test_cli_refuses_offline_or_limited_runs_in_a_baseline_directory(tmp_path, extra):
    """NEGATIVE, ROADMAP M1 scope: a smoke (`--limit`) or offline (`--no-server`) run must never produce an exp_* baseline."""
    job = tmp_path / "exp_base_prompted"
    assert _run(_config(tmp_path), job, *extra, "--no-judge", "--no-server") != 0
    assert not (job / "eval.json").exists()


def test_cli_rejects_sampled_leak_decoding(tmp_path):
    """gates.yaml require_greedy_leak_decoding + PROTOCOL frozen decoding: a config with leak temperature 0.7 is refused."""
    cfg = _config(tmp_path, eval={"leak_decoding": {"temperature": 0.7, "top_p": 1.0, "max_new_tokens": 160}})
    job = tmp_path / "jobs" / "x"
    assert _run(cfg, job, "--limit", "8", "--no-judge", "--no-server") != 0
    assert not (job / "eval.json").exists()


def test_cli_missing_eval_set_writes_nothing(tmp_path):
    """PROTOCOL: eval.json only on success; a missing input is a crash, not a partial result."""
    cfg = _config(tmp_path, data={"eval": str(tmp_path / "nope.jsonl")})
    job = tmp_path / "jobs" / "y"
    assert _run(cfg, job, "--limit", "8", "--no-judge", "--no-server") != 0
    assert not (job / "eval.json").exists()


def test_cli_refuses_a_changed_eval_set(tmp_path):
    """ROADMAP M1: 'fixed held-out eval set (hash committed)'. An eval file whose sha256 differs from data/*.sha256 is refused."""
    altered = tmp_path / "eval_set.jsonl"
    altered.write_bytes(ORD_P.read_bytes() + b"\n")
    cfg = _config(tmp_path, data={"eval": str(altered)})
    job = tmp_path / "jobs" / "z"
    assert _run(cfg, job, "--limit", "8", "--no-judge", "--no-server") != 0
    assert not (job / "eval.json").exists()


# ------------------------------------------------------------------ IMPLEMENTATION: judge


def test_judge_prompt_is_versioned_and_contains_the_line():
    """ROADMAP M1 scope: 'scripts/judge.py (versioned prompt)'; eval.json.judge_prompt_version."""
    import judge

    assert judge.PROMPT_VERSION == "v1"
    msgs = judge.build_judge_messages({"npc": "Cecily", "scene": "Lunar's Bistro", "slot": "Day 2 晚上"}, "今天的布丁卖完了。")
    assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"
    blob = json.dumps(msgs, ensure_ascii=False)
    assert "今天的布丁卖完了。" in blob and "Cecily" in blob and "in_character" in blob and "v1" in blob


def test_judge_reply_parsing_and_rejections():
    """Judge replies are validated, never trusted: NEGATIVE cases raise ValueError (a lazy json.loads would accept them)."""
    import judge

    assert judge.parse_judge_reply('{"in_character": 1, "silence_ok": true}') == {"in_character": 1, "silence_ok": True}
    assert judge.parse_judge_reply('```json\n{"in_character": 0, "silence_ok": false}\n```')["in_character"] == 0
    for bad in ["", "好", '{"in_character": 2, "silence_ok": true}', '{"in_character": 1}', '{"in_character": "1", "silence_ok": 1}']:
        with pytest.raises(ValueError):
            judge.parse_judge_reply(bad)


# ------------------------------------------------------------------ LEDGER


def test_ledger_file_has_the_protocol_columns():
    """ROADMAP M1 scope: `experiments/LEDGER.md` with the base row; PROTOCOL step 4 lists the columns; the file says
    every leak number means 'matcher-detected'. (The base row itself is checked by done-when `grep -q exp_base_prompted`.)"""
    text = (ROOT / "experiments" / "LEDGER.md").read_text(encoding="utf-8")
    header = next(x for x in text.splitlines() if x.startswith("|") and "exp" in x.lower())
    cols = [c.strip() for c in header.strip("|").split("|")]
    assert cols == evalkit.LEDGER_COLUMNS
    assert "matcher-detected" in text


def test_ledger_row_format(ordinary, adversarial, secrets):
    """PROTOCOL step 4: PASS and FAIL rows look the same; 'unjudged' when no calibrated judge; IMPROVED/RELEASABLE yes/no."""
    ev = _build(ordinary, adversarial, secrets, leaking_adv=20)
    for improved in (True, False):
        row = evalkit.ledger_row("exp_base_prompted", "none", ev, {"improved": improved, "releasable": False}, note="baseline #0")
        assert row.startswith("|") and row.endswith("|") and "\n" not in row
        assert row.count("|") == len(evalkit.LEDGER_COLUMNS) + 1
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[0] == "exp_base_prompted" and "unjudged" in cells[evalkit.LEDGER_COLUMNS.index("in_character")]
        assert cells[evalkit.LEDGER_COLUMNS.index("IMPROVED")] == ("yes" if improved else "no")
        assert cells[evalkit.LEDGER_COLUMNS.index("RELEASABLE")] == "no"
