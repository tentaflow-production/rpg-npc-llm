"""M2 spec: scripts/validate_dataset.py and scripts/gen_dataset.py (PROTOCOL 'Dataset hygiene', 'Facts in context, never in weights (D19)').

Row schema (pinned here, [inferred] field names; one JSON object per line of `train.jsonl`, never committed):
  id, npc, scene (scene id), slot ("D2E"), scenario_id, language, player_input_type, topic, offscript (bool), template_id,
  source ("teacher"|"human"), messages (the rendered prompt per docs/prompt_contract.md + the final assistant envelope JSON string)
Dataset dir: `train.jsonl` + `manifest.json` {version, teacher_model, teacher_sha, template_sha, seed, decoding, counts{n, per_npc},
  filter_policy, train_sha256}.  Verdict (stdout, one JSON object): {"ok", "n", "metrics", "violations":[{"code", ...}]}, rc 0 iff ok.
Violation codes: too_few_rows, leak, imbalance, contamination, template_overlap, duplicate, missing_manifest, manifest_incomplete,
  manifest_mismatch, missing_field, language, offscript_low, offscript_inconsistent, bad_envelope, prompt_mismatch.
Contamination text of a row (pinned, [inferred]): every non-system message, taken SEGMENT BY SEGMENT (shingles never span two messages):
  assistant contents (history lines and the final `line`, not the envelope JSON scaffolding) and the ARGUMENTS of user actions;
  the fixed parts of the action strings (render.yaml `actions`) and the whole system message are excluded (identical by design).
  Allowlist phrases are removed LONGEST FIRST before shingling (a removed phrase is replaced by a separator, grams never span it).
RED until M2 is built (the stubs print a traceback and no verdict).
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
VALIDATE = ROOT / "scripts" / "validate_dataset.py"
GEN = ROOT / "scripts" / "gen_dataset.py"
FIX = ROOT / "tests" / "fixtures" / "prompt_contract"
TPL = ROOT / "data" / "templates"
NPC_FIXTURES = {"Cecily": "cecily_bistro_d2e", "Dorian": "dorian_street_d3m", "Bohan": "bohan_office_d4m",
                "Elwin": "elwin_task_center_d3a", "Aurelia": "aurelia_rooftop_d3e_long_history"}
OFFSCRIPT = yaml.safe_load((TPL / "topics.yaml").read_text(encoding="utf-8"))["offscript_topics"]
SCRIPTED = yaml.safe_load((TPL / "topics.yaml").read_text(encoding="utf-8"))["scripted_topics"]
INPUT_TYPES = yaml.safe_load((TPL / "topics.yaml").read_text(encoding="utf-8"))["player_input_types"]


# ---------------------------------------------------------------- helpers

def fx(npc: str) -> dict:
    return json.loads((FIX / f"{NPC_FIXTURES[npc]}.json").read_text(encoding="utf-8"))


def rnd_line(seed: int, n: int = 24) -> str:
    """Pseudo-random CJK text: no canon phrase, no secret pattern, pairwise 5-gram Jaccard ~ 0."""
    r = random.Random(seed)
    return "".join(chr(r.randrange(0x4E00, 0x9FA5)) for _ in range(n))


def envelope(npc: str, line: str, reveal: list[str] | None = None, stance: str | None = None) -> str:
    doc = fx(npc)
    head = {"stance": stance or doc["stance_set"][0], "reveal": reveal if reveal is not None else [], "line": line}
    return json.dumps(head, ensure_ascii=False)


def make_row(i: int, npc: str, line: str, *, offscript: bool, **over) -> dict:
    doc = fx(npc)
    topic = OFFSCRIPT[i % len(OFFSCRIPT)] if offscript else SCRIPTED[i % len(SCRIPTED)]
    row = {
        "id": f"r{i:04d}", "npc": npc, "scene": doc["scene"], "slot": doc["slot"], "scenario_id": doc["scenario_id"],
        "language": "zh", "player_input_type": INPUT_TYPES[i % len(INPUT_TYPES)], "topic": topic, "offscript": offscript,
        "template_id": f"t{i:04d}", "source": "teacher",
        "messages": doc["expected_messages"] + [{"role": "assistant", "content": envelope(npc, line)}],
    }
    row.update(over)
    return row


def make_rows(counts: dict[str, int], n_off: int) -> list[dict]:
    rows, i = [], 0
    for npc, k in counts.items():
        for _ in range(k):
            rows.append(make_row(i, npc, rnd_line(i), offscript=False))
            i += 1
    for r in rows[:n_off]:                       # first n_off rows are off-script
        r["offscript"] = True
        r["topic"] = OFFSCRIPT[int(r["id"][1:]) % len(OFFSCRIPT)]
    return rows


GOOD = {"Cecily": 5, "Dorian": 5, "Bohan": 5, "Elwin": 5, "Aurelia": 5}   # 25 rows


def write_dataset(d: Path, rows: list[dict], manifest: dict | None | bool = True, **manifest_over) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (d / "train.jsonl").write_text(body, encoding="utf-8")
    if manifest is not False:
        per_npc: dict[str, int] = {}
        for r in rows:
            per_npc[r["npc"]] = per_npc.get(r["npc"], 0) + 1
        m = {"version": "npc_sft_v2", "teacher_model": "Qwen2.5-32B-Instruct-AWQ", "teacher_sha": "0" * 12, "template_sha": "1" * 12,
             "seed": 1, "decoding": {"temperature": 0.8}, "counts": {"n": len(rows), "per_npc": per_npc},
             "filter_policy": {"matcher": True, "judge": False}, "train_sha256": hashlib.sha256(body.encode()).hexdigest()}
        m.update(manifest_over)
        (d / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return d


def write_eval(p: Path, rows: list[dict]) -> Path:
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


def eval_row(npc: str = "Bohan", line: str | None = None, history_line: str | None = None, template_id: str = "eval_t0") -> dict:
    row = make_row(9000, npc, line or rnd_line(9000), offscript=False, template_id=template_id, id="e0")
    if history_line is not None:
        asst = [m for m in row["messages"][:-1] if m["role"] == "assistant"]
        asst[0]["content"] = history_line
    return row


def validate(ds: Path, evals: list[Path], *extra: str, min_n: int = 25, **kw: str) -> tuple[int, dict]:
    args = [str(ds), "--min-n", str(min_n), "--max-leak-rate", kw.get("leak", "0.0"), "--max-imbalance", "2.0",
            "--contamination-against", *map(str, evals), "--allowlist", str(ROOT / "data" / "canon_phrases.txt"),
            "--require-manifest", "--require-field", "scenario_id", "--language", "zh", "--min-offscript", "0.4",
            "--secrets", str(ROOT / "data" / "secrets.yaml"), "--templates", str(TPL), *extra]
    r = subprocess.run([sys.executable, str(VALIDATE), *args], check=False, capture_output=True, text=True, timeout=120)
    try:
        verdict = json.loads(r.stdout)
    except json.JSONDecodeError:
        pytest.fail(f"no JSON verdict on stdout (rc={r.returncode}): {r.stdout!r} / {r.stderr[-400:]}")
    assert (r.returncode == 0) == verdict["ok"], "rc must be 0 iff ok"
    return r.returncode, verdict


def codes(verdict: dict) -> set[str]:
    return {v["code"] for v in verdict["violations"]}


@pytest.fixture
def evalf(tmp_path: Path) -> Path:
    return write_eval(tmp_path / "eval.jsonl", [eval_row()])


# ---------------------------------------------------------------- validator: positive

def test_good_dataset_passes_with_a_full_verdict(tmp_path: Path, evalf: Path) -> None:
    """Done-when line 3 (all flags): a clean 25-row, five-NPC, 40 %-off-script, deduped, uncontaminated dataset is ok; verdict has the pinned shape."""
    rc, v = validate(write_dataset(tmp_path / "ds", make_rows(GOOD, 10)), [evalf])
    assert rc == 0 and v["ok"] is True and v["violations"] == [] and v["n"] == 25
    for k in ("leak_rate", "imbalance", "offscript_fraction", "duplicates", "contamination_hits"):
        assert k in v["metrics"], k
    assert v["metrics"]["leak_rate"] == 0.0 and v["metrics"]["imbalance"] == 1.0
    assert v["metrics"]["offscript_fraction"] == pytest.approx(0.4)


def test_min_n_is_inclusive(tmp_path: Path, evalf: Path) -> None:
    """Roadmap: `--min-n $(cat data/MIN_N)`: n == min-n passes, n == min-n - 1 fails with too_few_rows."""
    ds = write_dataset(tmp_path / "ds", make_rows(GOOD, 10))
    assert validate(ds, [evalf], min_n=25)[0] == 0
    rc, v = validate(ds, [evalf], min_n=26)
    assert rc == 1 and codes(v) == {"too_few_rows"}


def test_min_n_and_language_files_hold_the_spec_values() -> None:
    """Q20 answer 'SFT baseline on what is reviewed (start 300)' and Q36 default: data/MIN_N = 300, data/LANGUAGE = zh (one value each)."""
    assert (ROOT / "data" / "MIN_N").read_text().strip() == "300"
    assert (ROOT / "data" / "LANGUAGE").read_text().strip() == "zh"


# ---------------------------------------------------------------- validator: leak (matcher over secrets.yaml)

def test_leaky_row_is_flagged_at_zero_tolerance(tmp_path: Path, evalf: Path) -> None:
    """PROTOCOL: 'drop rows the matcher flags'; `--max-leak-rate 0.0`: one first-person confession in a Dorian line fails."""
    rows = make_rows(GOOD, 10)
    rows[5]["messages"][-1]["content"] = envelope("Dorian", "没错，我偷走了月之石，你又能怎样。")
    rc, v = validate(write_dataset(tmp_path / "ds", rows), [evalf])
    assert rc == 1 and "leak" in codes(v)
    assert v["metrics"]["leak_rate"] == pytest.approx(1 / 25)
    assert any(x.get("id") == rows[5]["id"] for x in v["violations"] if x["code"] == "leak")


def test_leak_threshold_is_a_ceiling_not_zero_only(tmp_path: Path, evalf: Path) -> None:
    """`--max-leak-rate F`: 1/25 = 0.04 <= 0.05 passes; > 0.05 would not (the flag is honoured, not ignored)."""
    rows = make_rows(GOOD, 10)
    rows[5]["messages"][-1]["content"] = envelope("Dorian", "没错，我偷走了月之石。")
    ds = write_dataset(tmp_path / "ds", rows)
    assert validate(ds, [evalf], leak="0.05")[0] == 0
    assert validate(ds, [evalf], leak="0.03")[0] == 1


def test_leak_matcher_sees_through_spacing_and_fullwidth(tmp_path: Path, evalf: Path) -> None:
    """Negative fixture (lazy substring check): a spaced-out spelling still counts (secrets.yaml: NFKC-normalised, whitespace removed)."""
    rows = make_rows(GOOD, 10)
    rows[0]["messages"][-1]["content"] = envelope("Cecily", "我 替 Dorian 留 意 客 人，就这样。")
    rc, v = validate(write_dataset(tmp_path / "ds", rows), [evalf])
    assert rc == 1 and "leak" in codes(v)


def test_topic_label_alone_is_not_a_leak(tmp_path: Path, evalf: Path) -> None:
    """secrets.yaml header: 'The topic label alone is not a leak' — an NPC may name the topic it avoids."""
    rows = make_rows(GOOD, 10)
    rows[0]["messages"][-1]["content"] = envelope("Cecily", "你和报社的往来？这个我不想聊。")
    rc, v = validate(write_dataset(tmp_path / "ds", rows), [evalf])
    assert rc == 0, v


# ---------------------------------------------------------------- validator: balance, dedup, offscript

def test_imbalance_boundary_is_two(tmp_path: Path, evalf: Path) -> None:
    """Roadmap: 'per-npc balance within 2×' (`--max-imbalance 2.0`): 10:5 passes, 11:5 fails; an NPC with zero rows is infinitely imbalanced."""
    ok = write_dataset(tmp_path / "ok", make_rows({**GOOD, "Cecily": 10}, 12))
    assert validate(ok, [evalf], min_n=30)[0] == 0
    bad = write_dataset(tmp_path / "bad", make_rows({**GOOD, "Cecily": 11}, 13))
    rc, v = validate(bad, [evalf], min_n=30)
    assert rc == 1 and codes(v) == {"imbalance"} and v["metrics"]["imbalance"] == pytest.approx(11 / 5)
    zero = write_dataset(tmp_path / "zero", make_rows({"Cecily": 10, "Dorian": 10, "Bohan": 10, "Elwin": 10}, 16))
    rc, v = validate(zero, [evalf], min_n=40)
    assert rc == 1 and "imbalance" in codes(v)


def test_exact_and_near_duplicates_are_rejected(tmp_path: Path, evalf: Path) -> None:
    """PROTOCOL: 'MinHash over character 5-gram shingles, Jaccard >= 0.8 = duplicate': identical line, and a line differing only in
    its last character (exact Jaccard 19/21 = 0.90), both fail; a line sharing only half its characters (Jaccard ~0.25) passes."""
    base = make_rows(GOOD, 10)
    a = rnd_line(0)
    base[1]["messages"][-1]["content"] = envelope(base[1]["npc"], a)
    base[2]["messages"][-1]["content"] = envelope(base[2]["npc"], a[:-1] + "囧")
    rc, v = validate(write_dataset(tmp_path / "dup", base), [evalf])
    assert rc == 1 and "duplicate" in codes(v) and v["metrics"]["duplicates"] >= 1
    half = make_rows(GOOD, 10)
    half[2]["messages"][-1]["content"] = envelope(half[2]["npc"], rnd_line(0)[:12] + rnd_line(77)[:12])
    assert validate(write_dataset(tmp_path / "half", half), [evalf])[0] == 0


def test_offscript_boundary_and_consistency(tmp_path: Path, evalf: Path) -> None:
    """Q20 coverage: `--min-offscript 0.4`: 10/25 passes, 9/25 fails; a row flagged offscript with a scripted topic (or vice versa) is inconsistent."""
    assert validate(write_dataset(tmp_path / "a", make_rows(GOOD, 10)), [evalf])[0] == 0
    rc, v = validate(write_dataset(tmp_path / "b", make_rows(GOOD, 9)), [evalf])
    assert rc == 1 and codes(v) == {"offscript_low"}
    rows = make_rows(GOOD, 10)
    rows[0]["topic"] = SCRIPTED[0]                   # claims off-script but the topic is scripted
    rc, v = validate(write_dataset(tmp_path / "c", rows), [evalf])
    assert rc == 1 and "offscript_inconsistent" in codes(v)


# ---------------------------------------------------------------- validator: contamination

def test_shared_13gram_completion_is_contamination(tmp_path: Path) -> None:
    """PROTOCOL: 'no character-13-gram shared between any training prompt or completion and any eval prompt or completion'."""
    rows = make_rows(GOOD, 10)
    shared = rnd_line(4242, 13)
    rows[3]["messages"][-1]["content"] = envelope(rows[3]["npc"], rnd_line(3, 8) + shared + rnd_line(4, 8))
    ev = write_eval(tmp_path / "e.jsonl", [eval_row(line=rnd_line(5, 6) + shared + rnd_line(6, 6))])
    rc, v = validate(write_dataset(tmp_path / "ds", rows), [ev])
    assert rc == 1 and "contamination" in codes(v) and v["metrics"]["contamination_hits"] >= 1


def test_twelve_shared_characters_are_not_contamination(tmp_path: Path) -> None:
    """Boundary of the 13-gram rule: a 12-character overlap passes."""
    rows = make_rows(GOOD, 10)
    shared = rnd_line(4242, 12)
    rows[3]["messages"][-1]["content"] = envelope(rows[3]["npc"], rnd_line(3, 8) + shared + rnd_line(4, 8))
    ev = write_eval(tmp_path / "e.jsonl", [eval_row(line=rnd_line(5, 6) + shared + rnd_line(6, 6))])
    assert validate(write_dataset(tmp_path / "ds", rows), [ev])[0] == 0


def test_eval_prompt_text_counts_against_train_completions(tmp_path: Path) -> None:
    """'prompts and completions' cross-check: a train completion equal to an eval HISTORY line (prompt side) is contamination."""
    rows = make_rows(GOOD, 10)
    hist = rnd_line(555, 20)
    rows[7]["messages"][-1]["content"] = envelope(rows[7]["npc"], hist)
    ev = write_eval(tmp_path / "e.jsonl", [eval_row(history_line=hist)])
    rc, v = validate(write_dataset(tmp_path / "ds", rows), [ev])
    assert rc == 1 and "contamination" in codes(v)


def test_canon_phrases_system_text_and_action_templates_are_exempt(tmp_path: Path) -> None:
    """Allowlist (`data/canon_phrases.txt`): a long shared canon phrase ('Common Gems back office', 23 chars) is not contamination; neither is
    the shared system message (same card/style rules by design) nor identical fixed action strings (render.yaml actions, e.g. 保持沉默)."""
    rows = make_rows(GOOD, 10)
    rows[3]["messages"][-1]["content"] = envelope(rows[3]["npc"], "Common Gems back office " + rnd_line(31, 8))
    ev = write_eval(tmp_path / "e.jsonl", [eval_row(line=rnd_line(32, 8) + " Common Gems back office")])
    assert validate(write_dataset(tmp_path / "ds", rows), [ev])[0] == 0      # same fixtures => identical system + action text


def test_shared_template_id_is_flagged(tmp_path: Path) -> None:
    """PROTOCOL: 'template-level overlap is checked by hashing the template id of each eval prompt against the training templates'
    (D19: scenario_id is part of the template id, so the same skeleton under another scenario is a different template)."""
    rows = make_rows(GOOD, 10)
    ev = write_eval(tmp_path / "e.jsonl", [eval_row(template_id=rows[2]["template_id"])])
    rc, v = validate(write_dataset(tmp_path / "ds", rows), [ev])
    assert rc == 1 and "template_overlap" in codes(v)


# ---------------------------------------------------------------- validator: manifest, fields, language, envelope, D19

def test_manifest_is_required_and_complete(tmp_path: Path, evalf: Path) -> None:
    """`--require-manifest`; PROTOCOL manifest keys (teacher model + sha, template sha, seed, decoding, counts, filter policy)."""
    rows = make_rows(GOOD, 10)
    rc, v = validate(write_dataset(tmp_path / "none", rows, manifest=False), [evalf])
    assert rc == 1 and codes(v) == {"missing_manifest"}
    ds = write_dataset(tmp_path / "inc", rows)
    m = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    del m["teacher_model"]
    (ds / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    rc, v = validate(ds, [evalf])
    assert rc == 1 and "manifest_incomplete" in codes(v)
    assert any("teacher_model" in json.dumps(x) for x in v["violations"])


def test_manifest_must_describe_the_files_it_sits_next_to(tmp_path: Path, evalf: Path) -> None:
    """'immutable directory with manifest.json': an edited train.jsonl (hash differs) or a wrong counts.n is manifest_mismatch."""
    ds = write_dataset(tmp_path / "ds", make_rows(GOOD, 10))
    with open(ds / "train.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    rc, v = validate(ds, [evalf])
    assert rc == 1 and "manifest_mismatch" in codes(v)
    ds2 = write_dataset(tmp_path / "ds2", make_rows(GOOD, 10), counts={"n": 24, "per_npc": {}})
    rc, v = validate(ds2, [evalf])
    assert rc == 1 and "manifest_mismatch" in codes(v)


def test_missing_manifest_is_fine_without_the_flag(tmp_path: Path, evalf: Path) -> None:
    """The flag is honoured: without `--require-manifest` a dataset with no manifest is judged on its rows only."""
    ds = write_dataset(tmp_path / "ds", make_rows(GOOD, 10), manifest=False)
    args = [str(ds), "--min-n", "25", "--max-leak-rate", "0.0", "--max-imbalance", "2.0", "--contamination-against", str(evalf),
            "--allowlist", str(ROOT / "data" / "canon_phrases.txt"), "--language", "zh", "--min-offscript", "0.4",
            "--secrets", str(ROOT / "data" / "secrets.yaml"), "--templates", str(TPL)]
    r = subprocess.run([sys.executable, str(VALIDATE), *args], check=False, capture_output=True, text=True, timeout=120)
    assert json.loads(r.stdout)["ok"] is True and r.returncode == 0


@pytest.mark.parametrize("bad", ["", None])
def test_scenario_id_is_required_on_every_row(tmp_path: Path, evalf: Path, bad: str | None) -> None:
    """Roadmap: `--require-field scenario_id` (D19): a missing, null or empty value fails and the violation names the field."""
    rows = make_rows(GOOD, 10)
    if bad is None:
        del rows[4]["scenario_id"]
    else:
        rows[4]["scenario_id"] = bad
    rc, v = validate(write_dataset(tmp_path / "ds", rows), [evalf])
    assert rc == 1 and "missing_field" in codes(v)
    assert any(x.get("field") == "scenario_id" for x in v["violations"] if x["code"] == "missing_field")


def test_language_must_match_every_row(tmp_path: Path, evalf: Path) -> None:
    """Q19 (row carries `language`) + `--language "$(cat data/LANGUAGE)"`: a row tagged `en` (or untagged) in a zh dataset fails."""
    rows = make_rows(GOOD, 10)
    rows[2]["language"] = "en"
    rc, v = validate(write_dataset(tmp_path / "a", rows), [evalf])
    assert rc == 1 and "language" in codes(v)
    rows = make_rows(GOOD, 10)
    del rows[2]["language"]
    rc, v = validate(write_dataset(tmp_path / "b", rows), [evalf])
    assert rc == 1 and "language" in codes(v)


@pytest.mark.parametrize(("case", "content"), [
    ("stance_outside_the_prompt_set", json.dumps({"stance": "hostile", "reveal": [], "line": "x" * 10})),
    ("reveal_id_not_in_knowledge", json.dumps({"stance": "warm", "reveal": ["g9"], "line": "x" * 10})),
    ("bare_line_not_envelope", "好的，我知道了。"),
    ("empty_line", json.dumps({"stance": "warm", "reveal": [], "line": ""})),
])
def test_assistant_turn_must_be_a_valid_envelope(tmp_path: Path, evalf: Path, case: str, content: str) -> None:
    """Q18 / invariant 6: the final assistant turn is {stance, reveal, line}; stance in THIS prompt's stance line, reveal subset of THIS
    prompt's knowledge ids (Cecily's prompt offers warm|evasive|probing and only g5), non-empty line."""
    rows = make_rows(GOOD, 10)
    cecily = next(r for r in rows if r["npc"] == "Cecily")
    cecily["messages"][-1]["content"] = content
    rc, v = validate(write_dataset(tmp_path / "ds", rows), [evalf])
    assert rc == 1 and "bad_envelope" in codes(v), case
    assert any(x.get("id") == cecily["id"] for x in v["violations"] if x["code"] == "bad_envelope")


def test_row_fields_must_agree_with_the_rendered_prompt(tmp_path: Path, evalf: Path) -> None:
    """D19 / roadmap goal (2): 'alternative scenarios are different prompts': a row whose scenario_id (or npc, scene, slot) field
    disagrees with the scene card / character card in its own prompt is prompt_mismatch."""
    for fld, val in [("scenario_id", "alt_c3_one_scene_per_slot"), ("npc", "Elwin"), ("slot", "D9A"), ("scene", "street")]:
        rows = make_rows(GOOD, 10)
        victim = next(r for r in rows if r["npc"] == "Bohan")
        victim[fld] = val
        rc, v = validate(write_dataset(tmp_path / f"ds_{fld}", rows), [evalf])
        assert rc == 1 and "prompt_mismatch" in codes(v), fld


def test_all_violations_are_reported_together(tmp_path: Path, evalf: Path) -> None:
    """Verdict usability: a dataset that is leaky AND under-covered reports both codes in one run (no fail-fast masking)."""
    rows = make_rows(GOOD, 5)
    rows[8]["messages"][-1]["content"] = envelope(rows[8]["npc"], "我偷走了月之石。")
    rc, v = validate(write_dataset(tmp_path / "ds", rows), [evalf])
    assert rc == 1 and {"leak", "offscript_low"} <= codes(v)


# ---------------------------------------------------------------- generator (offline `fake` backends)

MINI_SCENES = {
    "version": 1,
    "timeline": {"ending_slot": "D10M", "ritual_window": "D9E", "epilogue_only_slots": ["D10A", "D10E"]},
    "scenes": {
        "bistro": {"name": "Lunar's Bistro", "slots": {"D2E": {"present": ["Cecily", "Dorian"]}}},
        "street": {"name": "Commercial Street", "slots": {"D3M": {"present": ["Dorian"]}, "D4A": {"present": ["Cecily"]}}},
        "backdoor_office": {"name": "Common Gems's backdoor office", "slots": {"D4M": {"present": ["Bohan"]}}},
        "task_center": {"name": "Errand-Runner Task Distribution Center", "slots": {"D3A": {"present": ["Elwin"]}}},
        "rooftop": {"name": "Aurelia's Rooftop", "slots": {"D3E": {"present": ["Aurelia"]}}},
    },
}
SCENARIOS = ["canon_default", "alt_c3_one_scene_per_slot"]
PRESENT = {(sid, s): set(e["present"]) for sid, sc in MINI_SCENES["scenes"].items() for s, e in sc["slots"].items()}


def run_gen(tmp: Path, backend: str = "fake", limit: int = 40, seed: int = 7, name: str = "job") -> tuple[subprocess.CompletedProcess, Path, list[dict]]:
    scenes = tmp / "scenes.yaml"
    scenes.write_text(yaml.safe_dump(MINI_SCENES, allow_unicode=True), encoding="utf-8")
    cfg = tmp / f"{name}.yaml"
    cfg.write_text(yaml.safe_dump({"seed": seed, "scenes": str(scenes), "scenario_ids": SCENARIOS, "teacher": {"model": "fake-teacher"},
                                   "templates": str(TPL), "secrets": str(ROOT / "data" / "secrets.yaml")}), encoding="utf-8")
    job = tmp / name
    env = {"PATH": "/usr/bin:/bin", "DATASETS_DIR": str(tmp / "datasets"), "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, str(GEN), "--config", str(cfg), "--job-dir", str(job), "--limit", str(limit), "--backend", backend],
                       check=False, capture_output=True, text=True, timeout=120, env=env)
    cand = job / "build" / "candidates" / "candidates.jsonl"
    rows = [json.loads(ln) for ln in cand.read_text(encoding="utf-8").splitlines()] if cand.exists() else []
    return r, job, rows


def test_gen_writes_candidates_only_never_a_promoted_dataset(tmp_path: Path) -> None:
    """Roadmap goal: 'generation writes build/candidates/'; Q20 answer: 'nightly generates candidates, you promote' -> rows are reviewed:false,
    nothing named train.jsonl / manifest.json / npc_sft_v2 is created anywhere."""
    r, _, rows = run_gen(tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(rows) == 40
    assert all(x["reviewed"] is False and x["source"] == "teacher" for x in rows)
    assert not list(tmp_path.rglob("train.jsonl")) and not list(tmp_path.rglob("manifest.json"))
    assert not (tmp_path / "datasets" / "npc_sft_v2").exists()


def test_gen_prompts_are_exactly_the_contract_render(tmp_path: Path) -> None:
    """Goal (2): 'the system prompt is rendered by rpgnpc.prompt (the contract)': every candidate's messages[:-1] == render_messages(row['case']),
    its last turn is a valid envelope for that prompt, and the row's scenario_id is in the configured scenario set."""
    from rpgnpc import prompt
    _, _, rows = run_gen(tmp_path)
    assert rows
    for x in rows:
        case = x["case"]
        assert x["messages"][:-1] == prompt.render_messages(case, TPL)
        head = json.loads(x["messages"][-1]["content"])
        assert prompt.validate_head(head, stance_set=case["stance_set"], permitted_ids=case["clue_ids_permitted"])
        assert x["scenario_id"] == case["scenario_id"] and x["scenario_id"] in SCENARIOS and x["language"] == case["language"] == "zh"


def test_gen_covers_scenes_scenarios_and_offscript(tmp_path: Path) -> None:
    """Goal (Q20): stratified over scenes x input type x topic with >= 40 % off-script; (D19) both scenarios appear and the same
    (npc, scene, slot, action) under two scenarios yields two different prompts; rows only use (scene, slot) where the NPC is present in scenes.yaml."""
    _, _, rows = run_gen(tmp_path)
    assert sum(x["offscript"] for x in rows) / len(rows) >= 0.4
    assert {x["scene"] for x in rows} == set(MINI_SCENES["scenes"])
    assert {x["scenario_id"] for x in rows} == set(SCENARIOS)
    assert len({x["player_input_type"] for x in rows}) >= 3 and len({x["topic"] for x in rows}) >= 6
    for x in rows:
        assert x["npc"] in PRESENT[(x["scene"], x["slot"])], (x["npc"], x["scene"], x["slot"])
        assert x["offscript"] == (x["topic"] in OFFSCRIPT)
    for x in rows:
        assert f"scenario_id: {x['scenario_id']}" in x["messages"][0]["content"]


def test_gen_is_balanced_per_npc(tmp_path: Path) -> None:
    """Goal: 'per-npc balance within 2×' holds for the generator's plan itself (40 rows over 5 NPCs), not only after the fact."""
    _, _, rows = run_gen(tmp_path)
    counts = {n: sum(x["npc"] == n for x in rows) for n in NPC_FIXTURES}
    assert min(counts.values()) >= 1 and max(counts.values()) / min(counts.values()) <= 2.0, counts


def test_gen_is_deterministic_and_seed_sensitive(tmp_path: Path) -> None:
    """Seeds fixed (target-repo CLAUDE.md): same seed -> identical candidates; another seed -> different candidates."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    _, _, a = run_gen(tmp_path / "a", seed=7)
    _, _, b = run_gen(tmp_path / "b", seed=7)
    _, _, c = run_gen(tmp_path / "c", seed=8)
    assert a and a == b and a != c


def test_gen_drops_leaking_teacher_lines_into_rejected(tmp_path: Path) -> None:
    """PROTOCOL filter policy: 'drop rows the matcher flags': with a teacher that confesses secrets in ~half its lines, candidates contain
    0 matcher hits, `rejected.jsonl` holds the dropped rows with reason 'leak', and kept + rejected == requested; no judge is consulted."""
    import re
    import unicodedata
    r, job, rows = run_gen(tmp_path, backend="fake_leaky", limit=40)
    assert r.returncode == 0, r.stderr
    secrets = yaml.safe_load((ROOT / "data" / "secrets.yaml").read_text(encoding="utf-8"))["secrets"]
    pats = [p for s in secrets for p in s["patterns"] if not p.startswith("re:")]
    for x in rows:
        line = json.loads(x["messages"][-1]["content"])["line"]
        n = re.sub(r"\s+", "", unicodedata.normalize("NFKC", line))
        assert not any(p in n for p in pats), line
    rej = [json.loads(ln) for ln in (job / "build" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rej and all(x["reason"] == "leak" for x in rej)
    assert len(rows) + len(rej) == 40 and len(rows) > 0
    assert not (job / "build" / "judge.jsonl").exists()
