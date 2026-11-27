"""M2 spec: scripts/scenes_from_canon.py — scenes.yaml generated from canon 04's presence matrix + the conflict defaults C1-C13.

Roadmap M2 goal (1): "`scenes.yaml` is generated from canon 04's presence matrix + the case bible, not hand-maintained — the pipeline
currently encodes the xlsx timeline (Day-8 newspaper-office confrontation, Day-10-evening ending) that canon 04 resolved differently (C6, C12)".
Document schema (pinned here, [inferred]):
  version: 1
  timeline: {ending_slot, ritual_window, epilogue_only_slots: [..]}
  scenes: {<scene_id>: {name, slots: {D<d><M|A|E>: {present: [NPC..], conflicts?: [Q3-Cn..], epilogue_only?: bool, dead_drop?: bool}}}}
A slot key exists iff the scene is open in that slot. Conflict ids also appear as `# [conflict Q3-Cn]` comments in the written file.
RED until M2 is built.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts" / "scenes_from_canon.py"
CANON = Path(os.environ.get("SUPERVISOR_REPO", Path.home() / "projects" / "npc_game_supervisor")) / "docs" / "canon"

pytestmark = pytest.mark.skipif(not (CANON / "04-scenes-and-names.md").exists(), reason="canon dir not found ($SUPERVISOR_REPO)")

SCENES = {"cottage", "rooftop", "task_center", "street", "bistro", "backdoor_office", "newspaper_office", "alchemy_stall", "alley"}
NPCS = {"Aurelia", "Bohan", "Cecily", "Dorian", "Elwin"}
ALL_SLOTS = [f"D{d}{s}" for d in range(1, 11) for s in "MAE"]


def run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], check=False, capture_output=True, text=True, timeout=60)


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, str, Path]:
    out = tmp_path_factory.mktemp("scenes") / "scenes.yaml"
    r = run("--canon", CANON, "--out", out)
    assert r.returncode == 0, r.stderr
    text = out.read_text(encoding="utf-8")
    return yaml.safe_load(text), text, out


def slot(doc: dict, scene: str, s: str) -> dict | None:
    return doc["scenes"][scene]["slots"].get(s)


def present(doc: dict, scene: str, s: str) -> list[str] | None:
    e = slot(doc, scene, s)
    return None if e is None else list(e["present"])


# ---------- structure ----------

def test_scene_ids_and_names(generated) -> None:
    """04 'Scene descriptions' + case bible s5: the eight xlsx scenes plus the [inferred] Alley (Q11), nothing else."""
    doc = generated[0]
    assert doc["version"] == 1
    assert set(doc["scenes"]) == SCENES
    assert doc["scenes"]["street"]["name"] == "Commercial Street"
    assert doc["scenes"]["bistro"]["name"] == "Lunar's Bistro"


def test_slots_and_npcs_are_well_formed(generated) -> None:
    """04 'Time model': slots are D1M..D10E; only the six canon NPCs ever appear (Jay is the player, never 'present')."""
    for sid, sc in generated[0]["scenes"].items():
        for s, e in sc["slots"].items():
            assert s in ALL_SLOTS, (sid, s)
            assert set(e["present"]) <= NPCS, (sid, s, e)


def test_generation_is_deterministic(generated, tmp_path: Path) -> None:
    """Roadmap goal (1): generated, not hand-maintained — a second run is byte-identical."""
    out = tmp_path / "again.yaml"
    assert run("--canon", CANON, "--out", out).returncode == 0
    assert out.read_text(encoding="utf-8") == generated[1]


# ---------- verbatim xlsx cells (04 presence matrix) ----------

@pytest.mark.parametrize(("scene", "s", "who"), [
    ("rooftop", "D1E", ["Aurelia"]),
    ("task_center", "D1M", ["Elwin"]),
    ("backdoor_office", "D2M", ["Bohan"]),
    ("backdoor_office", "D4M", ["Bohan"]),
    ("street", "D3M", ["Dorian"]),
    ("street", "D3E", ["Elwin"]),
    ("street", "D5A", ["Aurelia"]),
    ("street", "D5E", ["Dorian"]),
    ("street", "D8M", ["Bohan"]),
    ("newspaper_office", "D3E", ["Dorian"]),
    ("newspaper_office", "D8E", ["Dorian"]),
    ("bistro", "D1A", ["Cecily"]),
    ("bistro", "D9E", ["Cecily"]),
    ("cottage", "D1M", []),
    ("cottage", "D9M", []),
])
def test_matrix_cells_are_verbatim(generated, scene: str, s: str, who: list[str]) -> None:
    """04 matrix: a name = that NPC is present; `·` = listed with nobody (e.g. Commercial Street D3M lists Dorian)."""
    assert present(generated[0], scene, s) == who


@pytest.mark.parametrize(("scene", "s"), [("cottage", "D1A"), ("cottage", "D10A"), ("task_center", "D2M"), ("newspaper_office", "D1M"),
                                          ("bistro", "D1M"), ("backdoor_office", "D3M")])
def test_blank_matrix_cells_are_not_listed(generated, scene: str, s: str) -> None:
    """04 matrix: 'Blank = the xlsx does not list the scene for that slot' -> no slot key (the scene is closed/unreachable)."""
    assert slot(generated[0], scene, s) is None


# ---------- the conflict defaults (04 'Conflicts', 02 wins) ----------

def has_tag(doc: dict, scene: str, s: str, tag: str) -> bool:
    e = slot(doc, scene, s)
    return e is not None and tag in e.get("conflicts", [])


def test_c1_task_center_afternoons(generated) -> None:
    """C1: 'Elwin at Task Center D3A and D6A in addition to the xlsx slots; tag [conflict Q3-C1]'."""
    d = generated[0]
    for s in ("D3A", "D6A"):
        assert "Elwin" in present(d, "task_center", s) and has_tag(d, "task_center", s, "Q3-C1")
    assert present(d, "task_center", "D3M") == ["Elwin"]


def test_c2_rooftop_d3e(generated) -> None:
    """C2: 'Aurelia on rooftop D3E' (02: 今天晚上在老地方见)."""
    d = generated[0]
    assert present(d, "rooftop", "D3E") == ["Aurelia"] and has_tag(d, "rooftop", "D3E", "Q3-C2")


def test_c3_day6_one_scene_per_slot(generated) -> None:
    """C3 default (one scene per slot): Bohan D6M, Aurelia D6A **not** D6M, Elwin D6A, Stall D6E."""
    d = generated[0]
    assert present(d, "backdoor_office", "D6M") == ["Bohan"]
    assert "Aurelia" not in (present(d, "rooftop", "D6M") or []) and present(d, "rooftop", "D6A") == ["Aurelia"]
    assert "Elwin" in present(d, "task_center", "D6A")
    assert slot(d, "alchemy_stall", "D6E") is not None
    assert has_tag(d, "rooftop", "D6A", "Q3-C3")


def test_c4_stall_is_open_every_slot_and_unstaffed(generated) -> None:
    """C4: 'Stall open (self-service, nobody present) every slot'."""
    d = generated[0]
    for s in ALL_SLOTS:
        assert present(d, "alchemy_stall", s) == [], s


def test_c5_c9_cecily_on_the_street_and_bistro_closed(generated) -> None:
    """C5/C9: Cecily is on the street D4A and D5M; the bistro is closed those slots (the D4A bistro row is dropped)."""
    d = generated[0]
    for s in ("D4A", "D5M"):
        assert present(d, "street", s) == ["Cecily"], s
        assert slot(d, "bistro", s) is None, s
    assert has_tag(d, "street", "D4A", "Q3-C9") or has_tag(d, "street", "D4A", "Q3-C5")


def test_c6_day10_rooftop_is_epilogue_only(generated) -> None:
    """C6: 'Ignore the Day 10 task; D10 rooftop = ending scene' — D10A/D10E slots carry epilogue_only and no other slot does."""
    doc = generated[0]
    assert set(doc["timeline"]["epilogue_only_slots"]) == {"D10A", "D10E"}
    for sid, sc in doc["scenes"].items():
        for s, e in sc["slots"].items():
            assert bool(e.get("epilogue_only")) == (s in {"D10A", "D10E"}), (sid, s)
    assert "Q3-C6" in (slot(doc, "rooftop", "D10A") or {}).get("conflicts", [])


def test_c7_dorian_cameo_at_the_bistro_d2e(generated) -> None:
    """C7: 'Dorian cameo at the bistro D2E' (Cecily: 坐在靠门口的灰色大衣的先生，他是Dorian) alongside Cecily."""
    d = generated[0]
    assert {"Cecily", "Dorian"} <= set(present(d, "bistro", "D2E"))
    assert "Q3-C7" in slot(d, "bistro", "D2E")["conflicts"]


def test_c10_d2e_rooftop_is_a_dead_drop(generated) -> None:
    """C10: '02 wins: Day 2 is a dead drop at the rooftop' -> the D2E rooftop slot is flagged dead_drop."""
    assert slot(generated[0], "rooftop", "D2E").get("dead_drop") is True


def test_c12_timeline(generated) -> None:
    """C12: 'ritual window is D9E; the game ends at D10M with the newspaper' (replaces the D10E full-moon default)."""
    t = generated[0]["timeline"]
    assert t["ritual_window"] == "D9E"
    assert t["ending_slot"] == "D10M"


def test_alley_is_open_and_empty(generated) -> None:
    """Case bible s5 'Alley (Q11)': nobody sees the Alley, any slot [inferred]; open D1M..D10E except epilogue slots keep the flag."""
    for s in ALL_SLOTS:
        assert present(generated[0], "alley", s) == [], s


def test_conflict_ids_appear_as_comments(generated) -> None:
    """Scope: 'with the conflict ids as comments': every conflict that changed data is written as `# [conflict Q3-Cn]`."""
    text = generated[1]
    for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9", "C10", "C12"):
        assert f"[conflict Q3-{c}]" in text, c


# ---------- it parses the canon, it does not hard-code it ----------

def test_a_changed_matrix_cell_changes_the_output(tmp_path: Path) -> None:
    """Anti-hard-coding: edit one matrix cell in a COPY of 04 (rooftop D1E Aurelia -> Cecily); the generated cell follows."""
    canon = tmp_path / "canon"
    shutil.copytree(CANON, canon, ignore=shutil.ignore_patterns("*.xlsx"))
    p = canon / "04-scenes-and-names.md"
    lines = p.read_text(encoding="utf-8").split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("| Aurelia's Rooftop |"))
    lines[i] = lines[i].replace("| Aurelia |", "| Cecily |", 1)
    p.write_text("\n".join(lines), encoding="utf-8")
    out = tmp_path / "s.yaml"
    assert run("--canon", canon, "--out", out).returncode == 0
    assert present(yaml.safe_load(out.read_text(encoding="utf-8")), "rooftop", "D1E") == ["Cecily"]


def test_missing_matrix_is_an_error_not_an_empty_document(tmp_path: Path) -> None:
    """Negative fixture: a canon dir without the presence matrix must fail loudly (rc != 0, message names the matrix), never emit empty scenes."""
    canon = tmp_path / "canon"
    canon.mkdir()
    (canon / "04-scenes-and-names.md").write_text("# nothing here\n", encoding="utf-8")
    r = run("--canon", canon, "--out", tmp_path / "x.yaml")
    assert r.returncode != 0
    assert "presence matrix" in r.stderr.lower()
    assert not (tmp_path / "x.yaml").exists()


# ---------- --check against the pipeline's hand-maintained file ----------

def test_check_accepts_the_generated_file(generated) -> None:
    """Done-when: `--check <pipeline scenes.yaml>` exits 0 on a file equal to the generated document."""
    r = run("--canon", CANON, "--check", generated[2])
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(("name", "mutate", "needle"), [
    ("xlsx_ending_slot",
     lambda d: d["timeline"].update(ending_slot="D10E", ritual_window="D10E"), "timeline"),
    ("c1_missing_elwin_afternoon",
     lambda d: d["scenes"]["task_center"]["slots"].pop("D3A"), "task_center"),
    ("c9_bistro_open_while_cecily_on_street",
     lambda d: d["scenes"]["bistro"]["slots"].update(D4A={"present": ["Cecily"]}), "bistro"),
    ("c6_day10_rooftop_task_slot",
     lambda d: d["scenes"]["rooftop"]["slots"]["D10A"].pop("epilogue_only", None), "rooftop"),
    ("c3_aurelia_on_rooftop_d6m",
     lambda d: d["scenes"]["rooftop"]["slots"].update(D6M={"present": ["Aurelia"]}), "rooftop"),
])
def test_check_rejects_the_xlsx_timeline_variants(generated, tmp_path: Path, name: str, mutate, needle: str) -> None:
    """Negative fixtures (roadmap goal 1): the pipeline's xlsx-timeline scenes.yaml (Day-10-evening ending, missing C1 afternoons,
    bistro open on D4A, Day-10 rooftop task, Aurelia on the roof D6M) is REJECTED and the message names the differing key."""
    doc = copy.deepcopy(generated[0])
    mutate(doc)
    bad = tmp_path / f"{name}.yaml"
    bad.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    r = run("--canon", CANON, "--check", bad)
    assert r.returncode == 1
    assert needle in r.stderr


def test_check_on_a_missing_file_is_an_error(tmp_path: Path) -> None:
    """--check with a non-existent file must not pass vacuously."""
    r = run("--canon", CANON, "--check", tmp_path / "nope.yaml")
    assert r.returncode != 0
    assert re.search(r"nope\.yaml", r.stderr)
