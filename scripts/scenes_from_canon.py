"""canon 04 presence matrix + conflict defaults -> scenes.yaml (M2).

Contract (pinned by tests/test_scenes_from_canon.py):
  scenes_from_canon.py --canon DIR [--out FILE]      write the generated scenes.yaml to FILE (default stdout)
  scenes_from_canon.py --canon DIR --check FILE      exit 0 iff FILE equals the generated document, else 1 with a diff-like
                                                     message naming the first differing keys on stderr

The matrix is parsed from ``DIR/04-scenes-and-names.md``; the conflict defaults C1-C13 are applied in code, because
04 states them as prose. Tags appear as ``# [conflict Q3-Cn]`` comments and as ``conflicts`` lists.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ALL_SLOTS = [f"D{d}{s}" for d in range(1, 11) for s in "MAE"]
EPILOGUE_SLOTS = ["D10A", "D10E"]
MATRIX_NAMES = {
    "Jay's Cottage": "cottage",
    "Aurelia's Rooftop": "rooftop",
    "Errand-Runner Task Distribution Center": "task_center",
    "Commercial Street": "street",
    "Lunar's Bistro": "bistro",
    "Common Gems's backdoor office": "backdoor_office",
    "Newspaper Office": "newspaper_office",
    "Alchemy Stall": "alchemy_stall",
}
ALLEY_NAME = "Alley"   # [inferred] 9th scene, case bible 05 s5 / Q11; nobody is ever there
NPCS = {"Aurelia", "Bohan", "Cecily", "Dorian", "Elwin"}


class CanonError(ValueError):
    pass


def parse_matrix(text: str) -> dict[str, dict[str, list[str]]]:
    """Return {scene_id: {slot: [npc..]}} from the markdown presence matrix; a blank cell is not a key."""
    lines = text.split("\n")
    head = next((i for i, ln in enumerate(lines) if re.match(r"\|\s*Scene\s*\|\s*D1M\s*\|", ln)), None)
    if head is None:
        raise CanonError("presence matrix not found (no '| Scene | D1M | ...' header)")
    cols = [c.strip() for c in lines[head].strip().strip("|").split("|")][1:]
    if cols != ALL_SLOTS:
        raise CanonError("presence matrix header is not D1M..D10E")
    out: dict[str, dict[str, list[str]]] = {}
    for ln in lines[head + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        name, cells = cells[0], cells[1:]
        if name not in MATRIX_NAMES:
            raise CanonError(f"presence matrix: unknown scene {name!r}")
        if len(cells) != len(ALL_SLOTS):
            raise CanonError(f"presence matrix: row {name!r} has {len(cells)} cells, expected {len(ALL_SLOTS)}")
        row: dict[str, list[str]] = {}
        for slot, cell in zip(ALL_SLOTS, cells):
            if not cell:
                continue
            who = [] if cell == "·" else [w.strip() for w in cell.split(",")]
            bad = [w for w in who if w not in NPCS]
            if bad:
                raise CanonError(f"presence matrix: unknown NPC {bad} in {name} {slot}")
            row[slot] = who
        out[MATRIX_NAMES[name]] = row
    if set(out) != set(MATRIX_NAMES.values()):
        raise CanonError("presence matrix: scenes missing: " + ", ".join(sorted(set(MATRIX_NAMES.values()) - set(out))))
    return out


def build_scenes(canon_dir: str | Path) -> dict[str, Any]:
    """Parse ``04-scenes-and-names.md`` (presence matrix) and apply the conflict defaults C1-C13; returns the scenes document."""
    path = Path(canon_dir) / "04-scenes-and-names.md"
    if not path.exists():
        raise CanonError(f"presence matrix not found: {path} does not exist")
    matrix = parse_matrix(path.read_text(encoding="utf-8"))
    slots: dict[str, dict[str, dict[str, Any]]] = {
        sid: {s: {"present": list(who)} for s, who in row.items()} for sid, row in matrix.items()
    }

    def tag(sid: str, s: str, c: str) -> None:
        e = slots[sid][s]
        e.setdefault("conflicts", [])
        if c not in e["conflicts"]:
            e["conflicts"].append(c)

    def put(sid: str, s: str, who: list[str], c: str) -> None:
        slots[sid].setdefault(s, {"present": []})
        for w in who:
            if w not in slots[sid][s]["present"]:
                slots[sid][s]["present"].append(w)
        tag(sid, s, c)

    for s in ("D3A", "D6A"):                                  # C1: Elwin at the Task Center in the afternoon
        put("task_center", s, ["Elwin"], "Q3-C1")
    put("rooftop", "D3E", ["Aurelia"], "Q3-C2")               # C2: 02 "今天晚上在老地方见"
    slots["rooftop"].pop("D6M", None)                         # C3 default (one scene per slot): Aurelia D6A, not D6M
    put("rooftop", "D6A", ["Aurelia"], "Q3-C3")
    put("alchemy_stall", "D6E", [], "Q3-C3")
    for s in ALL_SLOTS:                                       # C4: self-service stall open every slot, nobody present
        if s not in slots["alchemy_stall"]:
            put("alchemy_stall", s, [], "Q3-C4")
    for s in ("D4A", "D5M"):                                  # C5/C9: Cecily on the street, bistro closed
        put("street", s, ["Cecily"], "Q3-C5")
        slots["bistro"].pop(s, None)
    tag("street", "D4A", "Q3-C9")
    put("bistro", "D2E", ["Cecily", "Dorian"], "Q3-C7")       # C7: Dorian cameo (D2E only, Q37 default)
    slots["rooftop"]["D2E"]["present"] = []                   # C10: dead drop, nobody there [inferred]
    slots["rooftop"]["D2E"]["dead_drop"] = True
    tag("rooftop", "D2E", "Q3-C10")
    for s in ALL_SLOTS:                                       # Alley [inferred]: open and empty every slot
        slots.setdefault("alley", {})[s] = {"present": []}
    for per_scene in slots.values():                          # C6: Day-10 afternoon/evening are epilogue only
        for s in EPILOGUE_SLOTS:
            if s in per_scene:
                per_scene[s]["epilogue_only"] = True
    tag("rooftop", "D10A", "Q3-C6")

    scenes: dict[str, Any] = {}
    names = {v: k for k, v in MATRIX_NAMES.items()}
    names["alley"] = ALLEY_NAME
    for sid in [*MATRIX_NAMES.values(), "alley"]:
        scenes[sid] = {"name": names[sid], "slots": {s: slots[sid][s] for s in ALL_SLOTS if s in slots[sid]}}
    return {
        "version": 1,
        "timeline": {"ending_slot": "D10M", "ritual_window": "D9E", "epilogue_only_slots": list(EPILOGUE_SLOTS)},   # C12
        "scenes": scenes,
    }


def _flow(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(_flow(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_flow(x)}" for k, x in v.items()) + "}"
    return json.dumps(v, ensure_ascii=False)


def dump_scenes(doc: dict[str, Any]) -> str:
    """Deterministic YAML text with ``# [conflict Q3-Cn]`` comments."""
    t = doc["timeline"]
    out = [
        "# Generated by scripts/scenes_from_canon.py from canon 04 (presence matrix + conflict defaults). Do not edit.",
        f"version: {doc['version']}",
        "timeline:",
        f"  ending_slot: {t['ending_slot']}   # [conflict Q3-C12]",
        f"  ritual_window: {t['ritual_window']}   # [conflict Q3-C12]",
        f"  epilogue_only_slots: {_flow(t['epilogue_only_slots'])}   # [conflict Q3-C6]",
        "scenes:",
    ]
    for sid, sc in doc["scenes"].items():
        out += [f"  {sid}:", f"    name: {json.dumps(sc['name'], ensure_ascii=False)}", "    slots:"]
        for s, e in sc["slots"].items():
            note = "".join(f"   # [conflict {c}]" for c in e.get("conflicts", []))
            out.append(f"      {s}: {_flow(e)}{note}")
    return "\n".join(out) + "\n"


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    """Dotted path of the first key where ``a`` (generated) and ``b`` (checked file) differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                return f"{path}.{k}".lstrip(".") + " (missing in file)"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        for k in b:
            if k not in a:
                return f"{path}.{k}".lstrip(".") + " (not in canon)"
        return None
    return None if a == b else f"{path.lstrip('.')}: canon {a!r} != file {b!r}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canon", required=True, help="docs/canon directory")
    ap.add_argument("--out", help="write the generated scenes.yaml here (default stdout)")
    ap.add_argument("--check", help="compare this scenes.yaml with the generated document")
    args = ap.parse_args(argv)
    try:
        doc = build_scenes(args.canon)
    except CanonError as e:
        print(f"scenes_from_canon: {e}", file=sys.stderr)
        return 2
    if args.check:
        p = Path(args.check)
        if not p.is_file():
            print(f"scenes_from_canon: cannot read {p}", file=sys.stderr)
            return 2
        diff = first_difference(doc, yaml.safe_load(p.read_text(encoding="utf-8")))
        if diff:
            print(f"scenes_from_canon: {p} differs from canon at {diff}", file=sys.stderr)
            return 1
        return 0
    text = dump_scenes(doc)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
