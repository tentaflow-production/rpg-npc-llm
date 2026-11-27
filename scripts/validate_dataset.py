"""Dataset validator CLI (M2).

Prints ONE JSON object on stdout ``{"ok": bool, "n": int, "metrics": {...}, "violations": [{"code": str, ...}]}``
and exits 0 iff ok, else 1. Every check runs; violations are reported together.

Usage: validate_dataset.py DIR --min-n N --max-leak-rate F --max-imbalance F --contamination-against FILE... --allowlist FILE
       --require-manifest --require-field NAME --language L --min-offscript F [--secrets PATH] [--templates DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rpgnpc.prompt import validate_head

MANIFEST_KEYS = ["version", "teacher_model", "teacher_sha", "template_sha", "seed", "decoding", "counts", "filter_policy",
                 "train_sha256"]
SHINGLE = 5            # PROTOCOL: character 5-gram MinHash
JACCARD = 0.8
PERMS = 128
BANDS = 32             # 32 bands x 4 rows: a pair at Jaccard 0.8 collides with probability ~1
GRAM = 13              # PROTOCOL: character 13-gram contamination
SEP = "\x00"


def norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def template_regex(tpl: str) -> re.Pattern[str]:
    """Turn ``"a{x}b"`` into a regex with one named group per placeholder."""
    out, last = "", 0
    for m in re.finditer(r"\{(\w+)\}", tpl):
        out += re.escape(tpl[last:m.start()]) + f"(?P<{m.group(1)}>.+?)"
        last = m.end()
    return re.compile(out + re.escape(tpl[last:]) + r"\Z")


class Templates:
    def __init__(self, tdir: Path, language: str) -> None:
        render = yaml.safe_load((tdir / "render.yaml").read_text(encoding="utf-8"))
        lang = language if language in render["languages"] else render["languages"][0]
        self.scene_re = template_regex(render[lang]["scene_line"])
        self.stance_re = template_regex(render[lang]["stance_line"])
        self.stance_joiner = render[lang]["stance_joiner"]
        self.knowledge_re = template_regex(render[lang]["knowledge_line"])
        self.scene_ids = {name: sid for sid, name in render["scene_names"].items()}
        self.slot_codes = {label: code for code, label in render["slot_labels"][lang].items()}
        self.cards = yaml.safe_load((tdir / "cards.yaml").read_text(encoding="utf-8"))["cards"]
        self.npcs = list(yaml.safe_load((tdir / "stances.yaml").read_text(encoding="utf-8"))["stances"])
        topics = yaml.safe_load((tdir / "topics.yaml").read_text(encoding="utf-8"))
        self.offscript_topics = set(topics["offscript_topics"])
        self.action_res = {t: template_regex(spec[lang]) for t, spec in render["actions"].items() if spec["args"]}

    def parse_system(self, system: str) -> dict[str, Any]:
        """Facts the prompt itself states: npc, scene id, slot, scenario_id, stance set, permitted clue ids."""
        lines = system.split("\n")
        info: dict[str, Any] = {"npc": next((n for n, c in self.cards.items() if lines and lines[0] == c), None),
                                "stance_set": [], "clue_ids": []}
        for ln in lines:
            if m := self.scene_re.match(ln):
                label = m["slot_label"]
                info["scene"] = self.scene_ids.get(m["scene_name"])
                info["slot"] = f"D{m['day']}{self.slot_codes[label]}" if label in self.slot_codes else None
                info["scenario_id"] = m["scenario_id"]
            elif m := self.stance_re.match(ln):
                info["stance_set"] = m["stances"].split(self.stance_joiner)
            elif m := self.knowledge_re.match(ln):
                info["clue_ids"].append(m["clue_id"])
        return info

    def segments(self, messages: list[dict[str, str]], reference: str | None = None) -> list[str]:
        """Contamination text: assistant lines and action arguments, one segment each (system message and fixed action text excluded)."""
        segs: list[str] = []
        for i, m in enumerate(messages):
            if m["role"] == "system":
                continue
            if m["role"] == "user":
                for rx in self.action_res.values():
                    if mm := rx.match(m["content"]):
                        segs += list(mm.groupdict().values())
                        break
            else:
                segs.append(envelope_line(m["content"]) if i == len(messages) - 1 else m["content"])
        if reference:
            segs.append(envelope_line(reference))
        return segs


def envelope_line(content: str) -> str:
    try:
        head = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    return head["line"] if isinstance(head, dict) and isinstance(head.get("line"), str) else content


def parse_head(content: str) -> object:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def shingles(text: str, k: int) -> set[str]:
    return {text[i:i + k] for i in range(max(len(text) - k + 1, 1))} if text else set()


def minhash(sh: set[str], coeffs: list[tuple[int, int]]) -> list[int]:
    p = (1 << 61) - 1
    hs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh]
    return [min((a * h + b) % p for h in hs) for a, b in coeffs]


def find_duplicates(lines: list[str]) -> list[tuple[int, int, float]]:
    """Pairs (i, j) of lines whose exact 5-gram Jaccard >= 0.8, found by MinHash LSH and verified exactly."""
    rng = random.Random(0)
    p = (1 << 61) - 1
    coeffs = [(rng.randrange(1, p), rng.randrange(0, p)) for _ in range(PERMS)]
    sets = [shingles(norm(x), SHINGLE) for x in lines]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    rows = PERMS // BANDS
    for i, sh in enumerate(sets):
        if not sh:
            continue
        sig = minhash(sh, coeffs)
        for b in range(BANDS):
            buckets[(b, tuple(sig[b * rows:(b + 1) * rows]))].append(i)
    seen: set[tuple[int, int]] = set()
    found = []
    for ids in buckets.values():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                pair = (ids[x], ids[y])
                if pair in seen:
                    continue
                seen.add(pair)
                a, b = sets[pair[0]], sets[pair[1]]
                j = len(a & b) / len(a | b)
                if j >= JACCARD:
                    found.append((*pair, j))
    return sorted(found)


def grams(segments: list[str], allow: list[str]) -> set[str]:
    out: set[str] = set()
    for seg in segments:
        t = norm(seg)
        for phrase in allow:
            t = t.replace(phrase, SEP)
        for frag in t.split(SEP):
            out |= {frag[i:i + GRAM] for i in range(len(frag) - GRAM + 1)}
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def match_secret(text: str, secrets: list[dict[str, Any]]) -> str | None:
    n = norm(text)
    for s in secrets:
        for p in s["patterns"]:
            if (re.search(p[3:], n) if p.startswith("re:") else p in n):
                return s["id"]
    return None


def validate(args: argparse.Namespace) -> dict[str, Any]:
    viol: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    ds = Path(args.dataset)
    train = ds / "train.jsonl"
    if not train.is_file():
        return {"ok": False, "n": 0, "metrics": {}, "violations": [{"code": "missing_train", "path": str(train)}]}
    rows = load_jsonl(train)
    n = len(rows)
    tpl = Templates(Path(args.templates), args.language)
    secrets = yaml.safe_load(Path(args.secrets).read_text(encoding="utf-8"))["secrets"]

    if n < args.min_n:
        viol.append({"code": "too_few_rows", "n": n, "min_n": args.min_n})

    # manifest
    mpath = ds / "manifest.json"
    if args.require_manifest and not mpath.is_file():
        viol.append({"code": "missing_manifest"})
    elif mpath.is_file():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        missing = [k for k in MANIFEST_KEYS if k not in manifest]
        if missing:
            viol.append({"code": "manifest_incomplete", "missing": missing})
        if manifest.get("train_sha256") and manifest["train_sha256"] != hashlib.sha256(train.read_bytes()).hexdigest():
            viol.append({"code": "manifest_mismatch", "field": "train_sha256"})
        counts = manifest.get("counts")
        if isinstance(counts, dict):
            per = defaultdict(int)
            for r in rows:
                per[r.get("npc")] += 1
            if counts.get("n") != n:
                viol.append({"code": "manifest_mismatch", "field": "counts.n", "manifest": counts.get("n"), "actual": n})
            if "per_npc" in counts and counts["per_npc"] != dict(per):
                viol.append({"code": "manifest_mismatch", "field": "counts.per_npc"})

    leaks, per_npc = 0, defaultdict(int)
    leak_rows: list[dict[str, Any]] = []
    off = 0
    lines: list[str] = []
    for r in rows:
        rid = r.get("id")
        per_npc[r.get("npc")] += 1
        for f in args.require_field:
            if r.get(f) in (None, ""):
                viol.append({"code": "missing_field", "id": rid, "field": f})
        if r.get("language") != args.language:
            viol.append({"code": "language", "id": rid, "language": r.get("language")})
        msgs = r["messages"]
        info = tpl.parse_system(msgs[0]["content"])
        for f in ("npc", "scene", "slot", "scenario_id"):
            if r.get(f) and r[f] != info.get(f):
                viol.append({"code": "prompt_mismatch", "id": rid, "field": f, "row": r[f], "prompt": info.get(f)})
        final = msgs[-1]["content"]
        if msgs[-1]["role"] != "assistant" or not validate_head(parse_head(final), stance_set=info["stance_set"],
                                                                 permitted_ids=info["clue_ids"]):
            viol.append({"code": "bad_envelope", "id": rid})
        line = envelope_line(final)
        lines.append(line)
        if hit := match_secret(line, secrets):
            leaks += 1
            leak_rows.append({"code": "leak", "id": rid, "secret": hit})
        is_off = bool(r.get("offscript"))
        off += is_off
        if is_off != (r.get("topic") in tpl.offscript_topics):
            viol.append({"code": "offscript_inconsistent", "id": rid, "topic": r.get("topic")})

    metrics["leak_rate"] = leaks / n if n else 0.0
    if metrics["leak_rate"] > args.max_leak_rate:
        viol += leak_rows
    counts_by_npc = {k: per_npc.get(k, 0) for k in tpl.npcs}
    lo, hi = min(counts_by_npc.values()), max(counts_by_npc.values())
    metrics["per_npc"] = counts_by_npc
    metrics["imbalance"] = None if lo == 0 else hi / lo      # None = an NPC without rows (infinite)
    if lo == 0 or hi / lo > args.max_imbalance:
        viol.append({"code": "imbalance", "per_npc": counts_by_npc, "max": args.max_imbalance})
    metrics["offscript_fraction"] = off / n if n else 0.0
    if metrics["offscript_fraction"] < args.min_offscript:
        viol.append({"code": "offscript_low", "fraction": metrics["offscript_fraction"], "min": args.min_offscript})

    dups = find_duplicates(lines)
    metrics["duplicates"] = len({j for _, j, _ in dups})
    viol += [{"code": "duplicate", "ids": [rows[i].get("id"), rows[j].get("id")], "jaccard": round(jac, 3)} for i, j, jac in dups]

    # contamination
    allow = sorted({norm(x) for x in Path(args.allowlist).read_text(encoding="utf-8").splitlines()
                    if x.strip() and not x.startswith("#")}, key=len, reverse=True) if args.allowlist else []
    eval_grams: set[str] = set()
    eval_templates: set[str] = set()
    for ef in args.contamination_against:
        for er in load_jsonl(Path(ef)):
            eval_grams |= grams(tpl.segments(er["messages"], er.get("reference")), allow)
            if er.get("template_id"):
                eval_templates.add(er["template_id"])
    hits = 0
    for r in rows:
        shared = grams(tpl.segments(r["messages"]), allow) & eval_grams
        if shared:
            hits += 1
            viol.append({"code": "contamination", "id": r.get("id"), "example": min(shared)})
        if r.get("template_id") in eval_templates:
            viol.append({"code": "template_overlap", "id": r.get("id"), "template_id": r["template_id"]})
    metrics["contamination_hits"] = hits
    return {"ok": not viol, "n": n, "metrics": metrics, "violations": viol}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--min-n", type=int, required=True)
    ap.add_argument("--max-leak-rate", type=float, default=0.0)
    ap.add_argument("--max-imbalance", type=float, default=2.0)
    ap.add_argument("--contamination-against", nargs="*", default=[])
    ap.add_argument("--allowlist")
    ap.add_argument("--require-manifest", action="store_true")
    ap.add_argument("--require-field", action="append", default=[])
    ap.add_argument("--language", required=True)
    ap.add_argument("--min-offscript", type=float, default=0.0)
    ap.add_argument("--secrets", default=str(ROOT / "data" / "secrets.yaml"))
    ap.add_argument("--templates", default=str(ROOT / "data" / "templates"))
    verdict = validate(ap.parse_args(argv))
    print(json.dumps(verdict, ensure_ascii=False))
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
