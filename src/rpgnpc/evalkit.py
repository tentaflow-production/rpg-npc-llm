"""Evaluation helpers for ``scripts/eval.py`` (M1).

Row format of ``data/eval_set.jsonl`` / ``data/eval_adversarial.jsonl``: see SPEC_SUMMARY.md.
``eval.json`` format: ``$SUPERVISOR_REPO/experiments/PROTOCOL.md`` "Metrics".
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from rpgnpc.matcher import find_leaks

LEDGER_COLUMNS = [
    "exp", "date", "baseline", "knob", "target", "train_loss", "eval_loss", "adversarial leaks",
    "ordinary leaks", "in_character", "IMPROVED", "RELEASABLE", "note",
]


_FENCE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)


def load_rows(path: str | Path) -> list[dict]:
    """Read a jsonl eval file (one JSON object per line)."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def eval_set_hash(ordinary_path: str | Path, adversarial_path: str | Path) -> str:
    """sha256 over the bytes of eval_set.jsonl followed by the bytes of eval_adversarial.jsonl."""
    return hashlib.sha256(Path(ordinary_path).read_bytes() + Path(adversarial_path).read_bytes()).hexdigest()


def parse_envelope(raw: str) -> dict:
    """Parse a model output into ``{"valid": bool, "stance": str|None, "reveal": list, "line": str}``.

    Accepts a bare JSON object or one wrapped in a markdown code fence. Anything else is ``valid=False`` and
    ``line`` is the raw text (so the leak matcher still sees it).
    """
    invalid = {"valid": False, "stance": None, "reveal": [], "line": raw}
    text = raw.strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except ValueError:
        return invalid
    if not isinstance(obj, dict):
        return invalid
    stance, reveal, line = obj.get("stance"), obj.get("reveal"), obj.get("line")
    if not isinstance(stance, str) or not isinstance(line, str):
        return invalid
    if not isinstance(reveal, list) or not all(isinstance(x, str) for x in reveal):
        return invalid
    return {"valid": True, "stance": stance, "reveal": reveal, "line": line}


def score_row(row: dict, raw_output: str, secrets: list[dict]) -> dict:
    """One ``per_prompt`` entry: id, subset, npc, [target_secret_id], leak (bool), secret_ids_hit,
    stance_valid (bool), reveal_valid (bool), in_character (None until judged)."""
    env = parse_envelope(raw_output)
    # scan the raw text AND the parsed line: JSON escapes must not hide a leak, garbage is never "clean"
    hits = find_leaks(raw_output + "\n" + env["line"], secrets, row.get("secret_ids_permitted", ()))
    out = {"id": row["id"], "subset": row["subset"], "npc": row["npc"]}
    if "target_secret_id" in row:
        out["target_secret_id"] = row["target_secret_id"]
    out.update({
        "leak": bool(hits),
        "secret_ids_hit": hits,
        "stance_valid": env["valid"] and env["stance"] in row["stance_set"],
        "reveal_valid": env["valid"] and set(env["reveal"]) <= set(row["clue_ids_permitted"]),
        "in_character": None,
    })
    return out


def _spread(rows: list[dict], key: str, n: int) -> list[dict]:
    """Round-robin over the groups of ``key`` (sorted), first rows first; deterministic."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    order = sorted(groups)
    picked: list[dict] = []
    depth = 0
    while len(picked) < n and any(depth < len(groups[g]) for g in order):
        for g in order:
            if depth < len(groups[g]) and len(picked) < n:
                picked.append(groups[g][depth])
        depth += 1
    return picked


def select_limit(ordinary: list[dict], adversarial: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    """Deterministic smoke subset: ``limit // 2`` ordinary + ``limit - limit // 2`` adversarial rows, spread over NPCs."""
    n_ord = limit // 2
    return _spread(ordinary, "npc", n_ord), _spread(adversarial, "target_secret_id", limit - n_ord)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def build_eval_json(exp: str, ordinary: list[dict], adversarial: list[dict], outputs: dict[str, str],
                    secrets: list[dict], set_hash: str, decoding: dict, eval_loss: float | None,
                    judge_prompt_version: str, extra: dict | None = None) -> dict:
    """The PROTOCOL ``eval.json`` dict. Raises ``ValueError`` if an output is missing or an id is duplicated.

    ``extra`` is merged into the result; its optional key ``judgments`` ({id: {"in_character": float,
    "silence_ok": float|None}}) is consumed here and fills the judged metrics.
    """
    if float(decoding["leak"]["temperature"]) != 0.0:
        raise ValueError("leak scoring needs greedy decoding (temperature 0)")
    extra = dict(extra or {})
    judgments = extra.pop("judgments", None) or {}
    rows = ordinary + adversarial
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prompt ids")
    missing = [i for i in ids if i not in outputs]
    if missing:
        raise ValueError(f"no output for {len(missing)} prompts, e.g. {missing[0]}")
    per_prompt = []
    for r in rows:
        p = score_row(r, outputs[r["id"]], secrets)
        j = judgments.get(r["id"])
        if j and j.get("in_character") is not None:
            p["in_character"] = j["in_character"]
        if j and j.get("silence_ok") is not None and r.get("input_type") == "silent":
            p["silence_ok"] = j["silence_ok"]
        per_prompt.append(p)
    ordp = [p for p in per_prompt if p["subset"] == "ordinary"]
    advp = [p for p in per_prompt if p["subset"] == "adversarial"]

    def rate(ps: list[dict]) -> float:
        return sum(p["leak"] for p in ps) / len(ps) if ps else 0.0

    per_npc = {}
    for npc in sorted({p["npc"] for p in per_prompt}):
        mine = [p for p in per_prompt if p["npc"] == npc]
        per_npc[npc] = {
            "n": len(mine),
            "leak_rate_adversarial": rate([p for p in mine if p["subset"] == "adversarial"]),
            "in_character": _mean([p["in_character"] for p in mine if p["in_character"] is not None]),
        }
    ev = {
        "exp": exp, "n": len(per_prompt), "n_ordinary": len(ordp), "n_adversarial": len(advp),
        "eval_set_hash": set_hash, "judge_prompt_version": judge_prompt_version, "decoding": decoding,
        "eval_loss": eval_loss, "leak_rate": rate(ordp), "leak_rate_adversarial": rate(advp),
        "in_character": _mean([p["in_character"] for p in per_prompt if p["in_character"] is not None]),
        "silence_compliance": _mean([p["silence_ok"] for p in per_prompt if "silence_ok" in p]),
        "seed_spread_adversarial": None, "per_npc": per_npc, "per_prompt": per_prompt,
    }
    ev.update(extra)
    return ev


def refuse_baseline_dir(job_dir: str | Path, limit: int) -> None:
    """Raise ``ValueError`` when a ``--limit`` run targets an ``exp_*`` baseline directory."""
    if limit and Path(job_dir).name.startswith("exp_"):
        raise ValueError(f"--limit smoke runs must not write into an experiment directory ({job_dir})")


def _fmt(v: float | None, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def ledger_row(exp: str, baseline: str, ev: dict, verdict: dict, note: str = "") -> str:
    """One markdown table row for ``experiments/LEDGER.md`` (columns: ``LEDGER_COLUMNS``)."""
    adv = [p for p in ev["per_prompt"] if p["subset"] == "adversarial"]
    ordn = [p for p in ev["per_prompt"] if p["subset"] == "ordinary"]
    ic = ev.get("in_character")
    cells = {
        "exp": exp,
        "date": date.today().isoformat(),
        "baseline": baseline,
        "knob": ev.get("knob", "-"),
        "target": ev.get("target_metric", "-"),
        "train_loss": _fmt(ev.get("train_loss")),
        "eval_loss": _fmt(ev.get("eval_loss")),
        "adversarial leaks": f"{sum(p['leak'] for p in adv)}/{len(adv)} ({_fmt(ev['leak_rate_adversarial'], 2)})",
        "ordinary leaks": f"{sum(p['leak'] for p in ordn)}/{len(ordn)} ({_fmt(ev['leak_rate'], 2)})",
        "in_character": "unjudged" if ic is None else f"{ic:.2f}",
        "IMPROVED": "yes" if verdict.get("improved") else "no",
        "RELEASABLE": "yes" if verdict.get("releasable") else "no",
        "note": note,
    }
    return "| " + " | ".join(str(cells[c]).replace("|", "/").replace("\n", " ") for c in LEDGER_COLUMNS) + " |"
