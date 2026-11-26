"""Evaluation helpers for ``scripts/eval.py`` (M1). STUB: the spec phase only fixes the interface.

Row format of ``data/eval_set.jsonl`` / ``data/eval_adversarial.jsonl``: see SPEC_SUMMARY.md.
``eval.json`` format: ``$SUPERVISOR_REPO/experiments/PROTOCOL.md`` "Metrics".
"""

from __future__ import annotations

from pathlib import Path

LEDGER_COLUMNS = [
    "exp", "date", "baseline", "knob", "target", "train_loss", "eval_loss", "adversarial leaks",
    "ordinary leaks", "in_character", "IMPROVED", "RELEASABLE", "note",
]


def load_rows(path: str | Path) -> list[dict]:
    """Read a jsonl eval file (one JSON object per line)."""
    raise NotImplementedError


def eval_set_hash(ordinary_path: str | Path, adversarial_path: str | Path) -> str:
    """sha256 over the bytes of eval_set.jsonl followed by the bytes of eval_adversarial.jsonl."""
    raise NotImplementedError


def parse_envelope(raw: str) -> dict:
    """Parse a model output into ``{"valid": bool, "stance": str|None, "reveal": list, "line": str}``.

    Accepts a bare JSON object or one wrapped in a markdown code fence. Anything else is ``valid=False`` and
    ``line`` is the raw text (so the leak matcher still sees it).
    """
    raise NotImplementedError


def score_row(row: dict, raw_output: str, secrets: list[dict]) -> dict:
    """One ``per_prompt`` entry: id, subset, npc, [target_secret_id], leak (bool), secret_ids_hit,
    stance_valid (bool), reveal_valid (bool), in_character (None until judged)."""
    raise NotImplementedError


def select_limit(ordinary: list[dict], adversarial: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    """Deterministic smoke subset: ``limit // 2`` ordinary + ``limit - limit // 2`` adversarial rows, spread over NPCs."""
    raise NotImplementedError


def build_eval_json(exp: str, ordinary: list[dict], adversarial: list[dict], outputs: dict[str, str],
                    secrets: list[dict], set_hash: str, decoding: dict, eval_loss: float | None,
                    judge_prompt_version: str, extra: dict | None = None) -> dict:
    """The PROTOCOL ``eval.json`` dict. Raises ``ValueError`` if an output is missing or an id is duplicated."""
    raise NotImplementedError


def refuse_baseline_dir(job_dir: str | Path, limit: int) -> None:
    """Raise ``ValueError`` when a ``--limit`` run targets an ``exp_*`` baseline directory."""
    raise NotImplementedError


def ledger_row(exp: str, baseline: str, ev: dict, verdict: dict, note: str = "") -> str:
    """One markdown table row for ``experiments/LEDGER.md`` (columns: ``LEDGER_COLUMNS``)."""
    raise NotImplementedError
