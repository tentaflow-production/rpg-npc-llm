"""Train a LoRA adapter on the PRE-TOKENIZED NPC dialogue dataset, unattended.

Architectural decision (see docs/training.md and CLAUDE.md):
  Tokenization AND assistant-only (completion-only) label masking are owned by
  scripts/prepare_dataset.py. This script CONSUMES the input_ids /
  attention_mask / labels it computed, with a plain transformers.Trainer — NOT
  TRL's SFTTrainer. Loss is exactly cross-entropy on the masked labels.

Contract (experiments/PROTOCOL.md in the supervisor repo):
  --config   the FROZEN launched config (schema: experiments/config.template.yaml
             of the supervisor; every training value is explicit; ${VAR} is the
             only templating). Never written back to.
  --job-dir  $EXPERIMENTS_DIR/<exp>/ : receives metrics.csv, pip_freeze.txt,
             attempt markers and, ONLY on success, results.json.

Usage:
    python scripts/train_qwen_sft.py --config experiments/smoke.yaml --job-dir DIR --dry-run
    python scripts/train_qwen_sft.py --config experiments/smoke.yaml --job-dir DIR --dry-run --resume-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
IGNORE_INDEX = -100
REQUIRED_COLUMNS = {"input_ids", "attention_mask", "labels"}
LOSS_WINDOW = 10  # train_loss_final = mean of the last 10 logged training-loss values
DRY_RUN_STEPS = 5
RESUME_TEST_STEPS = 7  # --resume-test resumes the 5-step dry run and trains 2 more steps
DRY_CKPT = "dry_run_ckpt"
STARTED_MARKER = "trainer_started"

REQUIRED_KEYS: dict[str, list[str]] = {
    "": ["exp_name", "seed", "data_seed", "model", "data", "lora", "train"],
    "model": ["base", "dtype", "attn_implementation", "gradient_checkpointing"],
    "data": ["tokenized_dir", "val_fraction", "max_length", "dry_run_examples"],
    "lora": ["r", "alpha", "dropout", "target_modules", "bias", "task_type"],
    "train": [
        "epochs", "per_device_batch", "grad_accum", "lr", "scheduler", "warmup_ratio", "optim",
        "weight_decay", "max_grad_norm", "logging_steps", "eval_strategy", "save_strategy",
        "save_total_limit", "load_best_model_at_end", "metric_for_best_model",
        "resume_from_checkpoint", "report_to", "output_dir",
    ],
}


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


def env_defaults() -> dict[str, str]:
    home = Path.home()
    return {
        "ML_REPO": str(REPO_ROOT),
        "MODELS_DIR": str(home / "models"),
        "DATASETS_DIR": str(home / "datasets"),
        "CHECKPOINTS_DIR": str(home / "checkpoints"),
        "EXPERIMENTS_DIR": str(home / "experiments"),
        "LOGS_DIR": str(home / "logs"),
    }


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(obj: Any, env: dict[str, str] | None = None) -> Any:
    """Expand ${VAR} in every string of a nested structure. Nothing else is templated.

    Lookup order: the config's own top-level scalars are NOT consulted here; callers
    pass `env` (os.environ + defaults). `${exp_name}` is handled by the caller via env.
    """
    env = env if env is not None else {**env_defaults(), **os.environ}

    def sub(s: str) -> str:
        def rep(m: re.Match[str]) -> str:
            if m.group(1) not in env:
                raise KeyError(f"config refers to ${{{m.group(1)}}} but it is not set")
            return env[m.group(1)]

        return _VAR.sub(rep, s)

    if isinstance(obj, str):
        return sub(obj)
    if isinstance(obj, list):
        return [expand_env(v, env) for v in obj]
    if isinstance(obj, dict):
        return {k: expand_env(v, env) for k, v in obj.items()}
    return obj


def validate_config(cfg: dict[str, Any]) -> None:
    """Every training value must be explicit; refuse a config that relies on defaults."""
    for section, keys in REQUIRED_KEYS.items():
        node = cfg if section == "" else cfg.get(section)
        if not isinstance(node, dict):
            raise TypeError(f"config section '{section}' is missing")
        missing = [k for k in keys if k not in node]
        if missing:
            raise ValueError(f"config{'.' + section if section else ''} is missing: {missing}")
    if str(cfg["model"]["dtype"]) != "bf16":
        raise ValueError("model.dtype must be bf16 (PROTOCOL: bf16 only)")
    t = cfg["train"]
    if t["load_best_model_at_end"] and t["eval_strategy"] != t["save_strategy"]:
        raise ValueError("eval_strategy must equal save_strategy when load_best_model_at_end is true")
    if not 0 < float(cfg["data"]["val_fraction"]) < 1:
        raise ValueError("data.val_fraction must be in (0, 1)")


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    env = {**env_defaults(), **os.environ}
    if "exp_name" in raw:
        env["exp_name"] = str(raw["exp_name"])
    cfg = expand_env(raw, env)
    validate_config(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# Pure helpers (offline-testable)                                              #
# --------------------------------------------------------------------------- #


def build_training_kwargs(
    cfg: dict[str, Any],
    output_dir: str,
    logging_dir: str,
    *,
    dry_run: bool = False,
    max_steps: int = -1,
) -> dict[str, Any]:
    """Map the config to transformers.TrainingArguments kwargs (plain dict, no objects).

    Dry run: eval/save become `steps` with equal step counts (HF requires equality with
    load_best_model_at_end), and max_steps is set by the caller.
    """
    t = cfg["train"]
    kw: dict[str, Any] = {
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "per_device_train_batch_size": t["per_device_batch"],
        "per_device_eval_batch_size": t["per_device_batch"],
        "gradient_accumulation_steps": t["grad_accum"],
        "num_train_epochs": t["epochs"],
        "max_steps": max_steps,
        "learning_rate": float(t["lr"]),
        "lr_scheduler_type": t["scheduler"],
        "warmup_ratio": float(t["warmup_ratio"]),
        "optim": t["optim"],
        "weight_decay": float(t["weight_decay"]),
        "max_grad_norm": float(t["max_grad_norm"]),
        "logging_steps": t["logging_steps"],
        "eval_strategy": t["eval_strategy"],
        "save_strategy": t["save_strategy"],
        "save_total_limit": t["save_total_limit"],
        "load_best_model_at_end": bool(t["load_best_model_at_end"]),
        "metric_for_best_model": t["metric_for_best_model"],
        "greater_is_better": False,
        "bf16": True,
        "seed": cfg["seed"],
        "data_seed": cfg["data_seed"],
        # enable via TrainingArguments only (never also model.gradient_checkpointing_enable())
        "gradient_checkpointing": bool(cfg["model"]["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": t["report_to"],
        "run_name": cfg["exp_name"],
        "remove_unused_columns": False,  # pre-tokenized columns must survive
    }
    if dry_run:
        kw.update(
            eval_strategy="steps", save_strategy="steps", eval_steps=DRY_RUN_STEPS,
            save_steps=DRY_RUN_STEPS, logging_steps=1,
        )
    return kw


def resolve_tokenized_dir(path: str) -> str:
    """`path` is a tokenized dataset, or a parent holding dialogues_built_* (newest wins)."""
    p = Path(path)
    if (p / "dataset_info.json").exists() or (p / "dataset_dict.json").exists():
        return str(p)
    cands = sorted(c for c in p.glob("dialogues_built_*") if c.is_dir()) if p.is_dir() else []
    if not cands:
        raise FileNotFoundError(f"no tokenized dataset at or under {path}")
    return str(cands[-1])


def assert_dataset_columns(dataset: Any) -> None:
    cols = set(dataset.column_names)
    if cols != REQUIRED_COLUMNS:
        raise ValueError(
            f"Tokenized dataset must have columns {sorted(REQUIRED_COLUMNS)}, got {sorted(cols)}."
        )


def assert_max_length(dataset: Any, max_length: int) -> None:
    longest = max((len(x) for x in dataset["input_ids"]), default=0)
    if longest > max_length:
        raise ValueError(f"a row has {longest} tokens, over data.max_length={max_length}")


def take_first_n(dataset: Any, n: int | None) -> Any:
    if isinstance(n, int) and n > 0:
        return dataset.select(range(min(n, len(dataset))))
    return dataset


def split_train_val(dataset: Any, val_fraction: float, data_seed: int) -> tuple[Any, Any]:
    """Cut a validation split FROM train (never from data.eval). At least one val row."""
    n_val = max(1, round(len(dataset) * val_fraction))
    if n_val >= len(dataset):
        raise ValueError("dataset too small for a train/val split")
    parts = dataset.train_test_split(test_size=n_val, seed=data_seed, shuffle=True)
    return parts["train"], parts["test"]


def dataset_hash(dataset: Any) -> str:
    h = hashlib.sha256()
    for col in ("input_ids", "labels"):
        for row in dataset[col]:
            h.update(json.dumps(list(row), separators=(",", ":")).encode())
            h.update(b"\n")
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_sha256(model_dir: str | Path) -> str:
    """sha256 over names + bytes of every *.safetensors file (sorted) in the base-model dir."""
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(file_sha256(f).encode())
    return h.hexdigest()


def eval_set_hash(cfg: dict[str, Any], dry_run: bool) -> str | None:
    """sha256 of data.eval (+ data.eval_adversarial if given). None only in --dry-run when absent."""
    h = hashlib.sha256()
    paths = [cfg["data"].get(k) for k in ("eval", "eval_adversarial")]
    paths = [p for p in paths if p]
    if not paths or any(not Path(p).is_file() for p in paths):
        if dry_run:
            return None
        raise FileNotFoundError(f"eval set(s) missing: {paths}")
    for p in paths:
        h.update(file_sha256(p).encode())
    return h.hexdigest()


def train_loss_final(log_history: list[dict[str, Any]], window: int = LOSS_WINDOW) -> float | None:
    """Mean of the last `window` logged training-loss values (entries with a 'loss' key)."""
    losses = [float(e["loss"]) for e in log_history if "loss" in e]
    if not losses:
        return None
    tail = losses[-window:]
    return sum(tail) / len(tail)


# --- git / attempts / resume safety -----------------------------------------


def git_sha(repo: str | Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def tree_dirty(repo: str | Path) -> bool:
    out = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    return bool(out.strip())


def _attempt_dirs(job_dir: Path) -> list[Path]:
    def n(p: Path) -> int:
        return int(p.name.rsplit("-", 1)[1])

    return sorted(
        (p for p in job_dir.glob("attempt-train-*") if p.name.rsplit("-", 1)[1].isdigit()), key=n
    )


def begin_attempt(job_dir: Path, head_sha: str) -> tuple[Path, str | None]:
    """Return (current attempt dir, sha of the previous attempt that actually started).

    The dispatcher pre-creates attempt-train-N (with git_sha.txt) before launching; that
    unstarted dir is adopted. Run by hand, a new one is created. The current attempt is
    marked started so the NEXT attempt can compare against it.
    """
    attempts = _attempt_dirs(job_dir)
    if attempts and not (attempts[-1] / STARTED_MARKER).exists():
        current = attempts[-1]
        previous = [a for a in attempts[:-1] if (a / STARTED_MARKER).exists()]
    else:
        current = job_dir / f"attempt-train-{len(attempts) + 1}"
        current.mkdir(parents=True, exist_ok=True)
        previous = [a for a in attempts if (a / STARTED_MARKER).exists()]
    prev_sha = None
    if previous and (previous[-1] / "git_sha.txt").exists():
        prev_sha = (previous[-1] / "git_sha.txt").read_text().strip()
    (current / "git_sha.txt").write_text(head_sha + "\n")
    (current / STARTED_MARKER).write_text(head_sha + "\n")
    return current, prev_sha


def check_resume_sha(prev_sha: str | None, head_sha: str, allow_change: bool) -> None:
    """Refuse to resume a checkpoint written by a different git_sha (or an unknown one)."""
    if allow_change or prev_sha == head_sha:
        return
    if prev_sha is None:
        raise SystemExit(
            "refusing to resume: a checkpoint exists but the previous attempt's git_sha is unknown "
            "(pass --allow-sha-change to override)"
        )
    raise SystemExit(
        f"refusing to resume: previous attempt ran at {prev_sha[:10]}, HEAD is {head_sha[:10]} "
        "(pass --allow-sha-change to override)"
    )


def write_metrics_csv(path: Path, log_history: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for e in log_history:
        for k in e:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(log_history)


def write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def library_versions() -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "trl"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="frozen launched config (YAML)")
    ap.add_argument("--job-dir", required=True, help="experiment/job directory")
    ap.add_argument("--dry-run", action="store_true", help="32 examples, 5 steps, checkpoints in job dir")
    ap.add_argument("--resume-test", action="store_true", help="with --dry-run: resume the 5-step run, train to 7")
    ap.add_argument("--allow-sha-change", action="store_true", help="resume across a different git_sha")
    args = ap.parse_args(argv)
    if args.resume_test and not args.dry_run:
        ap.error("--resume-test requires --dry-run")

    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    t0 = time.time()

    cfg = load_config(args.config)
    head = git_sha(REPO_ROOT)
    if tree_dirty(REPO_ROOT):
        raise SystemExit("refusing to start: working tree of the ML repo is dirty (commit first)")

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "results.json").unlink(missing_ok=True)  # written only on success
    attempt_dir, prev_sha = begin_attempt(job_dir, head)

    import torch
    from datasets import Dataset, DatasetDict, load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(cfg["seed"])

    # --- data (pre-tokenized; we only consume it) ---
    tok_dir = resolve_tokenized_dir(cfg["data"]["tokenized_dir"])
    print(f"Loading tokenized dataset from {tok_dir}")
    ds = load_from_disk(tok_dir)
    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    assert isinstance(ds, Dataset)
    assert_dataset_columns(ds)
    if args.dry_run:
        ds = take_first_n(ds, cfg["data"]["dry_run_examples"])
    assert_max_length(ds, cfg["data"]["max_length"])
    train_ds, val_ds = split_train_val(ds, float(cfg["data"]["val_fraction"]), cfg["data_seed"])
    train_hash, val_hash = dataset_hash(train_ds), dataset_hash(val_ds)
    eval_hash = eval_set_hash(cfg, args.dry_run)
    print(f"train={len(train_ds)} val={len(val_ds)} eval_hash={'set' if eval_hash else 'None (dry run)'}")

    # --- output locations ---
    logs_root = os.environ.get("LOGS_DIR", env_defaults()["LOGS_DIR"])
    logging_dir = str(Path(logs_root) / cfg["exp_name"])
    output_dir = str(job_dir / DRY_CKPT) if args.dry_run else cfg["train"]["output_dir"]
    if args.dry_run and not args.resume_test:
        shutil.rmtree(output_dir, ignore_errors=True)  # a plain dry run is rerunnable
    max_steps = (RESUME_TEST_STEPS if args.resume_test else DRY_RUN_STEPS) if args.dry_run else -1

    # --- resume decision (and its git_sha safety check) ---
    resume = None
    if args.resume_test or cfg["train"]["resume_from_checkpoint"] == "auto":
        resume = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    if args.resume_test and resume is None:
        raise SystemExit(f"--resume-test: no checkpoint under {output_dir}; run --dry-run first")
    if resume is not None:
        check_resume_sha(prev_sha, head, args.allow_sha_change)
        print(f"Resuming from {resume}")

    # --- tokenizer / model ---
    base = cfg["model"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg["model"]["attn_implementation"],
        device_map={"": 0},
    )
    lc = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], bias=lc["bias"], task_type=lc["task_type"],
        ),
    )
    model.print_trainable_parameters()
    if cfg["model"]["gradient_checkpointing"]:
        model.enable_input_require_grads()  # REQUIRED so grads reach LoRA under grad ckpt
    model.config.use_cache = False

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX)
    sample = collator([train_ds[i] for i in range(min(2, len(train_ds)))])["labels"]
    if not ((sample != IGNORE_INDEX).any() and (sample == IGNORE_INDEX).any()):
        raise SystemExit("label sanity failed: a batch must contain both supervised and masked tokens")

    (job_dir / "pip_freeze.txt").write_text(
        subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True).stdout
    )

    class CsvMetrics(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            write_metrics_csv(job_dir / "metrics.csv", state.log_history)

    targs = TrainingArguments(
        **build_training_kwargs(cfg, output_dir, logging_dir, dry_run=args.dry_run, max_steps=max_steps)
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, processing_class=tokenizer, callbacks=[CsvMetrics()],
    )
    trainer.train(resume_from_checkpoint=resume)
    val = trainer.evaluate()
    write_metrics_csv(job_dir / "metrics.csv", trainer.state.log_history)
    if not args.dry_run:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    results = {
        "train_loss_final": train_loss_final(trainer.state.log_history),
        "train_loss_final_definition": f"mean of the last {LOSS_WINDOW} logged training-loss values",
        "val_loss": val.get("eval_loss"),
        "steps": trainer.state.global_step,
        "wall_sec": round(time.time() - t0, 1),
        "versions": library_versions(),
        "git_sha": head,
        "config_sha256": file_sha256(args.config),
        "train_hash": train_hash,
        "val_hash": val_hash,
        "eval_hash": eval_hash,
        "model_weights_sha256": weights_sha256(base),
        "seed": cfg["seed"],
        "data_seed": cfg["data_seed"],
        "dry_run": bool(args.dry_run),
        "resumed_from": resume,
        "attempt": attempt_dir.name,
    }
    write_json_atomic(job_dir / "results.json", results)
    print(f"Done. results.json written to {job_dir}")


if __name__ == "__main__":
    main()
