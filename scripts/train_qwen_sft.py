"""Train a LoRA adapter on the PRE-TOKENIZED NPC dialogue dataset.

Architectural decision (see docs/training.md and CLAUDE.md):
  Tokenization AND assistant-only label masking are owned entirely by
  scripts/prepare_dataset.py. This script merely CONSUMES the
  input_ids / attention_mask / labels it already computed. We therefore use a
  plain transformers.Trainer — NOT TRL's SFTTrainer, which is built to tokenize
  and mask itself and would conflict with our already-masked labels. Loss here
  is exactly cross-entropy on the labels we unit-tested — no in-library
  preprocessing black box.

I/O:
  --config  YAML config (default: configs/qwen_npc_v1.yaml). Supplies the base
            model, LoRA config, training hyperparameters, and paths. Crucially,
            paths.data_cache must point at a timestamped tokenized dir produced
            by prepare_dataset.py.

Usage:
    python scripts/train_qwen_sft.py [--config configs/qwen_npc_v1.yaml]
    python scripts/train_qwen_sft.py --config configs/smoke.yaml   # fast overfit
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "configs" / "qwen_npc_v1.yaml")

IGNORE_INDEX = -100
REQUIRED_COLUMNS = {"input_ids", "attention_mask", "labels"}


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Pure, offline-unit-testable helpers (no heavy objects, no GPU, no downloads) #
# --------------------------------------------------------------------------- #


def build_training_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map a loaded config dict to transformers.TrainingArguments kwargs.

    Pure function (returns a plain dict) so it can be asserted in unit tests
    without constructing a TrainingArguments object. max_steps defaults to -1
    (TrainingArguments' "ignore me, use epochs" sentinel); a positive max_steps
    in the config overrides num_train_epochs.
    """
    t = cfg["training"]
    paths = cfg["paths"]
    return {
        "output_dir": paths["output_dir"],
        "logging_dir": paths["log_dir"],
        "per_device_train_batch_size": t["per_device_train_batch_size"],
        "gradient_accumulation_steps": t["gradient_accumulation_steps"],
        "num_train_epochs": t["num_train_epochs"],
        "max_steps": t.get("max_steps", -1),
        "learning_rate": float(t["learning_rate"]),
        "lr_scheduler_type": t["lr_scheduler_type"],
        "warmup_ratio": t["warmup_ratio"],
        "logging_steps": t["logging_steps"],
        "save_strategy": t["save_strategy"],
        "save_steps": t["save_steps"],
        "save_total_limit": t["save_total_limit"],
        "bf16": t["bf16"],
        # Gradient checkpointing: enable via TrainingArguments only (do NOT also
        # call model.gradient_checkpointing_enable() — double-enabling is a known
        # footgun). use_reentrant=False is the supported path for PEFT + ckpt.
        "gradient_checkpointing": t.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": t.get("report_to", "none"),
        "run_name": cfg.get("run_name"),
        # We feed pre-tokenized columns straight through the collator; Trainer
        # must NOT drop input_ids/labels as "unused".
        "remove_unused_columns": False,
    }


def experiment_snapshot_dir(experiment_root: str, run_name: str, date_str: str) -> Path:
    """~/experiments/<run_name>_<YYYY_MM_DD>/ — per CLAUDE.md reproducibility rule."""
    return Path(experiment_root) / f"{run_name}_{date_str}"


def take_first_n(dataset: Dataset, subset_size: Any) -> Dataset:
    """Return the first `subset_size` rows when it's a positive int, else the
    whole dataset. Used for the config-driven smoke/overfit subset."""
    if isinstance(subset_size, int) and subset_size > 0:
        return dataset.select(range(min(subset_size, len(dataset))))
    return dataset


def assert_dataset_columns(dataset: Dataset) -> None:
    """Pre-flight (a): the tokenized cache must have exactly the three columns
    a plain Trainer + DataCollatorForSeq2Seq consumes — nothing more, nothing
    less. A stray 'text'/'messages' column means the wrong cache was passed."""
    cols = set(dataset.column_names)
    if cols != REQUIRED_COLUMNS:
        raise ValueError(
            f"Tokenized dataset must have columns {sorted(REQUIRED_COLUMNS)}, "
            f"got {sorted(cols)}. Point paths.data_cache at a dir produced by "
            f"prepare_dataset.py."
        )


# --------------------------------------------------------------------------- #
# Trainer with a compute_loss seam                                            #
# --------------------------------------------------------------------------- #


class NpcTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # SEAM: add custom loss terms here later (e.g. RL/preference shaping).
        # Pass-through for now. **kwargs forwards num_items_in_batch, which newer
        # transformers passes to scale loss correctly under gradient
        # accumulation — dropping it makes the loss WRONG.
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )


# --------------------------------------------------------------------------- #
# Runtime self-checks (every invocation; cheap; fail fast)                     #
# --------------------------------------------------------------------------- #


def _sample_batch(dataset: Dataset, collator: DataCollatorForSeq2Seq, n: int = 2) -> dict:
    rows = [dataset[i] for i in range(min(n, len(dataset)))]
    return collator(rows)


def preflight_label_check(dataset: Dataset, collator: DataCollatorForSeq2Seq) -> None:
    """Pre-flight (b): a collated batch must contain BOTH supervised tokens and
    -100 — neither all-supervised nor all-masked."""
    labels = _sample_batch(dataset, collator)["labels"]
    n_supervised = int((labels != IGNORE_INDEX).sum())
    n_masked = int((labels == IGNORE_INDEX).sum())
    assert n_supervised > 0 and n_masked > 0, (
        f"label sanity failed: supervised={n_supervised}, masked={n_masked} "
        "(expected both > 0; all-masked or all-supervised ⇒ misaligned labels)"
    )
    print(f"[pre-flight] labels OK: {n_supervised} supervised / {n_masked} masked in sample batch")


def preflight_loss_read(model, dataset: Dataset, collator: DataCollatorForSeq2Seq) -> None:
    """Pre-flight (c, advisory): one forward pass; print the initial loss and a
    magnitude read. Does not assert — just surfaces obvious misalignment."""
    batch = {k: v.to(model.device) for k, v in _sample_batch(dataset, collator).items()}
    with torch.no_grad():
        loss = float(model(**batch).loss)
    print(
        f"[pre-flight] initial loss = {loss:.3f}  "
        "(1–3 normal; near ln(vocab)≈11.6 ⇒ likely label misalignment; 0 or NaN ⇒ problem)"
    )


def snapshot_experiment(exp_dir: Path, config_path: str) -> None:
    """Save the config + a pip freeze of THIS interpreter's env (the llm env)
    into the experiment dir, for reproducibility."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, exp_dir / "config.yaml")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    (exp_dir / "pip_freeze.txt").write_text(freeze.stdout)
    print(f"Experiment snapshot -> {exp_dir}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    base_model = cfg["model"]["base_model"]
    lora_cfg = cfg["lora"]
    data_cache = cfg["paths"]["data_cache"]
    output_dir = cfg["paths"]["output_dir"]

    # --- dataset (pre-tokenized; we only consume it) ---
    print(f"Loading tokenized dataset from {data_cache} ...")
    ds = load_from_disk(data_cache)
    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    assert isinstance(ds, Dataset)
    assert_dataset_columns(ds)  # pre-flight (a)

    subset_size = cfg["training"].get("subset_size")
    ds = take_first_n(ds, subset_size)
    note = f" (subset_size={subset_size})" if subset_size else ""
    print(f"Training on {len(ds)} examples{note}")

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- model (single-GPU; force everything on GPU 0 for Trainer device control) ---
    print("Loading base model (loads the 7B; may take >1 min) ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
    )
    # Order matters: attach LoRA, THEN enable input grads, THEN disable cache.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # enable_input_require_grads / config are PreTrainedModel APIs that the PEFT
    # wrapper delegates to the base model. Cast narrows to that interface so the
    # type checker resolves them (nn.Module.__getattr__ otherwise widens any
    # delegated attribute to Tensor | Module).
    pt_model = cast(PreTrainedModel, model)
    pt_model.enable_input_require_grads()  # REQUIRED so grads reach LoRA under grad ckpt
    pt_model.config.use_cache = False      # incompatible with gradient checkpointing

    # Required collator: pads labels with -100 (the default collator does not,
    # which would silently misalign them). Do NOT pass model= (triggers
    # encoder-decoder decoder_input_ids logic we don't want).
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX
    )

    preflight_label_check(ds, collator)        # (b) hard assert
    preflight_loss_read(model, ds, collator)   # (c) advisory

    # --- experiment snapshot (config + pip freeze) ---
    date_str = datetime.now().strftime("%Y_%m_%d")
    exp_dir = experiment_snapshot_dir(cfg["experiment_dir"], cfg["run_name"], date_str)
    snapshot_experiment(exp_dir, args.config)

    # --- train ---
    training_args = TrainingArguments(**build_training_kwargs(cfg))
    trainer = NpcTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted — saving current adapter before exit ...")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done. Adapter + tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
