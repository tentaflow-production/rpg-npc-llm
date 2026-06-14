"""Tokenize the built NPC dialogues into a training-ready Arrow dataset.

Reads the merged dialogues JSONL (single- and multi-round conversations in
OpenAI chat format), applies Qwen's chat template with tokenization, and writes
a dataset of `input_ids` / `attention_mask` / `labels` to disk via
`save_to_disk`.

Two things it does deliberately:
  1. Label masking — only assistant tokens carry loss (everything else is
     -100). The system prompt here is a long persona block and the user turns
     are the player's lines; training on those would teach the model to
     reproduce the prompt, not to *be* the NPC.
  2. Left-truncation — when a row exceeds max_length we keep the TAIL, not the
     head. The assistant reply (the supervised target) lives at the end, so
     right-truncation would slice off exactly the part we train on.

I/O (all overridable on the CLI; see DEFAULT_* below):
  --config  YAML config; supplies model.base_model (tokenizer) and
            training.max_length (truncation threshold)
  --input   built dialogues JSONL
  --output  PARENT directory; the dataset is saved to a timestamped subdir
            `<output>/dialogues_built_<YYYYmmdd_HHMMSS>` so runs don't clobber

Usage:
    python scripts/prepare_dataset.py [--config ...] [--input ...] [--output ...]
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import yaml
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

IGNORE_INDEX = -100

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "configs" / "qwen_npc_v1.yaml")
DEFAULT_INPUT = "/home/seabass1/datasets/rpg-npc-llm-dataset-pipeline/build/dialogues_built.jsonl"
DEFAULT_OUTPUT = "/home/seabass1/datasets/rpg-npc-llm-dataset-pipeline/tokenized"


class SupportsChatTemplate(Protocol):
    """The single tokenizer capability this module depends on.

    Typing against this structural protocol (rather than the concrete
    PreTrainedTokenizerBase) keeps the contract honest and lets test doubles
    satisfy it without inheriting from / casting to the real tokenizer class.
    """

    def apply_chat_template(
        self,
        conversation: Any,
        /,
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        **kwargs: Any,
    ) -> Any: ...


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def tokenized_dir(parent: str | Path, timestamp: str) -> Path:
    """Per-run output directory under `parent`, name-stamped with `timestamp`
    (format YYYYmmdd_HHMMSS) so repeated runs don't overwrite each other."""
    return Path(parent) / f"dialogues_built_{timestamp}"


def encode_chat(
    tokenizer: SupportsChatTemplate, messages: list, add_generation_prompt: bool
) -> list[int]:
    # apply_chat_template's declared return type is a broad union (str / tensors /
    # Encoding / ...) since it depends on runtime args. With tokenize=True and no
    # return_tensors it's a flat list[int]; cast so callers get correct typing.
    return cast(
        list[int],
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        ),
    )


def tokenize_example(
    example: dict, tokenizer: SupportsChatTemplate, max_length: int
) -> dict:
    messages = example["messages"]

    # Full sequence (no generation prompt — the assistant turns are already here).
    input_ids = encode_chat(tokenizer, messages, add_generation_prompt=False)
    labels = [IGNORE_INDEX] * len(input_ids)

    # Supervise only assistant turns. For each assistant message, the target span
    # is [prefix .. this turn], where `prefix` is the conversation up to (but not
    # including) the assistant content — i.e. with the generation prompt header.
    # Qwen's template is prefix-consistent, so these slices line up against the
    # full sequence.
    for k, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = encode_chat(tokenizer, messages[:k], add_generation_prompt=True)
        upto = encode_chat(tokenizer, messages[: k + 1], add_generation_prompt=False)
        labels[len(prefix) : len(upto)] = input_ids[len(prefix) : len(upto)]

    # Left-truncate to keep the answer at the tail.
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Built dialogues JSONL to tokenize",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Parent directory; the dataset is saved to a timestamped subdir "
        "<output>/dialogues_built_<YYYYmmdd_HHMMSS>",
    )
    args = parser.parse_args()

    cfg        = load_config(args.config)
    model_id   = cfg["model"]["base_model"]
    max_length = cfg["training"]["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=args.input, split="train")
    # split="train" returns a single Dataset; assert it so the type checker
    # (and we) can rely on .column_names / .map / len / indexing below.
    assert isinstance(ds, Dataset)
    ds = ds.map(
        lambda ex: tokenize_example(ex, tokenizer, max_length),
        remove_columns=ds.column_names,
    )

    n_truncated = sum(1 for ids in ds["input_ids"] if len(ids) >= max_length)
    n_empty = sum(1 for lab in ds["labels"] if all(x == IGNORE_INDEX for x in lab))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = tokenized_dir(args.output, timestamp)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Tokenized {len(ds)} examples -> {out_dir}")
    if n_truncated:
        print(f"  note: {n_truncated} example(s) hit max_length={max_length} (left-truncated)")
    if n_empty:
        print(f"  WARNING: {n_empty} example(s) have no supervised tokens after truncation")


if __name__ == "__main__":
    main()
