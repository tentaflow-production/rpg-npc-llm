import argparse
from typing import Any

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def format_example(example: dict, tokenizer: AutoTokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    model_id  = cfg["model"]["base_model"]
    data_file = cfg["paths"]["data_file"]
    out_cache = cfg["paths"]["data_cache"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=data_file, split="train")
    ds = ds.map(lambda ex: format_example(ex, tokenizer), remove_columns=ds.column_names)
    ds.save_to_disk(out_cache)
    print("Saved processed dataset to", out_cache)


if __name__ == "__main__":
    main()
