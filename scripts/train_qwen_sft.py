import argparse
from typing import Any

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t   = cfg["training"]
    l   = cfg["lora"]

    base_model = cfg["model"]["base_model"]
    data_cache = cfg["paths"]["data_cache"]
    output_dir = cfg["paths"]["output_dir"]
    log_dir    = cfg["paths"]["log_dir"]

    print("Loading dataset...")
    ds = load_from_disk(data_cache)

    print("Loading tokenizer & base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        target_modules=l["target_modules"],
        bias=l["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        logging_dir=log_dir,
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=t["bf16"],
        report_to=t["report_to"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=t["max_seq_length"],
        args=training_args,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training done, model saved to", output_dir)


if __name__ == "__main__":
    main()
