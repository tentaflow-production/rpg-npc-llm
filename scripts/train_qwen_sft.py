import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_CACHE = " /home/seabass1/projects/rpg-npc-llm/data_processed/npc_dialogues_tok"
OUTPUT_DIR = " /home/seabass1/projects/rpg-npc-llm/models/qwen-npc-lora-v1"
LOG_DIR = " /home/seabass1/projects/rpg-npc-llm/logs/qwen-npc-lora-v1"

def main():
    print("Loading dataset...")
    ds = load_from_disk(DATA_CACHE)

    print("Loading tokenizer & base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto",
    )

    # LoRA 配置（可以之后慢慢调）
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_dir=LOG_DIR,
        logging_steps=20,

        save_strategy="steps",   # 关键：保存 checkpoint
        save_steps=200,
        save_total_limit=3,      # 最多保留 3 个 checkpoint
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",        # 先不开 wandb
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )

    trainer.train()

    # 保存最终 LoRA + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training done, model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
