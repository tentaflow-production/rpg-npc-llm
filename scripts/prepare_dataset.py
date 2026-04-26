from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE = "/home/seabass1/projects/rpg-npc-llm/data_processed/npc_dialogues.jsonl"
OUT_CACHE = "/home/seabass1/projects/rpg-npc-llm/data_processed/npc_dialogues_tok"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_example(example):
    # 使用 Qwen 自带 chat template
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

def main():
    ds = load_dataset("json", data_files=DATA_FILE, split="train")
    ds = ds.map(format_example, remove_columns=ds.column_names)
    # 可选：先 tokenize 一遍，避免训练时频繁 tokenize
    # ds = ds.map(lambda ex: tokenizer(ex["text"]), batched=True)
    ds.save_to_disk(OUT_CACHE)
    print("Saved processed dataset to", OUT_CACHE)

if __name__ == "__main__":
    main()
