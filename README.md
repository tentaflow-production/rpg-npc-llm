# RPG NPC LLM

Fine-tune a large language model to play fantasy RPG non-player characters. Built on **Qwen2.5-7B-Instruct** with LoRA adapters trained via supervised fine-tuning (SFT), with DPO/PPO reinforcement learning planned as a second stage.

The goal is an NPC that stays in character, knows the game world's lore, and responds naturally to player input — not a generic chatbot.

```
Qwen2.5-7B-Instruct (base)
    └── SFT / LoRA  — dialogue style, world lore, NPC voice
            └── RL (DPO / PPO)  — in-character behaviour and preference alignment
```

---

## Requirements

- Conda environment `llm` (Python 3.10, PyTorch nightly cu128)
- RTX 5090 or any GPU with bf16 support
- WSL2 Ubuntu

Activate the environment before running anything:

```bash
conda activate llm
```

---

## Directory layout

| Location | Contents |
|---|---|
| `~/projects/rpg-npc-llm/` | This repo — code and configs only |
| `~/datasets/rpg-npc-llm/` | Raw JSONL + tokenized Arrow cache |
| `~/checkpoints/` | LoRA adapter weights from each run |
| `~/logs/` | Training logs |
| `~/experiments/` | Per-run config snapshots for reproducibility |

---

## Running the scripts

All scripts take a `--config` argument pointing to a YAML file in `configs/`.

### Step 1 — Tokenize the dataset

Reads the raw JSONL, applies Qwen's chat template, and saves an Arrow cache to disk. Only needs to be re-run when the dataset changes.

```bash
python scripts/prepare_dataset.py --config configs/qwen_npc_v1.yaml
```

Output: `~/datasets/rpg-npc-llm/npc_dialogues_tok/`

### Step 2 — Train

Loads the Arrow cache, attaches LoRA adapters to the base model, and runs SFT training. Checkpoints are saved every 200 steps; the final adapter is written to `~/checkpoints/qwen-npc-lora-v1/`.

```bash
python scripts/train_qwen_sft.py --config configs/qwen_npc_v1.yaml
```

Training time is roughly 30–60 min per epoch on an RTX 5090 for a small dataset.

### Creating a new experiment config

Copy the base config and edit the fields you want to change:

```bash
cp configs/qwen_npc_v1.yaml configs/qwen_npc_v2.yaml
# edit paths.output_dir, lora.r, training.learning_rate, etc.
python scripts/train_qwen_sft.py --config configs/qwen_npc_v2.yaml
```

---

## Adding data to the dataset

Training examples live at `~/datasets/rpg-npc-llm/npc_dialogues.jsonl`. Each line is a JSON object with a `messages` array in OpenAI chat format:

```jsonl
{"messages": [
  {"role": "system",    "content": "You are a gruff blacksmith NPC in a fantasy RPG. You speak in short sentences and distrust strangers."},
  {"role": "user",      "content": "Can you make me a sword?"},
  {"role": "assistant", "content": "Depends. You got coin? And a reason I should trust you?"}
]}
```

**Rules for good training examples:**

- `system` — describes the NPC's identity, tone, and any world context they would know
- `user` — what the player says (keep it natural, vary the phrasing)
- `assistant` — the NPC's reply; this is what the model learns to produce

After adding examples, re-run `prepare_dataset.py` to rebuild the Arrow cache before training.

---

## Inspecting checkpoints

Checkpoints are saved as LoRA adapter weights (not full model weights). Each checkpoint directory contains `adapter_config.json` and `adapter_model.safetensors`.

### List saved checkpoints

```bash
ls ~/checkpoints/qwen-npc-lora-v1/
```

### Load and run inference from a checkpoint

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id  = "Qwen/Qwen2.5-7B-Instruct"
checkpoint_dir = "/home/seabass1/checkpoints/qwen-npc-lora-v1"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, checkpoint_dir)
model.eval()

messages = [
    {"role": "system",  "content": "You are an NPC innkeeper in a fantasy RPG."},
    {"role": "user",    "content": "Any rumors in town lately?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.9, top_p=0.9, do_sample=True)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Load a mid-training checkpoint (step-level)

```bash
ls ~/checkpoints/qwen-npc-lora-v1/checkpoint-200/
```

Pass `checkpoint-200` as the checkpoint path in `PeftModel.from_pretrained` instead of the top-level directory.

### Inspect adapter config

```bash
cat ~/checkpoints/qwen-npc-lora-v1/adapter_config.json
```

Shows the LoRA rank, target modules, and base model the adapter was trained on.
