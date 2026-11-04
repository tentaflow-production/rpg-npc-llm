# RPG NPC LLM

Fine-tune a large language model to play fantasy RPG non-player characters. Built on **Qwen2.5-7B-Instruct** with LoRA adapters trained via supervised fine-tuning (SFT), with DPO/PPO reinforcement learning planned as a second stage.

The goal is an NPC that stays in character, knows the game world's lore, and responds naturally to player input — not a generic chatbot.

```
Qwen2.5-7B-Instruct (base)
    └── SFT / LoRA  — dialogue style, world lore, NPC voice
            └── RL (DPO / PPO)  — in-character behaviour and preference alignment
```

📖 **Training details, architecture, and health metrics:** see [docs/training.md](docs/training.md).

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
| `~/datasets/rpg-npc-llm-dataset-pipeline/` | Dataset pipeline — `build/dialogues_built.jsonl` (merged sources) + `tokenized/` per-run timestamped Arrow caches |
| `~/checkpoints/` | LoRA adapter weights from each run |
| `~/logs/` | Training logs |
| `~/experiments/` | Per-run config snapshots for reproducibility |

Inside the repo (committed):

| Path | Contents |
|---|---|
| `scripts/` | Training and preprocessing scripts (`prepare_dataset.py`, `train_qwen_sft.py`, `infer.py`) |
| `configs/` | YAML config files — the single source of truth for paths and hyperparameters (`qwen_npc_v1.yaml`, `smoke.yaml`) |
| `tests/` | Unit tests (`unittest`, runnable under `pytest`) |
| `notebooks/` | One-off inspection/validation only (`smoke_test.ipynb`) — never the training path |
| `docs/` | Project docs (`training.md`) |

---

## Running the scripts

All scripts take a `--config` argument pointing to a YAML file in `configs/`.

### Step 1 — Tokenize the dataset

Reads the built dialogues JSONL, applies Qwen's chat template **with tokenization**, and saves an Arrow cache of `input_ids` / `attention_mask` / `labels` to disk. Single- and multi-round conversations are handled the same way. Two deliberate behaviors:

- **Label masking** — only assistant tokens carry loss; the system persona and the player's `user` turns are masked to `-100`, so the model learns to *be* the NPC rather than to reproduce the prompt.
- **Left-truncation** — rows longer than `training.max_length` keep their tail, preserving the assistant reply (the supervised target) instead of slicing it off the end.

Only needs to be re-run when the dataset changes.

```bash
python scripts/prepare_dataset.py --config configs/qwen_npc_v1.yaml
```

- Input (default): `~/datasets/rpg-npc-llm-dataset-pipeline/build/dialogues_built.jsonl`
- Output: a **timestamped** subdir under the `--output` parent (default
  `~/datasets/rpg-npc-llm-dataset-pipeline/tokenized/`), e.g.
  `tokenized/dialogues_built_20260614_162022/` — so re-runs don't clobber prior
  caches. The script prints the exact path; point training's `data_cache` at it.

Override either with `--input` / `--output`.

### Step 2 — Train

Loads the **pre-tokenized** Arrow cache and trains a LoRA adapter with a plain
`transformers.Trainer` — tokenization and label masking are already done by
`prepare_dataset.py`, so the training side just consumes `input_ids` /
`attention_mask` / `labels` directly (no `SFTTrainer`). First paste the tokenized
path from Step 1 into `paths.data_cache`, then:

```bash
python scripts/train_qwen_sft.py --config configs/qwen_npc_v1.yaml
```

Checkpoints save every `save_steps` to `~/checkpoints/qwen-npc-lora-v1/`, loss
goes to wandb (`wandb login` once), and an experiment snapshot (config + pip
freeze) is written to `~/experiments/<run_name>_<date>/`. Training time is
roughly 30–60 min per epoch on an RTX 5090 for a small dataset.

**Validate the pipeline first** with the smoke run (same model, 8 rows, ~60
steps) — walk `notebooks/smoke_test.ipynb`, then:

```bash
python scripts/train_qwen_sft.py --config configs/smoke.yaml
```

See [docs/training.md](docs/training.md) for the architecture, the
gradient-checkpointing gotcha, and how to read the health metrics.

### Creating a new experiment config

Copy the base config and edit the fields you want to change:

```bash
cp configs/qwen_npc_v1.yaml configs/qwen_npc_v2.yaml
# edit paths.output_dir, lora.r, training.learning_rate, etc.
python scripts/train_qwen_sft.py --config configs/qwen_npc_v2.yaml
```

---

## Tests

Unit tests live in `tests/` and are written as `unittest.TestCase` classes.
They use a fake tokenizer, so they run fast and offline — no model download.
Run with either runner (`pytest` is installed in the `llm` env; `unittest` is
the stdlib fallback):

```bash
python -m pytest tests/ -q
# or, no extra dependency:
python -m unittest discover -s tests
```

`tests/test_prepare_dataset.py` covers the tokenization logic: assistant-only
label masking, left-truncation keeping the answer at the tail, shape alignment,
and config parsing.

---

## Adding data to the dataset

Training examples are produced by the dataset pipeline at `~/datasets/rpg-npc-llm-dataset-pipeline/`, which merges the character/scene/dialogue sources into `build/dialogues_built.jsonl`. Each line is a JSON object with a `messages` array in OpenAI chat format, plus a `metadata` block (dropped at tokenization). Conversations may be single- or multi-round (`system → user → assistant → user → assistant → …`):

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

After rebuilding `dialogues_built.jsonl`, re-run `prepare_dataset.py` to regenerate the tokenized Arrow cache before training.

---

## Inspecting checkpoints

Checkpoints are saved as LoRA adapter weights (not full model weights). Each checkpoint directory contains `adapter_config.json` and `adapter_model.safetensors`.

### List saved checkpoints

```bash
ls ~/checkpoints/qwen-npc-lora-v1/
```

### Run inference with `scripts/infer.py`

Use the final saved adapter (default):

```bash
python scripts/infer.py --config configs/qwen_npc_v1.yaml
```

Use a specific mid-training checkpoint by step number:

```bash
python scripts/infer.py --config configs/qwen_npc_v1.yaml --checkpoint 200
```

Use a full path to any checkpoint directory:

```bash
python scripts/infer.py --config configs/qwen_npc_v1.yaml --checkpoint ~/checkpoints/qwen-npc-lora-v1/checkpoint-400
```

Pass a one-shot prompt directly (non-interactive):

```bash
python scripts/infer.py --config configs/qwen_npc_v1.yaml \
  --system "You are a gruff blacksmith who distrusts strangers." \
  --prompt "Can you make me a sword?"
```

Override generation parameters at runtime:

```bash
python scripts/infer.py --config configs/qwen_npc_v1.yaml \
  --temperature 0.7 --max_new_tokens 300
```

Omitting `--prompt` starts **interactive mode** — type player messages and get NPC responses in a loop. Press `Ctrl+C` or type `quit` to exit.

Default generation parameters (`temperature`, `top_p`, `max_new_tokens`, `default_system`) are set in the `inference:` section of the config YAML and can be overridden per-run with CLI flags.

### Inspect adapter config

```bash
cat ~/checkpoints/qwen-npc-lora-v1/adapter_config.json
```

Shows the LoRA rank, target modules, and base model the adapter was trained on.

### RoadMap
Smoke test
Add Datasets for the story (500 lines for each character)
Add Benchmark (OOC metrics (for tones of the answers), and content evaluation (is she answering the question), and is the answer based on the character's memory (give dummy memory for now)) - use LLama/GPT4 to give scores, and I will also give scores.
Try Finetune with LORA
Try RAG (world setting)
ADD memory module for each character with memory limit (so we proceed with the story, and characters can forget)

### Dataset
{"messages": [
  {"role": "system", "content": "You are an NPC innkeeper in a fantasy RPG."},
  {"role": "user", "content": "Hi, I just arrived in town, what can you tell me?"},
  {"role": "assistant", "content": "Welcome, traveler! Our town looks peaceful, but odd rumors are spreading about the old mine to the north..."}
]}

