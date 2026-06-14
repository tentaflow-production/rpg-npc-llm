# Claude Code Instructions — RPG NPC LLM

## Role
You are an expert ML engineer assistant working on a smart NPC dialogue system.
You have full access to the project files. Always read CLAUDE.md first for project context.

## Environment Rules
- Always assume conda env `llm` is active
- Never upgrade or reinstall torch — the nightly cu128 build must be preserved
- Use bf16 (not fp16) for all training — RTX 5090 (sm_120) requires it
- All scripts run inside WSL Ubuntu at ~/projects/rpg-npc-llm/
- Python path: check with `which python` — must show conda llm env path

## Code Style
- Python only (no notebooks for training scripts)
- Use argparse or yaml config files for all hyperparameters — no hardcoded values
- Every training script must accept `--config` argument pointing to a YAML file
- Add logging to ~/logs/ for every script run
- Use tqdm for progress bars
- Type hints on all function signatures

## Directory Layout

### Workstation directories (outside the repo, never committed)
| Path | Contents |
|---|---|
| `~/models/` | Local LLM weights (Qwen, Llama, etc.) |
| `~/datasets/` | SFT and RL datasets, current and future |
| `~/checkpoints/` | Training checkpoints, one subdir per run |
| `~/experiments/` | Experiment snapshots, e.g. `qwen_sft_2025_12_06/` — each contains `config.yaml` and a pip-freeze snapshot |
| `~/logs/` | Training logs, one subdir per run |
| `~/scripts/` | Utility and setup scripts |

### Repo directories (committed)
| Path | Contents |
|---|---|
| `scripts/` | Training and preprocessing Python scripts |
| `configs/` | YAML config files |

### Standard path conventions for scripts
```
DATA_CACHE  = ~/datasets/rpg-npc-llm/<dataset_name>
OUTPUT_DIR  = ~/checkpoints/<run_name>
LOG_DIR     = ~/logs/<run_name>
EXPERIMENT  = ~/experiments/<run_name>_YYYY_MM_DD/
```

The repo must never contain model weights, datasets, checkpoints, or logs.
Processed data (`.jsonl`, Arrow caches) lives in `~/datasets/`, not in the repo.

## When Making Changes

Any change that touches a path, hyperparameter, or interface must be propagated to every file that references it. The dependency chain is:

```
configs/*.yaml   ← single source of truth for all values
    ↓
scripts/*.py     ← read from config, no hardcoded values
    ↓
claude.md        ← documents conventions and path layout
```

Before marking a task done, check:
- If a path changed → update the config YAML, both scripts, and claude.md if it appears there
- If a new hyperparameter was added → add it to the config YAML first, then read it in the script
- If a script's interface changed (new arg, new output) → update claude.md to reflect it
- Never leave a value hardcoded in a script if it already exists in the config

## When Writing Training Scripts
- Always include: seed fixing, wandb logging, checkpoint saving, gradient checkpointing
- Default batch size should account for 32GB VRAM — suggest 4-8 for 7B models with LoRA
- Always save config.yaml snapshot into the experiment directory at run start
- Handle keyboard interrupt gracefully (save checkpoint before exit)

## When Writing RAG Code
- Vector store: ChromaDB (already installed)
- Embedding model: sentence-transformers (already installed)
- Keep retrieval logic modular — separate from inference logic
- Document schema must include: character, scene, content, metadata

## Git Workflow
- After completing any task that modifies files, suggest a git commit
- Commit message format: `<type>: <short description>`
  - types: feat / fix / data / config / refactor / docs
- Never commit: model weights, .jsonl datasets, __pycache__, .env files

## Experiment Reproducibility
- Every experiment must have a config.yaml with:
  - model_id, lora_config, training_args, dataset_path, seed, date, notes
- Log the full config at the start of every training run
- Save pip freeze output alongside each experiment

## Response Style
- Be concise — show the code, explain only non-obvious decisions
- When editing existing files, show only the changed sections
- Always verify file paths exist before writing
- If a command might take >1 min, warn the user first

## Session Log

At the end of every session, append an entry to `devlog.md` in the project root with this format:

### YYYY-MM-DD
**Built / Changed**
- <what was added or modified>

**Key Decisions**
- <any architectural or design choices made>

**Next Steps**
- <what to pick up next session>

---

Always append, never overwrite previous entries. If devlog.md doesn't exist, create it.