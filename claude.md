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

## File Conventions
- Training scripts → scripts/
- Config files → configs/ (YAML format)
- Raw data → data_raw/
- Processed data → data_processed/ (.jsonl format)
- Each experiment → ~/experiments/<name>_YYYY_MM_DD/
- Never write model weights or large datasets into the git repo

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