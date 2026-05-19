# Devlog

---

### 2026-04-29

**Built / Changed**
- Ran full end-to-end audit of the repo — identified leading-space path bug in `train_qwen_sft.py`, misplaced script in `data_processed/`, stale gitignore entries, and missing directories
- Defined workstation vs repo directory layout: data → `~/datasets/`, checkpoints → `~/checkpoints/`, logs → `~/logs/`, experiments → `~/experiments/`
- Updated `claude.md` with full directory layout table, path conventions, and new "When Making Changes" propagation rule
- Fixed `scripts/train_qwen_sft.py`: removed leading spaces from all path constants, corrected paths to workstation directories, changed `torch_dtype="auto"` to `torch.bfloat16`
- Fixed `scripts/prepare_dataset.py`: corrected paths to `~/datasets/rpg-npc-llm/`
- Deleted `data_processed/` from the repo; moved `npc_dialogues.jsonl` to `~/datasets/rpg-npc-llm/`
- Removed empty unused directories: `data_raw/`, `models/`, `notebooks/`
- Updated `.gitignore`: removed `data_raw/` and `data_processed/` (data no longer lives in the repo)
- Created `configs/qwen_npc_v1.yaml` with all hyperparameters, paths, LoRA config, seed, date, and notes
- Refactored both scripts to accept `--config` argument — all hardcoded values removed, everything reads from the YAML

**Key Decisions**
- Data, checkpoints, and logs live entirely outside the repo at workstation-level paths — repo holds only code and configs
- `configs/*.yaml` is the single source of truth for all values; scripts contain no hardcoded hyperparameters
- `torch.bfloat16` set explicitly (not `"auto"`) to guarantee bf16 on RTX 5090 (sm_120)

**Next Steps**
- Build out the actual NPC dialogue dataset (currently only 2 placeholder examples)
- Run `prepare_dataset.py` to regenerate the Arrow cache at `~/datasets/rpg-npc-llm/npc_dialogues_tok`
- Do a smoke-run of `train_qwen_sft.py` to verify the config→script pipeline end to end
- Add seed fixing, gradient checkpointing, and wandb logging to `train_qwen_sft.py`

---
