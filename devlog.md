# Devlog

---

### 2026-06-14 — Training side: plain Trainer + smoke pipeline

**Built / Changed**
- Rewrote `scripts/train_qwen_sft.py`: dropped TRL `SFTTrainer`/`SFTConfig`; now a plain `transformers.Trainer` consuming the pre-tokenized `input_ids`/`attention_mask`/`labels` directly, with `DataCollatorForSeq2Seq(label_pad_token_id=-100)`, `device_map={"":0}`, LoRA + `enable_input_require_grads()` + `use_cache=False` (correct grad-ckpt order), an `NpcTrainer.compute_loss` pass-through seam (keeps `**kwargs` for `num_items_in_batch`), `set_seed`, graceful `KeyboardInterrupt` save, config+pip-freeze experiment snapshot, and always-on pre-flight checks (column guard, label sanity, advisory initial-loss read).
- Factored pure helpers (`build_training_kwargs`, `experiment_snapshot_dir`, `take_first_n`, `assert_dataset_columns`) and covered them in `tests/test_train_qwen_sft.py` (14 offline tests; suite now 30 green).
- Configs: fixed `qwen_npc_v1.yaml` (placeholder `data_cache` + comment, `report_to: wandb`, `gradient_checkpointing`, `run_name`, `experiment_dir`, `max_steps: -1`; dropped dead `data_file`); added `configs/smoke.yaml` (same model; 8 rows, 60 steps, grad-accum 1, grad-ckpt ON — with the two IMPORTANT comments).
- Added `notebooks/smoke_test.ipynb` (one-off inspection: masking → collation → model+LoRA → initial-loss → overfit-to-0 → optional generate), reading everything from `smoke.yaml` and importing the real training helpers.
- Added `docs/training.md` (architecture, the grad-ckpt gotcha, smoke vs actual, symptom→cause table) and linked it near the top of `README.md`; updated README training section + repo layout.
- `.vscode`/tooling: added `pyrightconfig.json` (`extraPaths: ["scripts"]`); `.gitignore` += `.ipynb_checkpoints/`.
- Installed (cleanly, dry-run-guarded, torch/numpy/transformers/triton untouched): `ipykernel`, `matplotlib` (pinned in `requirements.txt`); `basedpyright` + `nbconvert` as dev/CI tools (not pinned).

**Verification**
- basedpyright: 0 errors on touched files (`infer.py` has pre-existing errors — untouched, out of scope).
- Tier 1: `pytest` 30 green; `train_qwen_sft.py --help` + `prepare_dataset.py --help` run.
- Tier 2 (real GPU smoke, `configs/smoke.yaml`): pre-flight passed (375 supervised / 1673 masked), initial loss 3.49, trainable 0.527%, overfit 3.27 → 0.0005, snapshot + adapter saved.
- Notebook executed headless (no cell errors); outputs cleared before commit.

**Key Decisions**
- `prepare_dataset` owns tokenization+masking; training only consumes it → plain `Trainer`, not `SFTTrainer`. `compute_loss` is a seam (keep `**kwargs`).
- Smoke is "small" via subset+steps, never by swapping the model (the tokenized cache is tokenizer-specific). Grad-ckpt stays ON in smoke so it exercises the `enable_input_require_grads` path.
- Type fixes by narrowing (`cast` to `PreTrainedModel`), never `# type: ignore`.
- Notebooks are inspection-only; the training path is always a script (reconciled in CLAUDE.md).

**Next Steps**
- For a real run: re-run `prepare_dataset.py`, paste the printed tokenized path into `qwen_npc_v1.yaml` `data_cache`, `wandb login`, then `train_qwen_sft.py --config configs/qwen_npc_v1.yaml`.
- Data lineage / manifest was out of scope this pass.

---

### 2026-06-14

**Built / Changed**
- Rewrote `scripts/prepare_dataset.py` to genuinely tokenize the dataset pipeline output: reads `~/datasets/rpg-npc-llm-dataset-pipeline/build/dialogues_built.jsonl`, applies Qwen's chat template with `tokenize=True`, and saves `input_ids` / `attention_mask` / `labels` to `~/datasets/rpg-npc-llm-dataset-pipeline/tokenized/`
- Added **label masking** (`-100` everywhere except assistant tokens) so loss is computed only on the NPC reply, not the long persona system prompt or the player's lines
- Added **left-truncation** at `training.max_length` so the assistant target at the tail survives instead of being cut by right-truncation (2/111 rows exceed 1024)
- Gave the script zero-required-arg defaults: `--config` (resolved relative to the file), `--input`, `--output`
- Cleared all Pylance/type-checker warnings: annotated the tokenizer as `PreTrainedTokenizerBase` (not the `AutoTokenizer` factory), `assert isinstance(ds, Dataset)` to narrow the `load_dataset` union, and an `encode_chat()` helper that `cast`s `apply_chat_template` output to `list[int]`; switched row-iteration to column access
- Converted the module header into a proper docstring (now drives `--help`)
- Added project skill `.claude/skills/code-hygiene/` documenting the cleanup standard; extended it to **require unit tests** for changed logic (added section + checklist item)
- Added `tests/test_prepare_dataset.py` (14 `unittest.TestCase` cases) covering assistant-only label masking, multi-round, left-truncation keeping the tail, shape alignment, and config parsing — uses a prefix-consistent **fake tokenizer** so tests run offline with no model download
- Installed `pytest` (9.1.0) into the `llm` env — dry-run confirmed only 3 small pure-Python additions (`pytest`, `pluggy`, `iniconfig`), no upgrades, torch nightly untouched; added those 3 pins to `requirements.txt`. Tests stay `unittest`-compatible so both runners work
- Fixed Pylance "Import 'prepare_dataset' could not be resolved" in tests by adding `python.analysis.extraPaths: ["scripts"]` to `.vscode/settings.json` (the runtime `sys.path.insert` in the test handles execution; extraPaths handles static analysis); documented the pattern in the code-hygiene skill
- README: added committed repo structure (`scripts/`/`configs/`/`tests/`) to the directory layout, and documented `prepare_dataset.py`'s label-masking + left-truncation behavior in Step 1
- Typed `encode_chat`/`tokenize_example` against a minimal `SupportsChatTemplate` `Protocol` instead of the concrete `PreTrainedTokenizerBase`, so the test's `FakeTokenizer` satisfies the signature structurally (no cast/subclassing) — fixes the Pylance "not assignable" error; documented the pattern in the code-hygiene skill
- `prepare_dataset.py` now saves to a **timestamped** subdir `<output>/dialogues_built_<YYYYmmdd_HHMMSS>` (new pure helper `tokenized_dir`, unit-tested) so re-runs don't clobber prior caches; cleaned stale top-level arrow files left in `tokenized/` by earlier non-timestamped runs
- Updated `README.md` (directory layout, Step 1 tokenize, "Adding data", new Tests section) for the new pipeline paths and tokenized output

**Key Decisions**
- `prepare_dataset.py` now emits pre-tokenized token IDs with masked labels, not a `text` field — the `tokenized/` cache name reflects what it holds
- Left-truncate rather than filter the 2 over-length rows, to preserve the supervised target; recommended bumping `max_length` to ~1152 to keep them fully intact (not yet applied)
- Type warnings fixed by narrowing (`cast`/`assert isinstance`), never `# type: ignore`
- New cleanup standard lives as a **project** skill (committed with the repo) rather than personal, since it references repo files (devlog/README/requirements) and complements `claude.md`
- `requirements.txt`: unchanged — no new third-party imports (`pathlib`, `typing.cast` are stdlib; `PreTrainedTokenizerBase` ships with the existing `transformers` dep)

**Next Steps**
- Wire training to the pre-tokenized cache: point `paths.data_cache` at `.../tokenized` and drop `dataset_text_field="text"` in `train_qwen_sft.py` so `SFTTrainer` consumes the masked `input_ids`/`labels` directly (currently still expects a `text` field at the old path)
- Optionally set `training.max_length: 1152` to keep all 111 rows un-truncated
- Smoke-run `train_qwen_sft.py` against the new tokenized cache
