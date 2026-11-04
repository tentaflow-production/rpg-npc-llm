# Training

Fine-tune a LoRA adapter on the pre-tokenized NPC dialogue dataset.

There are two modes:

| Mode | What | When | Saves anything? |
|---|---|---|---|
| **Smoke** | `scripts/train_qwen_sft.py --config configs/smoke.yaml` | Validate the pipeline end to end before a real run, or after changing the training code | No — throwaway dirs |
| **Actual** | `scripts/train_qwen_sft.py --config configs/qwen_npc_v1.yaml` | The real training run | Yes — checkpoints + an experiment snapshot, logged to wandb |

Smoke proves the *plumbing* (masking → collation → LoRA → loss goes down). Actual is the reproducible run you keep.

---

## Architecture: who owns what

**Tokenization and label masking are owned entirely by [`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py).** It turns dialogues into `input_ids` / `attention_mask` / `labels`, where `labels` is `-100` everywhere except the assistant (NPC) tokens. That logic is unit-tested in [`tests/test_prepare_dataset.py`](../tests/test_prepare_dataset.py).

**The training script only *consumes* those tensors.** It does not tokenize and does not mask.

### Why a plain `Trainer`, not TRL's `SFTTrainer`

`SFTTrainer` is built to tokenize and mask *itself* (via `dataset_text_field`, completion-only collators, etc.). Feeding it our already-masked data invites a conflict: whether it preserves our `labels` is undocumented and version-dependent in trl 0.25.1. So the script uses a plain `transformers.Trainer` consuming the pre-tokenized dataset directly. **Loss is then exactly the cross-entropy on the labels we already unit-tested** — no in-library preprocessing black box.

### The pieces that make this work

- **`DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)`** — *required*. The **default** collator does **not** pad `labels`, so a padded batch would have `labels` shorter than `input_ids` and silently misalign. This collator pads `labels` with `-100` (ignored by the loss). We do **not** pass `model=` to it — that triggers encoder-decoder `decoder_input_ids` logic we don't want for a causal LM.

- **Gradient checkpointing + `enable_input_require_grads()`** — the gotcha. Under gradient checkpointing the base model's inputs don't require grad by default, so gradients never reach the LoRA adapters and **training silently does nothing** (flat loss). The fix is the exact order in the script:
  1. `model = get_peft_model(model, lora_config)`
  2. `model.enable_input_require_grads()`  ← makes grads flow to LoRA under ckpt
  3. `model.config.use_cache = False`  ← KV cache is incompatible with ckpt

  Checkpointing itself is enabled **once**, via `TrainingArguments(gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})` — we do not also call `gradient_checkpointing_enable()` (double-enabling is a footgun). The smoke config keeps checkpointing **ON** on purpose, so smoke exercises this exact path; turning it off for smoke would let smoke pass while real training fails.

- **`compute_loss` seam.** `NpcTrainer` overrides `compute_loss` as a pass-through:
  ```python
  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
      return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
  ```
  It does nothing today — it's a hook for future loss terms (e.g. preference/RL shaping). The `**kwargs` is **not cosmetic**: newer `transformers` passes `num_items_in_batch` through `compute_loss` to scale the loss correctly under gradient accumulation. Dropping it makes the loss wrong, so it must stay.

- **Single-GPU load.** Training uses `device_map={"": 0}` (everything on GPU 0). We deliberately do **not** copy inference's `device_map="auto"`, which can offload layers to CPU and breaks `Trainer`'s device control.

---

## Smoke procedure

The smoke run is "small" via an **8-row subset + ~60 steps**, *not* by swapping models — it uses the same `Qwen2.5-7B-Instruct` as the real run, because the tokenized cache is tokenizer-specific (a smaller model would need its own cache and risks a silent tokenizer/embedding mismatch). Everything is config-only via `configs/smoke.yaml`.

### What a healthy smoke shows

The script runs these checks headless (pre-flight checks below, then the overfit); `tests/test_prepare_dataset.py` covers masking offline. Read the output against:

1. **Masking (no GPU).** Decode the tokens where `label != -100` for one row — it should read as the **NPC's reply**, not the system persona or the player's line. Supervised fraction strictly between 0 and 1.
2. **Collation (no GPU).** Collate two rows; all three tensors share a shape; the shorter row's padded label tail is all `-100`.
3. **Model + LoRA (GPU, loads the 7B — >1 min).** Trainable params print ~0.05%–2%. One forward gives an initial-loss read (see magnitude guide below).
4. *(covered within step 3 — the initial-loss read)*
5. **Overfit (GPU).** Train the first 8 rows for ~60 steps with checkpointing ON; the loss curve should fall toward ~0 (final loss < 0.5). Flat loss ⇒ suspect the missing `enable_input_require_grads`.

### Running it

```bash
python scripts/train_qwen_sft.py --config configs/smoke.yaml
```

This runs the full path, including the always-on pre-flight checks (below), and overfits the 8 rows. Throwaway output (`*_smoke` dirs); nothing to keep.

---

## Actual procedure

1. **Tokenize** (only when the dataset changed) and copy the printed path:
   ```bash
   python scripts/prepare_dataset.py
   # -> .../tokenized/dialogues_built_<YYYYmmdd_HHMMSS>
   ```
2. **Point the config at it.** Paste that path into `paths.data_cache` in `configs/qwen_npc_v1.yaml` (it ships as a `REPLACE_ME` placeholder — the timestamp changes every run).
3. **Log in to wandb once** (the real config sets `report_to: "wandb"`):
   ```bash
   wandb login
   ```
4. **Train:**
   ```bash
   python scripts/train_qwen_sft.py --config configs/qwen_npc_v1.yaml
   ```

What you get:
- **Loss curves in wandb**, under the run named by `run_name`.
- **Checkpoints** every `save_steps` in `paths.output_dir` (LoRA adapter weights), plus a final adapter + tokenizer.
- **An experiment snapshot** at `~/experiments/<run_name>_<YYYY_MM_DD>/` containing `config.yaml` and a `pip_freeze.txt` of the `llm` env, for reproducibility.

Interrupting with `Ctrl+C` saves the current adapter before exiting.

---

## Always-on pre-flight checks

Every invocation of the script runs three cheap checks before training (they fail fast):

- **(a) Dataset columns** are exactly `{input_ids, attention_mask, labels}` — a stray `text`/`messages` column means the wrong cache was passed. *Hard assert.*
- **(b) Label sanity** — after one batch through the collator, `labels` contain **both** supervised tokens **and** `-100`. All-masked or all-supervised ⇒ misalignment. *Hard assert.*
- **(c) Initial-loss read** — one forward pass prints the starting loss and a magnitude hint. *Advisory (prints, does not assert).*

---

## How to view loss & health metrics

| Signal | Smoke (script) | Actual (script + wandb) |
|---|---|---|
| **Loss curve** | printed each step (`logging_steps: 1`) | wandb run page (live) |
| **Initial-loss read** | printed by pre-flight (c) | printed by pre-flight (c) at run start |
| **Trainable params %** | `print_trainable_parameters()` at run start | printed at run start |
| **Supervised-token fraction** | pre-flight (b) label check; `tests/test_prepare_dataset.py` | (validated once via smoke) |
| **Overfit-to-0** | read the last printed loss (< 0.5) | n/a (real data shouldn't hit 0) |

**Healthy reads:**
- **Initial loss ≈ 1–3.** Near `ln(vocab) ≈ 11.6` ⇒ likely label misalignment. `0` or `NaN` ⇒ a real problem.
- **Trainable params ≈ 0.05%–2%** for LoRA on a 7B. Much higher ⇒ you're training too much; ~0% ⇒ LoRA didn't attach.
- **Supervised-token fraction ∈ (0, 1).** `0` ⇒ everything masked (loss can't learn); `1` ⇒ nothing masked (training on the prompt).
- **Overfit-8 → ~0.** If it can't memorize 8 rows, the gradient path is broken.

### Symptom → cause → where to check

| Symptom | Likely cause | Where to look |
|---|---|---|
| Loss is exactly `0` | A batch is entirely `-100` labels | pre-flight (b); `prepare_dataset` masking |
| Loss is `NaN` | Numeric issue, or an all-`-100` batch | pre-flight (b/c); dtype (must be bf16) |
| Initial loss ≈ `ln(vocab)` (~11.6) | Label misalignment (labels shifted vs inputs) | the collator (must be `DataCollatorForSeq2Seq`); `prepare_dataset` |
| **Loss flat during training** | Grads not reaching LoRA — the grad-ckpt gotcha | `enable_input_require_grads()` present and ordered after `get_peft_model`; `use_cache=False` |
| Trainable % ≈ 0 | LoRA didn't attach | `get_peft_model` / `target_modules` in the config |

---

## Files

| File | Role |
|---|---|
| [`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py) | Owns tokenization + masking → tokenized Arrow cache |
| [`scripts/train_qwen_sft.py`](../scripts/train_qwen_sft.py) | Consumes the cache; plain `Trainer` + LoRA; the `compute_loss` seam |
| [`configs/qwen_npc_v1.yaml`](../configs/qwen_npc_v1.yaml) | The real run |
| [`configs/smoke.yaml`](../configs/smoke.yaml) | Smoke: same model, shrunk data + steps |
| [`tests/test_train_qwen_sft.py`](../tests/test_train_qwen_sft.py) | Offline unit tests for the training logic |
