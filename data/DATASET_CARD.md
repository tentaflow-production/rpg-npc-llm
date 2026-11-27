# Dataset card — `npc_sft_v2` (SFT dataset v2 from canon)

Status: **template** (M2 build). The dataset itself lives in `$DATASETS_DIR/npc_sft_v2/` and is never committed. This card is
completed when a human promotes reviewed candidates; the numbers below come from `manifest.json` and the validator verdict.

## What it is
Chat-format SFT rows for five NPCs (Aurelia, Bohan, Cecily, Dorian, Elwin) of the demo game. Each row is the prompt rendered by
`rpgnpc.prompt` per `docs/prompt_contract.md` (system message, up to 6 history turns, one closed-set player action) followed by
the assistant envelope `{"stance", "reveal", "line"}`.

## Principles
- **Facts live in the prompt, never in the weights (D19).** Every row carries `scenario_id`; alternative scenarios are different
  prompts, never contradictory completions under the same prompt.
- **Secrets are withheld by topic label only.** The content of a secret is never in a prompt; `data/secrets.yaml` patterns never
  occur in a completion (`validate_dataset.py --max-leak-rate 0.0`).
- **Scenes come from canon.** `scripts/scenes_from_canon.py` generates `scenes.yaml` from the presence matrix of canon 04 plus the
  conflict defaults (C1–C12 tagged `[conflict Q3-Cn]`); NPCs appear only where the matrix places them.
- **Language:** `data/LANGUAGE` (zh). Rows in another language are rejected.

## Provenance
- Candidates: `scripts/gen_dataset.py` (teacher job, `job_dispatch.sh --type gen`) writes `build/candidates/` with `reviewed: false`;
  a human review promotes rows (Q20). Teacher model and sha, template sha, seed and decoding are recorded in `manifest.json`.
- Voice cards and few-shots of `rpg-npc-llm-dataset-pipeline` are the teacher's style reference (translated, Q19).

## Coverage and filtering
- Stratified over scenes × `player_input_type` × topic; at least 40 % **off-script** rows (small talk, island daily life, prices and
  weather, other NPCs, refusals, adversarial probes), enforced by `--min-offscript 0.4`.
- Filters: matcher-flagged rows dropped; **no filtering on the uncalibrated local judge**; character 5-gram MinHash dedup (Jaccard ≥ 0.8);
  per-NPC balance within 2×.
- Contamination: no character 13-gram shared with `data/eval_set.jsonl` or `data/eval_adversarial.jsonl` (prompts and completions,
  canon-phrase allowlist `data/canon_phrases.txt`), and no shared template id.

## Known limits
- Wording, character cards, belief and clue texts are `[inferred]` placeholders (`data/templates/`) until human-written versions replace them.
- The matcher is uncalibrated until M4 reports recall; a clean verdict means "no matcher-detected leak", nothing more.
- Q6 (actions per slot) is open; scenes use the one-scene-per-slot default (C3).
