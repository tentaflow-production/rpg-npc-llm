# SPEC_SUMMARY — M1 Evaluation harness and baseline #0

**Read first (Q32):** `data/eval_set.jsonl` and `data/eval_adversarial.jsonl` exist in the working tree of the box but **cannot be committed** (`*.jsonl` is ignored and refused by the pre-commit hook, H6). Only their `.sha256` files are in git, so `sha256sum -c` fails on a fresh clone until you answer Q32 (default: exempt exactly those two files and commit them by hand). Also open: the rows and `data/secrets.yaml` spell out the Moonstone solution in a **public** repo (D26).

## What the oracles encode
- `data/eval_set.jsonl`: 240 ordinary rows = 5 NPCs × 8 contexts × 6 action types (listen, small_talk, silent, place_item, buy, deliver). `data/eval_adversarial.jsonl`: 112 probes = 14 secrets × 8 probes (topic questions, pressure after a refusal, silence after pressure, item placement), each addressed to the NPC that holds the secret, each with `target_secret_id`.
- Every row stores contract-rendered `messages` (system card, scene, `Day N slot`, `scenario_id`, belief line, knowledge, **withheld topics as labels only**, stance set, style), a `reference` envelope `{"stance","reveal","line"}` (clean, used for `eval_loss`), and the structured fields.
- `data/secrets.yaml`: 14 secrets (Dorian 7, Bohan 2, Aurelia 3, Cecily 1, Elwin 1) from case bible §1/2/4/7 with topic labels and surface patterns. `data/canon_phrases.txt`: names and scene names.
- Tests (`tests/test_eval.py`, 43): 15 oracle tests pass today (sizes, stratification, hashes, prompt-contract invariants, "no secret pattern appears in any prompt", references are clean and valid); 28 implementation tests are red (matcher, scoring, `eval.json` builder, gate compatibility, CLI, judge, ledger).

## `[inferred]` values (all of them; none is canon wording)
1. All Chinese prompt text, the one-sentence NPC cards, belief line, reference lines and probe wording — inferred (style from `prompt_contract.md`).
2. The 14 secret ids/topic labels/patterns — source: case bible §1, §2, §3, §4, §7 (paraphrased); patterns are guesses at surface forms.
3. Contexts (npc × scene × day × slot): 32 of 40 from canon 04 presence matrix; 4 Bohan contexts (back office D3M/D5M/D7M, street D9M) inferred to avoid duplicate prompts.
4. Stance sets per NPC: contract says "subsets of data/stances.yaml" (not existing); I used warm|evasive|probing|confiding (Dorian: charming|probing|cold) — contract text.
5. Clue texts g1, g5, g7 reworded to carry no secret pattern; `scenario_id` is `default` everywhere; every row `language: zh`.
6. Matcher: NFKC + whitespace removal, `re:` prefix for regex; `--limit N` = N/2 ordinary + N/2 adversarial, spread over NPCs; system prompt ≤ 900 characters as proxy for 600 tokens.
7. `eval.json` extras: `smoke`, `limit`, `backend`, `stance_valid`, `reveal_valid` per prompt; `experiments/LEDGER.md` columns = PROTOCOL step 4 list.

## Defaults taken for open questions
Q32 → exempt the two files; Q33 → `--no-server` = offline reference backend, `eval_loss` from server `prompt_logprobs`; Q34 → rows carry rendered messages.

## Not verifiable offline (honest limits)
The "base model leaks 10–30 %" headroom is **not** checked by any test: it needs the real 7B on the adversarial set (the done-when's `--baseline-free` gate). If it fails, this adversarial set needs rework, not the model. Real-server paths of `eval.py` (server start, sampling, judge calls) have no test; the gate-compat tests need `SUPERVISOR_REPO` (skipped otherwise).

## Negative fixtures
Matcher: topic label alone is not a leak; clean text; permitted ids. `build_eval_json`: missing output, duplicate id, sampled leak decoding. Gate: 0 % adversarial leaks → headroom FAIL. CLI: `--limit`/`--no-server` into `exp_*`, sampled decoding, missing or altered eval set. `load_secrets`: duplicate id, empty patterns, no topic. Judge: out-of-range or malformed replies.
