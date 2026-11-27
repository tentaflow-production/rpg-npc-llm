# SPEC_SUMMARY — M2 (SFT dataset v2 from canon) — spec run 2026-11-27

Nothing is implemented. 4 stubs raise `NotImplementedError`; 110 new tests are collected, 105 are red, 5 are green oracle self-checks.
Run: `python -m pytest tests/test_prompt_contract.py tests/test_dataset.py tests/test_scenes_from_canon.py -q`.

## Every `[inferred]` value first
| Value | Where | Source tag |
|---|---|---|
| All Chinese wording: scene line, belief line, knowledge/withheld/stance lines, style rules, action strings | `data/templates/render.yaml` | inferred (contract fixes layout, not strings) |
| Character cards = one-sentence placeholders | `data/templates/cards.yaml` | inferred; traits from 04 name table; replaced by `data/characters/*.md` later |
| Belief bucket ids + texts (`unaware`, `heard_rumour`, `suspects_dorian`, `suspects_aurelia`, `convinced_dorian`) | `data/templates/beliefs.yaml` | inferred (05 §8 names states loosely) |
| Clue texts for g1–g9, p1t/p3t/p4t/p4s, t1, i1–i3 (ids from 05 §7) | `data/templates/clues.yaml` | ids: 05 §7; wording inferred |
| Stance sets: Cecily no `hostile`; Dorian `charming/probing/cold` | `data/templates/stances.yaml` | contract text; Aurelia/Bohan/Elwin sets inferred |
| Off-script topic list, input types | `data/templates/topics.yaml` | Q20 answer; list inferred |
| Scene names (04); the Alley as 9th scene, empty every slot | `render.yaml`, tests | 04 scene list; Alley = 05 §5 / Q11 (inferred) |
| 1 turn = 1 (action, line) pair, max 6; token estimate = non-ASCII chars + ceil(ASCII/3); budget 600/1024 | `render.yaml`, tests | contract invariant 5; estimator inferred |
| `data/MIN_N` = 300 | file | Q20 answer ("start 300") |
| `data/LANGUAGE` = `zh` | file | Q19 + default of Q36 |
| `scenes.yaml` schema (`timeline`, per-slot `conflicts`, `epilogue_only`, `dead_drop`) | `tests/test_scenes_from_canon.py` | inferred |
| Row schema, dataset-dir layout, manifest keys, verdict JSON, violation codes | `tests/test_dataset.py` header | inferred |
| `secrets.yaml`, `canon_phrases.txt` | `data/` | copied byte-identical from the M1 branch (all inferred there) |

## What the oracles encode
- **Prompt contract** (6 fixtures, one per NPC plus an alternate scenario and a 7-turn history): exact rendered messages; knowledge holds only permitted clue ids; withheld secrets appear by topic label only and their patterns never occur in the prompt; belief shown as text, never as bucket id; closed action set; stance set must fit the NPC; `scenario_id` changes only the scene-card line; history cut to 6; token budgets enforced; `en` refused; envelope validator with valid and invalid heads per fixture.
- **scenes_from_canon**: parses 04's matrix (verbatim cells, blanks = closed) and applies C1, C2, C3 (one scene per slot), C4, C5/C9, C6, C7 (D2E), C10, C12; tags appear as `# [conflict Q3-Cn]`; editing a matrix cell in a copy of 04 changes the output.
- **validate_dataset**: min-n, leak rate (spaced-out spellings caught, topic label alone not a leak), per-NPC balance (10:5 ok, 11:5 not), exact and near duplicates, off-script ≥ 0.4 with topic consistency, 13-gram contamination, template-id overlap, manifest presence/completeness/hash, required field, language, envelope validity per row, field-vs-prompt agreement (D19), all violations reported at once.
- **gen_dataset**: offline `fake`/`fake_leaky` backends; candidates only, `reviewed: false`, never a promoted dataset; prompts equal the contract render; both scenarios, ≥ 40 % off-script, NPCs only where present, balanced, deterministic by seed; leaking teacher lines go to `rejected.jsonl`.

## Defaults taken for unanswered questions (all in `docs/QUESTIONS.md`)
Q36 `zh` only · Q37 Dorian at the bistro D2E only · Q38 contamination ignores the system message and fixed action strings · Q39 M1 owns `secrets.yaml` and `canon_phrases.txt`.

## Negative fixtures (what a lazy implementation would get wrong)
- scenes `--check` rejects: xlsx-timeline ending (D10E), Elwin missing at D3A (C1), bistro open on D4A (C9), Day-10 rooftop not epilogue-only (C6), Aurelia on the roof D6M (C3); a canon dir with no matrix is an error, not an empty file.
- validator rejects: confession (also spaced out), duplicate and near-duplicate, 11:5 imbalance, zero-row NPC, 9/25 off-script, 13-gram overlap, shared template id, missing/incomplete/stale manifest, empty `scenario_id`, `en` row, bad stance/reveal/bare line, scenario/npc/slot/scene field that disagrees with its prompt.
- renderer rejects: free-text action, `hostile` for Cecily, `warm` for Dorian, unknown clue id or bucket, oversize prompts.

## Known gaps
- `$DATASETS_DIR/rpg-npc-llm-dataset-pipeline` does not exist on this machine, so done-when line 2 cannot be tried yet; the spec tests use the canon directly.
- The pipeline's real `build_dataset.py` import of `rpgnpc.prompt`, `npc_sft_v2` itself and the 300 reviewed rows are build/human work, not tested here.
- Near-duplicate test margin is 0.90 vs the 0.8 threshold; an exact Jaccard or ≥ 128-permutation MinHash passes.
- Conflicts C8, C11, C13 change no scene data and are not tested; Q6 (actions per slot) is open, C3 uses its default.
