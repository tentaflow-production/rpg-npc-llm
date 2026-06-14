---
name: code-hygiene
description: Code cleanup standard for this repo's Python scripts. Use when finishing or reviewing an edit to a script in scripts/ — to clear Pylance/type-checker warnings, ensure CLI defaults (config + paths) exist, add/extend unit tests, and propagate changes to devlog.md, README.md, and requirements.txt. Trigger when the user mentions "clean up", "fix warnings", "Pylance", type errors, tests, or after any non-trivial code change.
---

# Code Hygiene Standard

Apply this before calling any non-trivial edit to a `scripts/*.py` file "done".
It complements `claude.md` (which owns the config→script→docs dependency chain).

## 1. Clear type-checker (Pylance) warnings

These are the recurring patterns in this codebase and their fixes. The theme:
the library's *declared* type is broader than what we actually get at runtime —
narrow it explicitly rather than ignoring the warning.

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot access attribute "X" for class "AutoTokenizer"` | `AutoTokenizer` is a **factory**; only `from_pretrained` lives on it. The methods (`apply_chat_template`, `encode`, …) live on the returned instance. | Annotate the variable/param as `PreTrainedTokenizerBase`, not `AutoTokenizer`. Keep `AutoTokenizer` imported only for the `from_pretrained` call. |
| `remove_columns` / `save_to_disk` / `len()` / `ds[...]` errors on a `load_dataset` result | `load_dataset(...)` is typed as a 4-way union (`Dataset \| DatasetDict \| IterableDataset \| IterableDatasetDict`); `split="train"` collapses it to `Dataset` but the checker can't know. | `assert isinstance(ds, Dataset)` right after `load_dataset` to narrow it for the rest of the function. |
| `No overloads for "__getitem__"/"__setitem__"` when slicing / indexing chat-template output | `apply_chat_template`'s return type is a broad union (`str \| list[int] \| Encoding \| BatchEncoding \| …`) because it depends on runtime args. | Wrap the call in a helper that `cast(list[int], …)` once; with `tokenize=True` and no `return_tensors` the result really is `list[int]`. Reuse the helper at every call site. |
| `__getitem__` warning when doing `for row in ds: row["col"]` | Iterating a `Dataset` yields a loosely-typed row. | Prefer column access — `ds["col"]` returns a typed `list` — over per-row dict indexing. It's also faster. |
| `Import "<module>" could not be resolved` in a test importing from `scripts/` | The import resolves at runtime via `sys.path.insert(...)`, which Pylance can't follow statically. | Add the source dir to `python.analysis.extraPaths` in `.vscode/settings.json` (e.g. `["scripts"]`), or a `pyrightconfig.json` with `extraPaths`. Keep the runtime `sys.path.insert` for execution — the two solve different layers (static vs runtime). |
| `"<FakeTokenizer>" is not assignable to parameter of type "<LibraryClass>"` (passing a test double) | The function is typed against a concrete library class but actually uses only one or two of its methods, so a fake can't be passed. | Type the parameter against a minimal `typing.Protocol` describing **just the methods the function calls** — not the whole library class. Both the real object and the fake satisfy it structurally (no inheritance, no `cast`). Make protocol method params positional-only (`/`) where the impl's param names shouldn't have to match. This also keeps the test double's concrete type, so its helper attributes stay type-valid. |

General rules:
- **Narrow, don't suppress.** Use `assert isinstance(...)` or `typing.cast`, not `# type: ignore`. A `cast` is a zero-runtime hint; centralize it in a small helper if used more than once.
- **Annotate with the instance type, not the factory.** `PreTrainedTokenizerBase`, `Dataset`, etc.
- **Verify after the fix**: re-run the script (or `--help`) to confirm behavior is unchanged — type fixes must never alter runtime.

How to find them: read the file in the IDE/Pylance output, or scan with the
project's type checker if configured. Address every warning in a touched file,
not just the one reported.

## 2. CLI defaults: always a default config AND default paths

Every script under `scripts/` must run with **zero required arguments**:
- `--config` defaults to the canonical config, resolved relative to the file so
  cwd doesn't matter:
  `DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "configs" / "<name>.yaml")`
- Input/output `--input` / `--output` (or equivalent) have sensible `DEFAULT_*`
  constants at module top.
- Keep hyperparameters in the YAML (per `claude.md`) — defaults are for *paths
  and the config location*, not for values that belong in the config.

## 3. Unit tests are required

Any change to logic in a `scripts/*.py` file must be covered by unit tests
before it is "done". Tests live in `tests/test_<module>.py` and are written as
`unittest.TestCase` classes — keep them framework-neutral so they run under
either runner. `pytest` is installed in the `llm` env; stdlib `unittest` is the
fallback that always works.

Run them (either):
- `python -m pytest tests/ -q`
- `python -m unittest discover -s tests`

Requirements for the tests:
- **No network, no model downloads.** Do not load real HF tokenizers/models in
  unit tests. Use a **fake tokenizer** that reproduces only the structural
  property the code relies on — for `apply_chat_template` that's
  *prefix-consistency* (encoding `messages[:k]` with a generation prompt is a
  prefix of encoding `messages[:k+1]`). See `tests/test_prepare_dataset.py`.
- **Make the production code accept the fake by design.** Type functions against
  a minimal `typing.Protocol` of the capability they use (e.g.
  `SupportsChatTemplate`), so the fake satisfies the signature structurally — no
  `cast`, no subclassing the heavy library class.
- **Cover the logic, not the library.** Test our masking, truncation, shape, and
  config-parsing behavior — assert that only assistant tokens are supervised,
  that truncation keeps the tail, that shapes stay aligned, etc.
- **Test the edges**: single- vs multi-round, under/over `max_length`, empty or
  degenerate inputs.
- **Imports**: tests reach modules under `scripts/` via a runtime
  `sys.path.insert(...)`. For the type checker to resolve them too, `scripts` is
  listed in `python.analysis.extraPaths` (`.vscode/settings.json`). If you add a
  new source dir tests import from, add it there as well.
- New logic without a corresponding test is not done. If a bug is fixed, add a
  test that would have caught it.

## 4. Propagate after any relevant edit

When an edit changes behavior, interface, paths, or dependencies, update the
docs in the same pass — do not leave them stale:

- **devlog.md** — append (never overwrite) a dated entry: `Built / Changed`,
  `Key Decisions`, `Next Steps`.
- **README.md** — update any section describing the changed command, its I/O
  paths, flags, or output format.
- **requirements.txt** — if (and only if) a new third-party import was added.
  Stdlib additions (`pathlib`, `typing`, …) need no change — say so explicitly
  rather than silently skipping.

## Done-checklist
- [ ] No Pylance warnings remain in the touched file
- [ ] Script runs with no args (default config + default paths)
- [ ] No hardcoded values that belong in the YAML config
- [ ] Unit tests added/updated for changed logic, and the suite passes (`python -m pytest tests/ -q` or `python -m unittest discover -s tests`)
- [ ] devlog.md entry appended
- [ ] README.md updated if the interface/paths/output changed
- [ ] requirements.txt updated if a new dependency was added (else noted as N/A)
- [ ] Re-ran the script to confirm runtime behavior is unchanged
