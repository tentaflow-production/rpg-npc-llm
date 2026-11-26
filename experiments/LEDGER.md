# Experiment ledger

One row per experiment, PASS and FAIL alike (a FAIL is a finding). Every leak number in this file means
**matcher-detected leak** (`data/secrets.yaml`); recall is unknown until M4. `in_character` is `unjudged` until
the judge is calibrated (M4). Rows are appended after `eval_gate.py --record-only`; the first row is `exp_base_prompted`.

| exp | date | baseline | knob | target | train_loss | eval_loss | adversarial leaks | ordinary leaks | in_character | IMPROVED | RELEASABLE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
