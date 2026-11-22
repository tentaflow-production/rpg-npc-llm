"""Unit tests for scripts/train_qwen_sft.py — pure logic only: offline, no model, no GPU."""

import subprocess
import sys
from pathlib import Path

import pytest
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import train_qwen_sft as tr


def make_cfg(**train_overrides) -> dict:
    cfg = {
        "exp_name": "exp_x",
        "seed": 7,
        "data_seed": 9,
        "model": {
            "base": "/models/q", "dtype": "bf16", "attn_implementation": "sdpa",
            "gradient_checkpointing": True,
        },
        "data": {
            "tokenized_dir": "/d", "val_fraction": 0.1, "max_length": 1024, "dry_run_examples": 32,
        },
        "lora": {
            "r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj"], "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "train": {
            "epochs": 2, "per_device_batch": 1, "grad_accum": 16, "lr": "2e-4", "scheduler": "cosine",
            "warmup_ratio": 0.03, "optim": "adamw_torch", "weight_decay": 0.0, "max_grad_norm": 1.0,
            "logging_steps": 10, "eval_strategy": "epoch", "save_strategy": "epoch",
            "save_total_limit": 3, "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
            "resume_from_checkpoint": "auto", "report_to": "none", "output_dir": "/ckpt/exp_x",
        },
    }
    cfg["train"].update(train_overrides)
    return cfg


def tiny_ds(n: int = 20) -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [[i, 2, 3] for i in range(n)],
            "attention_mask": [[1, 1, 1]] * n,
            "labels": [[-100, 2, 3]] * n,
        }
    )


# --- config --------------------------------------------------------------


def test_expand_env_substitutes_nested():
    out = tr.expand_env({"a": ["${X}/m", 3], "b": {"c": "${X}"}}, {"X": "/r"})
    assert out == {"a": ["/r/m", 3], "b": {"c": "/r"}}


def test_expand_env_missing_var_raises():
    with pytest.raises(KeyError):
        tr.expand_env("${NOPE_NOT_SET}", {})


def test_validate_accepts_full_config():
    tr.validate_config(make_cfg())


def test_validate_rejects_missing_value():
    cfg = make_cfg()
    del cfg["train"]["weight_decay"]
    with pytest.raises(ValueError, match="weight_decay"):
        tr.validate_config(cfg)


def test_validate_rejects_unequal_strategies_with_load_best():
    with pytest.raises(ValueError, match="strategy"):
        tr.validate_config(make_cfg(eval_strategy="steps"))


def test_validate_rejects_non_bf16():
    cfg = make_cfg()
    cfg["model"]["dtype"] = "fp16"
    with pytest.raises(ValueError, match="bf16"):
        tr.validate_config(cfg)


# --- kwargs mapping ------------------------------------------------------


def test_kwargs_core_mapping():
    kw = tr.build_training_kwargs(make_cfg(), "/out", "/log")
    assert kw["output_dir"] == "/out" and kw["logging_dir"] == "/log"
    assert kw["per_device_train_batch_size"] == 1 and kw["gradient_accumulation_steps"] == 16
    assert kw["learning_rate"] == pytest.approx(2e-4) and isinstance(kw["learning_rate"], float)
    assert kw["eval_strategy"] == kw["save_strategy"] == "epoch"
    assert kw["load_best_model_at_end"] is True and kw["greater_is_better"] is False
    assert kw["seed"] == 7 and kw["data_seed"] == 9
    assert kw["bf16"] is True and kw["report_to"] == "none"
    assert kw["remove_unused_columns"] is False
    assert kw["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert kw["max_steps"] == -1


def test_kwargs_dry_run_uses_equal_step_strategies():
    kw = tr.build_training_kwargs(make_cfg(), "/o", "/l", dry_run=True, max_steps=5)
    assert kw["eval_strategy"] == kw["save_strategy"] == "steps"
    assert kw["eval_steps"] == kw["save_steps"] == 5
    assert kw["max_steps"] == 5


def test_kwargs_accepted_by_training_arguments(tmp_path):
    from transformers import TrainingArguments

    kw = tr.build_training_kwargs(make_cfg(), str(tmp_path), str(tmp_path / "l"))
    TrainingArguments(**kw)  # raises on an invalid combination (e.g. unequal strategies)


# --- data ----------------------------------------------------------------


def test_split_train_val_disjoint_deterministic_and_nonempty():
    ds = tiny_ds(20)
    t1, v1 = tr.split_train_val(ds, 0.1, 5)
    t2, v2 = tr.split_train_val(ds, 0.1, 5)
    assert len(v1) == 2 and len(t1) == 18
    assert tr.dataset_hash(v1) == tr.dataset_hash(v2) and tr.dataset_hash(t1) == tr.dataset_hash(t2)
    assert not set(map(tuple, t1["input_ids"])) & set(map(tuple, v1["input_ids"]))
    _, v_tiny = tr.split_train_val(tiny_ds(5), 0.01, 5)
    assert len(v_tiny) == 1


def test_split_seed_changes_split():
    ds = tiny_ds(40)
    assert tr.dataset_hash(tr.split_train_val(ds, 0.2, 1)[1]) != tr.dataset_hash(
        tr.split_train_val(ds, 0.2, 2)[1]
    )


def test_assert_dataset_columns():
    tr.assert_dataset_columns(tiny_ds(2))
    with pytest.raises(ValueError):
        tr.assert_dataset_columns(Dataset.from_dict({"input_ids": [[1]]}))


def test_assert_max_length():
    tr.assert_max_length(tiny_ds(2), 3)
    with pytest.raises(ValueError):
        tr.assert_max_length(tiny_ds(2), 2)


def test_take_first_n():
    assert len(tr.take_first_n(tiny_ds(5), 2)) == 2
    assert len(tr.take_first_n(tiny_ds(5), 99)) == 5
    assert len(tr.take_first_n(tiny_ds(5), None)) == 5


def test_resolve_tokenized_dir_picks_newest(tmp_path):
    for n in ("dialogues_built_20260101_000000", "dialogues_built_20260301_000000"):
        (tmp_path / n).mkdir()
    assert tr.resolve_tokenized_dir(str(tmp_path)).endswith("20260301_000000")
    direct = tmp_path / "dialogues_built_20260101_000000"
    (direct / "dataset_info.json").write_text("{}")
    assert tr.resolve_tokenized_dir(str(direct)) == str(direct)
    with pytest.raises(FileNotFoundError):
        tr.resolve_tokenized_dir(str(tmp_path / "missing"))


# --- results.json fields -------------------------------------------------


def test_train_loss_final_is_mean_of_last_ten_logged_losses():
    hist = [{"loss": float(i), "step": i} for i in range(1, 16)] + [{"eval_loss": 99.0}]
    assert tr.train_loss_final(hist) == pytest.approx(sum(range(6, 16)) / 10)
    assert tr.train_loss_final([{"loss": 2.0}, {"loss": 4.0}]) == pytest.approx(3.0)
    assert tr.train_loss_final([{"eval_loss": 1.0}]) is None


def test_weights_hash_changes_with_content(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"one")
    h1 = tr.weights_sha256(tmp_path)
    (tmp_path / "a.safetensors").write_bytes(b"two")
    assert tr.weights_sha256(tmp_path) != h1
    with pytest.raises(FileNotFoundError):
        tr.weights_sha256(tmp_path / "empty_nowhere")


def test_eval_hash_null_only_in_dry_run(tmp_path):
    cfg = make_cfg()
    cfg["data"]["eval"] = str(tmp_path / "nope.jsonl")
    assert tr.eval_set_hash(cfg, dry_run=True) is None
    with pytest.raises(FileNotFoundError):
        tr.eval_set_hash(cfg, dry_run=False)
    (tmp_path / "nope.jsonl").write_text("{}\n")
    assert tr.eval_set_hash(cfg, dry_run=False) == tr.eval_set_hash(cfg, dry_run=True)


def test_metrics_csv_unions_keys(tmp_path):
    p = tmp_path / "m.csv"
    tr.write_metrics_csv(p, [{"step": 1, "loss": 2.0}, {"step": 2, "eval_loss": 1.0}])
    lines = p.read_text().splitlines()
    assert lines[0] == "step,loss,eval_loss" and len(lines) == 3


# --- attempts, resume, sha refusal --------------------------------------


def test_begin_attempt_creates_then_compares_with_previous(tmp_path):
    cur, prev = tr.begin_attempt(tmp_path, "aaa")
    assert cur.name == "attempt-train-1" and prev is None
    cur2, prev2 = tr.begin_attempt(tmp_path, "bbb")
    assert cur2.name == "attempt-train-2" and prev2 == "aaa"


def test_begin_attempt_adopts_dispatcher_created_dir(tmp_path):
    first = tmp_path / "attempt-train-1"
    first.mkdir()
    (first / "git_sha.txt").write_text("aaa\n")
    tr.begin_attempt(tmp_path, "aaa")  # adopts attempt-train-1 and marks it started
    disp = tmp_path / "attempt-train-2"  # dispatcher pre-creates the retry
    disp.mkdir()
    (disp / "git_sha.txt").write_text("bbb\n")
    cur, prev = tr.begin_attempt(tmp_path, "bbb")
    assert cur == disp and prev == "aaa"
    assert not (tmp_path / "attempt-train-3").exists()


def test_check_resume_sha():
    tr.check_resume_sha("abc", "abc", False)
    tr.check_resume_sha("abc", "def", True)
    with pytest.raises(SystemExit, match="refusing to resume"):
        tr.check_resume_sha("abc", "def", False)
    with pytest.raises(SystemExit, match="unknown"):
        tr.check_resume_sha(None, "def", False)


# --- dirty tree ----------------------------------------------------------


def test_tree_dirty(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "f").write_text("1")
    git("add", "f")
    git("commit", "-qm", "init")
    assert tr.tree_dirty(tmp_path) is False
    (tmp_path / "f").write_text("2")
    assert tr.tree_dirty(tmp_path) is True
    assert len(tr.git_sha(tmp_path)) == 40


def test_main_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "tree_dirty", lambda repo: True)
    monkeypatch.setattr(tr, "git_sha", lambda repo: "x" * 40)
    cfgp = tmp_path / "c.yaml"
    import yaml

    cfgp.write_text(yaml.safe_dump(make_cfg()))
    with pytest.raises(SystemExit, match="dirty"):
        tr.main(["--config", str(cfgp), "--job-dir", str(tmp_path / "job"), "--dry-run"])
    assert not (tmp_path / "job" / "results.json").exists()
