"""Unit tests for scripts/train_qwen_sft.py — pure logic only, offline, no model.

Covers the config→TrainingArguments mapping, experiment-dir naming, the
take-first-N subset helper, and the dataset-column guard. Where a real Dataset
is needed, a tiny in-memory datasets.Dataset is built with from_dict — that is
fully offline and needs no download.

Run: python -m pytest tests/ -q
"""

import sys
import unittest
from pathlib import Path

from datasets import Dataset

# Make scripts/ importable without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import train_qwen_sft as tr  # noqa: E402


def make_cfg(**training_overrides) -> dict:
    cfg = {
        "model": {"base_model": "Qwen/Qwen2.5-7B-Instruct"},
        "run_name": "run-x",
        "experiment_dir": "/exp",
        "paths": {
            "data_cache": "/data/tok",
            "output_dir": "/ckpt/run-x",
            "log_dir": "/logs/run-x",
        },
        "training": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "num_train_epochs": 2,
            "learning_rate": "2e-4",  # YAML often hands this through as a string
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "save_strategy": "steps",
            "save_steps": 200,
            "save_total_limit": 3,
            "bf16": True,
            "logging_steps": 20,
        },
        "seed": 42,
    }
    cfg["training"].update(training_overrides)
    return cfg


def tiny_tokenized_dataset(n: int = 5) -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3] for _ in range(n)],
            "attention_mask": [[1, 1, 1] for _ in range(n)],
            "labels": [[-100, 2, 3] for _ in range(n)],
        }
    )


class TestBuildTrainingKwargs(unittest.TestCase):
    def test_core_mapping(self):
        kw = tr.build_training_kwargs(make_cfg())
        self.assertEqual(kw["output_dir"], "/ckpt/run-x")
        self.assertEqual(kw["logging_dir"], "/logs/run-x")
        self.assertEqual(kw["per_device_train_batch_size"], 1)
        self.assertEqual(kw["gradient_accumulation_steps"], 16)
        self.assertEqual(kw["num_train_epochs"], 2)
        self.assertEqual(kw["lr_scheduler_type"], "cosine")
        self.assertEqual(kw["save_strategy"], "steps")
        self.assertTrue(kw["bf16"])
        self.assertEqual(kw["run_name"], "run-x")

    def test_learning_rate_coerced_to_float(self):
        # YAML may parse "2e-4" as a string; the mapping must hand a float to TA.
        kw = tr.build_training_kwargs(make_cfg(learning_rate="2e-4"))
        self.assertIsInstance(kw["learning_rate"], float)
        self.assertAlmostEqual(kw["learning_rate"], 2e-4)

    def test_max_steps_defaults_to_minus_one(self):
        self.assertEqual(tr.build_training_kwargs(make_cfg())["max_steps"], -1)

    def test_max_steps_override(self):
        self.assertEqual(tr.build_training_kwargs(make_cfg(max_steps=60))["max_steps"], 60)

    def test_grad_ckpt_and_remove_unused_columns(self):
        kw = tr.build_training_kwargs(make_cfg())
        self.assertTrue(kw["gradient_checkpointing"])
        self.assertEqual(kw["gradient_checkpointing_kwargs"], {"use_reentrant": False})
        self.assertFalse(kw["remove_unused_columns"])  # pre-tokenized cols must survive

    def test_report_to_default_and_override(self):
        self.assertEqual(tr.build_training_kwargs(make_cfg())["report_to"], "none")
        self.assertEqual(
            tr.build_training_kwargs(make_cfg(report_to="wandb"))["report_to"], "wandb"
        )


class TestExperimentSnapshotDir(unittest.TestCase):
    def test_naming(self):
        out = tr.experiment_snapshot_dir("/exp", "run-x", "2026_06_14")
        self.assertEqual(out, Path("/exp/run-x_2026_06_14"))


class TestTakeFirstN(unittest.TestCase):
    def test_positive_n_takes_first_n(self):
        out = tr.take_first_n(tiny_tokenized_dataset(5), 2)
        self.assertEqual(len(out), 2)

    def test_n_larger_than_dataset_is_clamped(self):
        out = tr.take_first_n(tiny_tokenized_dataset(3), 99)
        self.assertEqual(len(out), 3)

    def test_none_returns_full(self):
        self.assertEqual(len(tr.take_first_n(tiny_tokenized_dataset(4), None)), 4)

    def test_zero_or_negative_returns_full(self):
        ds = tiny_tokenized_dataset(4)
        self.assertEqual(len(tr.take_first_n(ds, 0)), 4)
        self.assertEqual(len(tr.take_first_n(ds, -1)), 4)


class TestAssertDatasetColumns(unittest.TestCase):
    def test_exact_columns_pass(self):
        tr.assert_dataset_columns(tiny_tokenized_dataset(2))  # must not raise

    def test_extra_column_raises(self):
        ds = Dataset.from_dict(
            {
                "input_ids": [[1]],
                "attention_mask": [[1]],
                "labels": [[1]],
                "text": ["x"],
            }
        )
        with self.assertRaises(ValueError):
            tr.assert_dataset_columns(ds)

    def test_missing_column_raises(self):
        ds = Dataset.from_dict({"input_ids": [[1]], "attention_mask": [[1]]})
        with self.assertRaises(ValueError):
            tr.assert_dataset_columns(ds)


if __name__ == "__main__":
    unittest.main()
