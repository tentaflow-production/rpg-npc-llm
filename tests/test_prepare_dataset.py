"""Unit tests for scripts/prepare_dataset.py.

These use a fake tokenizer that reproduces the *structural* contract we rely on
from Qwen's chat template — prefix-consistency: encoding messages[:k] with a
generation prompt is a prefix of encoding messages[:k+1]. That lets us test the
label-masking and left-truncation logic deterministically and offline, with no
model download.

Run:  python -m unittest discover -s tests
"""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

# Make scripts/ importable without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import prepare_dataset as pd  # noqa: E402


def _sid(s: str) -> int:
    """Deterministic, process-stable token id for a string."""
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "big")


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's apply_chat_template.

    Renders each message as [IM_START, role_id, *word_ids, IM_END], optionally
    appending the assistant generation header. This is prefix-consistent the
    same way Qwen's template is, which is the only property tokenize_example
    depends on.
    """

    IM_START = 1_000_000
    IM_END = 1_000_001

    def _role(self, role: str) -> int:
        return _sid("role:" + role)

    def _word(self, w: str) -> int:
        return _sid("word:" + w)

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, **_kwargs
    ):
        assert tokenize is True, "tests only exercise the tokenize=True path"
        ids: list[int] = []
        for m in messages:
            ids.append(self.IM_START)
            ids.append(self._role(m["role"]))
            ids.extend(self._word(w) for w in m["content"].split())
            ids.append(self.IM_END)
        if add_generation_prompt:
            ids.append(self.IM_START)
            ids.append(self._role("assistant"))
        return ids


SINGLE = [
    {"role": "system", "content": "you are a gruff blacksmith"},
    {"role": "user", "content": "can you make a sword"},
    {"role": "assistant", "content": "depends got coin"},
]

MULTI = [
    {"role": "system", "content": "you are a gruff blacksmith"},
    {"role": "user", "content": "hello there"},
    {"role": "assistant", "content": "what do you want"},
    {"role": "user", "content": "a sword please"},
    {"role": "assistant", "content": "fifty gold up front"},
]


class TestEncodeChat(unittest.TestCase):
    def test_returns_list_of_ints(self):
        tok = FakeTokenizer()
        out = pd.encode_chat(tok, SINGLE, add_generation_prompt=False)
        self.assertIsInstance(out, list)
        self.assertTrue(all(isinstance(x, int) for x in out))

    def test_generation_prompt_is_prefix(self):
        # The property tokenize_example relies on.
        tok = FakeTokenizer()
        prefix = pd.encode_chat(tok, SINGLE[:2], add_generation_prompt=True)
        upto = pd.encode_chat(tok, SINGLE[:3], add_generation_prompt=False)
        self.assertEqual(upto[: len(prefix)], prefix)


class TestTokenizeExampleMasking(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def _supervised_ids(self, out):
        return [
            tid
            for tid, lab in zip(out["input_ids"], out["labels"])
            if lab != pd.IGNORE_INDEX
        ]

    def test_shapes_align(self):
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=1000)
        n = len(out["input_ids"])
        self.assertEqual(len(out["labels"]), n)
        self.assertEqual(len(out["attention_mask"]), n)

    def test_attention_mask_all_ones(self):
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=1000)
        self.assertTrue(all(m == 1 for m in out["attention_mask"]))

    def test_single_round_supervises_only_assistant_reply(self):
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=1000)
        expected = [self.tok._word(w) for w in "depends got coin".split()]
        expected.append(FakeTokenizer.IM_END)  # answer content + closing token
        self.assertEqual(self._supervised_ids(out), expected)

    def test_assistant_header_is_masked(self):
        # The [IM_START, role:assistant] header must NOT be supervised.
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=1000)
        sup = self._supervised_ids(out)
        self.assertNotIn(self.tok._role("assistant"), sup)
        self.assertNotIn(FakeTokenizer.IM_START, sup)

    def test_system_and_user_fully_masked(self):
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=1000)
        sup = self._supervised_ids(out)
        for w in "you are a gruff blacksmith can make sword".split():
            self.assertNotIn(self.tok._word(w), sup)

    def test_labels_match_input_ids_where_supervised(self):
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=1000)
        for tid, lab in zip(out["input_ids"], out["labels"]):
            if lab != pd.IGNORE_INDEX:
                self.assertEqual(lab, tid)

    def test_multi_round_supervises_both_assistant_turns(self):
        out = pd.tokenize_example({"messages": MULTI}, self.tok, max_length=1000)
        sup = self._supervised_ids(out)
        expected = (
            [self.tok._word(w) for w in "what do you want".split()]
            + [FakeTokenizer.IM_END]
            + [self.tok._word(w) for w in "fifty gold up front".split()]
            + [FakeTokenizer.IM_END]
        )
        self.assertEqual(sup, expected)


class TestTokenizeExampleTruncation(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_no_truncation_when_under_limit(self):
        full = self.tok.apply_chat_template(SINGLE, add_generation_prompt=False)
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=10_000)
        self.assertEqual(out["input_ids"], full)

    def test_left_truncation_keeps_the_tail(self):
        full = self.tok.apply_chat_template(SINGLE, add_generation_prompt=False)
        max_len = len(full) - 3
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=max_len)
        self.assertEqual(len(out["input_ids"]), max_len)
        self.assertEqual(out["input_ids"], full[-max_len:])  # kept the tail, not head
        self.assertEqual(out["labels"], out["labels"][-max_len:])  # parallel slice

    def test_left_truncation_preserves_answer_end(self):
        # max_length big enough to hold the answer but not the whole prompt:
        # the closing IM_END (and thus the supervised tail) must survive.
        full = self.tok.apply_chat_template(SINGLE, add_generation_prompt=False)
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=len(full) - 2)
        self.assertEqual(out["input_ids"][-1], FakeTokenizer.IM_END)
        self.assertTrue(any(lab != pd.IGNORE_INDEX for lab in out["labels"]))

    def test_truncation_keeps_shapes_aligned(self):
        full = self.tok.apply_chat_template(SINGLE, add_generation_prompt=False)
        max_len = len(full) - 4
        out = pd.tokenize_example({"messages": SINGLE}, self.tok, max_length=max_len)
        self.assertEqual(len(out["input_ids"]), max_len)
        self.assertEqual(len(out["labels"]), max_len)
        self.assertEqual(len(out["attention_mask"]), max_len)
        self.assertTrue(all(m == 1 for m in out["attention_mask"]))


class TestTokenizedDir(unittest.TestCase):
    def test_name_contains_timestamp_under_parent(self):
        out = pd.tokenized_dir("/data/tokenized", "20260614_120000")
        self.assertEqual(
            out, Path("/data/tokenized/dialogues_built_20260614_120000")
        )

    def test_parent_is_preserved(self):
        out = pd.tokenized_dir(Path("/a/b"), "20260101_000000")
        self.assertEqual(out.parent, Path("/a/b"))
        self.assertIn("20260101_000000", out.name)


class TestLoadConfig(unittest.TestCase):
    def test_round_trips_yaml(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False
        ) as f:
            f.write("model:\n  base_model: foo\ntraining:\n  max_length: 128\n")
            path = f.name
        cfg = pd.load_config(path)
        self.assertEqual(cfg["model"]["base_model"], "foo")
        self.assertEqual(cfg["training"]["max_length"], 128)
        Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
