from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_dataset import Record, encode_record, records_from_file, split_records


class PrepareDatasetTest(unittest.TestCase):
    def tokenizer(self) -> Tokenizer:
        vocab = {
            "<unk>": 0,
            "<eos>": 1,
            "<answer>": 2,
            "</answer>": 3,
            "<instruction>": 4,
            "</instruction>": 5,
            "hello": 6,
            "world": 7,
        }
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        return tokenizer

    def test_instruction_examples_are_kept_as_complete_records(self) -> None:
        text = (
            "<instruction> one </instruction> <answer> hello </answer>\n"
            "<instruction> two </instruction> <answer> world </answer>\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "examples.txt"
            path.write_text(text, encoding="utf-8")
            records = records_from_file(path, root)

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.is_instruction for record in records))
        self.assertTrue(all("<answer>" in record.text for record in records))

    def test_answer_only_mask(self) -> None:
        tokenizer = self.tokenizer()
        record = Record(
            source="x.txt",
            index=0,
            text="<instruction> hello </instruction> <answer> world </answer>",
            is_instruction=True,
        )
        ids, mask = encode_record(tokenizer, record, eos_id=1, instruction_loss_mask=True)
        answer_open = list(ids).index(2)
        answer_close = list(ids).index(3)

        self.assertTrue(np.all(mask[: answer_open + 1] == 0))
        self.assertTrue(np.all(mask[answer_open + 1 : answer_close + 1] == 1))
        self.assertEqual(mask[-1], 1)

    def test_record_split_is_deterministic_and_disjoint(self) -> None:
        records = [Record("x.txt", index, str(index), False) for index in range(20)]
        train_a, val_a = split_records(records, 0.2, seed=1337)
        train_b, val_b = split_records(records, 0.2, seed=1337)

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertFalse(set(train_a) & set(val_a))
        self.assertEqual(len(val_a), 4)


if __name__ == "__main__":
    unittest.main()
