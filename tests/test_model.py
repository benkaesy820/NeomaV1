from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyllm.model import TinyConfig, TinyLanguageModel, count_parameters


class ModelTest(unittest.TestCase):
    def make_model(self) -> TinyLanguageModel:
        torch.manual_seed(7)
        return TinyLanguageModel(
            TinyConfig(
                vocab_size=128,
                seq_len=16,
                n_layers=2,
                d_model=64,
                n_heads=4,
                n_kv_heads=2,
                d_ff=128,
            )
        )

    def test_forward_loss_and_generate(self) -> None:
        model = self.make_model()
        config = model.config
        x = torch.randint(0, config.vocab_size, (2, config.seq_len))
        logits, loss = model(x, x)

        self.assertEqual(logits.shape, (2, config.seq_len, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(count_parameters(model), 0)

        model.train()
        generated = model.generate(x[:, :4], max_new_tokens=2, temperature=0)
        self.assertEqual(generated.shape, (2, 6))
        self.assertTrue(model.training, "generate should restore the previous training mode")

    def test_attention_is_causal(self) -> None:
        model = self.make_model().eval()
        prefix = torch.tensor([[1, 2, 3, 4, 5]])
        left = torch.cat((prefix, torch.tensor([[6, 7, 8]])), dim=1)
        right = torch.cat((prefix, torch.tensor([[50, 51, 52]])), dim=1)

        left_logits, _ = model(left)
        right_logits, _ = model(right)
        self.assertTrue(torch.allclose(left_logits[:, :5], right_logits[:, :5], atol=1e-6))

    def test_loss_mask_matches_manual_cross_entropy(self) -> None:
        model = self.make_model().eval()
        x = torch.randint(0, model.config.vocab_size, (1, 8))
        targets = torch.randint(0, model.config.vocab_size, (1, 8))
        mask = torch.zeros_like(targets, dtype=torch.float32)
        mask[:, 3:6] = 1

        logits, masked_loss = model(x, targets, mask)
        self.assertIsNotNone(masked_loss)
        manual = F.cross_entropy(logits[:, 3:6].reshape(-1, logits.size(-1)), targets[:, 3:6].reshape(-1))
        self.assertTrue(torch.allclose(masked_loss, manual, atol=1e-6))

    def test_rope_cache_is_not_saved_in_checkpoint(self) -> None:
        model = self.make_model()
        state = model.state_dict()
        self.assertNotIn("rope_cos", state)
        self.assertNotIn("rope_sin", state)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TinyLanguageModel(TinyConfig(d_model=63, n_heads=4))


if __name__ == "__main__":
    unittest.main()
