import unittest

import torch

from tinyllm.model import TinyConfig, TinyLanguageModel, count_parameters


class ModelTest(unittest.TestCase):
    def test_forward_loss_and_generate(self) -> None:
        config = TinyConfig(
            vocab_size=128,
            seq_len=16,
            n_layers=2,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
        )
        model = TinyLanguageModel(config)
        x = torch.randint(0, config.vocab_size, (2, config.seq_len))
        logits, loss = model(x, x)

        self.assertEqual(logits.shape, (2, config.seq_len, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(count_parameters(model), 0)

        generated = model.generate(x[:, :4], max_new_tokens=2)
        self.assertEqual(generated.shape, (2, 6))


if __name__ == "__main__":
    unittest.main()
