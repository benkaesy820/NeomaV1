from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_stage_a_250k_extended_probe import MILESTONES, repeated_token_diagnostics
from tinyllm.model import TinyConfig, TinyLanguageModel, count_parameters


class StageA250KExtendedProbeTest(unittest.TestCase):
    def test_extended_config_is_bounded_to_diagnostic_scale(self) -> None:
        config = json.loads((ROOT / "configs/stage_a_250k_extended_8k_cpu.json").read_text(encoding="utf-8"))
        self.assertEqual(config["run_name"], "stage_a_250k_extended_8k")
        self.assertEqual(config["out_dir"], "runs/stage_a_250k_extended_8k")
        self.assertEqual(config["vocab_size"], 8000)
        self.assertEqual(config["max_steps"], 2000)
        self.assertEqual(config["eval_interval"], 100)
        self.assertEqual(config["save_interval"], 500)
        self.assertEqual(
            config["max_steps"] * config["batch_size"] * config["seq_len"] * config["grad_accum_steps"],
            1_024_000,
        )
        self.assertIsNone(config["train_loss_mask"])
        self.assertIsNone(config["val_loss_mask"])
        model = TinyLanguageModel(TinyConfig.from_dict(config))
        self.assertEqual(count_parameters(model), 3_307_200)

    def test_milestones_cover_500_to_2000(self) -> None:
        self.assertEqual(MILESTONES, (500, 1000, 1500, 2000))

    def test_repetition_diagnostics_detect_repeated_tokens(self) -> None:
        result = repeated_token_diagnostics([
            {"id": "a", "text": "alpha beta gamma"},
            {"id": "b", "text": "x x x y"},
        ])
        self.assertEqual(result["max_repeated_token_run"], 3)
        self.assertLess(result["lowest_unique_token_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
