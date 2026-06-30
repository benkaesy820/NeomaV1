from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from build_stage_a_250k_slice import validate_plan
from prepare_stage_a_250k_dataset import split_families
from run_stage_a_250k_probe import comparison, load_probe_rows, summarize_smoke_baseline
from tinyllm.model import TinyConfig, TinyLanguageModel, count_parameters
from train import peak_rss_bytes


class StageA250KProbeTest(unittest.TestCase):
    def test_plan_and_config_are_bounded(self) -> None:
        plan = json.loads(
            (ROOT / "data/foundation/manifests/stage_a_250k_probe_v0_1_plan.json").read_text(
                encoding="utf-8"
            )
        )
        validate_plan(plan)
        self.assertEqual(plan["baseline"], "321f0f2")
        self.assertNotIn("frozen_stage_b", {row["group_id"] for row in plan["groups"]})
        config = json.loads(
            (ROOT / "configs/stage_a_250k_probe_8k_cpu.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["vocab_size"], 8000)
        self.assertEqual(config["max_steps"], 500)
        self.assertEqual(
            config["max_steps"] * config["batch_size"] * config["seq_len"] * config["grad_accum_steps"],
            256000,
        )
        self.assertIsNone(config["train_loss_mask"])
        self.assertIsNone(config["val_loss_mask"])
        model = TinyLanguageModel(TinyConfig.from_dict(config))
        self.assertEqual(count_parameters(model), 3_307_200)

    def test_family_split_is_deterministic_and_disjoint(self) -> None:
        rows = []
        counts = {}
        for source in ("python", "docs", "english"):
            for index in range(8):
                record_id = f"{source}-{index}"
                rows.append({
                    "record_id": record_id,
                    "source_id": source,
                    "family_id": f"{source}:family:{index}",
                })
                counts[record_id] = 1000 + index
        train_a, val_a, _ = split_families(rows, counts, 0.10, 1616)
        train_b, val_b, _ = split_families(rows, counts, 0.10, 1616)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertFalse(train_a & val_a)
        self.assertTrue(train_a)
        self.assertTrue(val_a)

    def test_probe_prompts_are_excluded_and_balanced(self) -> None:
        rows = load_probe_rows(ROOT / "data/eval/stage_a_250k_probe_prompts_v0_1.jsonl")
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["category"] for row in rows}, {"english", "code"})
        self.assertTrue(all(row["training_allowed"] is False for row in rows))

    def test_peak_rss_is_recordable(self) -> None:
        value = peak_rss_bytes()
        if os.name == "nt":
            self.assertIsInstance(value, int)
        self.assertTrue(value is None or value > 0)

    def test_comparison_keeps_unavailable_baseline_memory_null(self) -> None:
        current = {
            "training_tokens_seen": 256000,
            "effective_tokens_per_second": 1000.0,
            "peak_rss_bytes": 500000000,
            "checkpoint_bytes": 40000000,
        }
        baseline = {
            "training_tokens_seen": 51200,
            "effective_tokens_per_second": 900.0,
            "peak_rss_bytes": None,
            "checkpoint_bytes": 39000000,
        }
        result = comparison(current, baseline)
        self.assertEqual(result["ratios"]["training_tokens_seen"], 5.0)
        self.assertIsNone(result["ratios"]["peak_rss_bytes"])

    def test_smoke_summary_derives_speed_without_inventing_ram(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            report = {
                "status": "passed",
                "phase1_completed_step": 30,
                "completed_step": 100,
                "training_tokens_seen": 51200,
                "initial_train_loss": 9.0,
                "final_train_loss": 7.0,
                "initial_val_loss": 8.9,
                "final_val_loss": 7.2,
                "latest_checkpoint": str(checkpoint),
                "generation_verified": True,
                "special_tokens_verified": True,
                "evaluation_points": [
                    {"step": 0, "elapsed_seconds": 0.0},
                    {"step": 30, "elapsed_seconds": 10.0},
                    {"step": 100, "elapsed_seconds": 20.0},
                ],
            }
            report_path = root / "smoke_probe_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            summary = summarize_smoke_baseline(report_path)
            self.assertAlmostEqual(summary["effective_tokens_per_second"], 51200 / 30.0)
            self.assertIsNone(summary["peak_rss_bytes"])
            self.assertEqual(summary["checkpoint_bytes"], len(b"checkpoint"))


if __name__ == "__main__":
    unittest.main()
