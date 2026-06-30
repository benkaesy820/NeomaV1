from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from build_stage_a_smoke_slice import validate_plan
from prepare_stage_a_smoke_dataset import split_families
from stage_a_staging_common import canonical_json_bytes, sha256_file
from tinyllm.model import TinyConfig, TinyLanguageModel, count_parameters
from train import get_batch, validate_resume_signature


class StageASmokeProbeTest(unittest.TestCase):
    def test_tracked_plan_and_config_are_bounded(self) -> None:
        plan = json.loads(
            (ROOT / "data/foundation/manifests/stage_a_smoke_probe_v0_1_plan.json").read_text(
                encoding="utf-8"
            )
        )
        validate_plan(plan)
        config = json.loads(
            (ROOT / "configs/stage_a_smoke_probe_8k_cpu.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["vocab_size"], 8000)
        self.assertEqual(config["max_steps"] * config["batch_size"] * config["seq_len"] * config["grad_accum_steps"], 51200)
        self.assertIsNone(config["train_loss_mask"])
        self.assertIsNone(config["val_loss_mask"])
        model = TinyLanguageModel(TinyConfig.from_dict(config))
        self.assertEqual(count_parameters(model), 3_307_200)

    def test_family_split_is_deterministic_and_disjoint(self) -> None:
        rows = []
        counts = {}
        for source in ("a", "b", "c"):
            for family_index in range(4):
                record_id = f"{source}-{family_index}"
                rows.append({"record_id": record_id, "source_id": source, "family_id": f"{source}:f{family_index}"})
                counts[record_id] = 100 + family_index
        train_a, val_a, _ = split_families(rows, counts, 0.10, 1515)
        train_b, val_b, _ = split_families(rows, counts, 0.10, 1515)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertFalse(train_a & val_a)
        self.assertTrue(val_a)

    def test_fixed_generator_repeats_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            np.arange(200, dtype=np.uint16).tofile(path)
            data = np.memmap(path, dtype=np.uint16, mode="r")
            generator_a = torch.Generator(device="cpu").manual_seed(9)
            generator_b = torch.Generator(device="cpu").manual_seed(9)
            first = get_batch(data, 2, 8, torch.device("cpu"), generator=generator_a)
            second = get_batch(data, 2, 8, torch.device("cpu"), generator=generator_b)
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))
            del data

    def test_resume_rejects_changed_schedule(self) -> None:
        checkpoint = {"resume_signature": {"max_steps": 10}}
        config = {"max_steps": 11}
        with self.assertRaises(SystemExit):
            validate_resume_signature(checkpoint, config)

    def test_tiny_training_stops_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
            for index in range(4, 64):
                vocab[f"t{index}"] = index
            tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))

            train_path = root / "train.bin"
            val_path = root / "val.bin"
            np.asarray([index % 64 for index in range(256)], dtype=np.uint16).tofile(train_path)
            np.asarray([(index * 3) % 64 for index in range(128)], dtype=np.uint16).tofile(val_path)
            manifest = {
                "model_training_allowed": True,
                "training_scope": "test_smoke",
                "train_bin_sha256": sha256_file(train_path),
                "val_bin_sha256": sha256_file(val_path),
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            run_root = root / "run"
            config = {
                "run_name": "test_smoke",
                "training_scope": "test_smoke",
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_sha256": sha256_file(tokenizer_path),
                "train_data": str(train_path),
                "val_data": str(val_path),
                "dataset_manifest": str(manifest_path),
                "dataset_manifest_sha256": sha256_file(manifest_path),
                "out_dir": str(run_root),
                "seed": 3,
                "eval_seed": 30,
                "torch_threads": 1,
                "vocab_size": 64,
                "seq_len": 8,
                "batch_size": 1,
                "grad_accum_steps": 1,
                "max_steps": 4,
                "eval_interval": 1,
                "eval_iters": 1,
                "save_interval": 1,
                "learning_rate": 0.001,
                "min_learning_rate": 0.0001,
                "warmup_steps": 1,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "n_layers": 1,
                "d_model": 16,
                "n_heads": 2,
                "n_kv_heads": 1,
                "d_ff": 32,
                "dropout": 0.0,
                "rope_base": 10000.0,
                "train_loss_mask": None,
                "val_loss_mask": None,
            }
            config_path = root / "config.json"
            config_path.write_bytes(canonical_json_bytes(config))
            first = subprocess.run(
                [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(config_path), "--stop-after-step", "2"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            checkpoint = torch.load(run_root / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["step"], 2)

            second = subprocess.run(
                [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(config_path), "--auto-resume"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("completed step 2", second.stdout)
            checkpoint = torch.load(run_root / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["step"], 4)
            metrics = [json.loads(line) for line in (run_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(metrics[0]["step"], 0)
            self.assertEqual(metrics[-1]["step"], 4)


if __name__ == "__main__":
    unittest.main()
