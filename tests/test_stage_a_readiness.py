from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


READY = load_module("validate_stage_a_readiness", "scripts/validate_stage_a_readiness.py")
SCORE = load_module("score_stage_a_english_eval", "scripts/score_stage_a_english_eval.py")


class StageAReadinessTests(unittest.TestCase):
    def test_choice_extraction(self) -> None:
        self.assertEqual(SCORE.extract_choice("B"), "B")
        self.assertEqual(SCORE.extract_choice("Answer: c."), "C")
        self.assertIsNone(SCORE.extract_choice("unknown"))

    def test_new_suites_have_expected_shape(self) -> None:
        dev, errors = READY.validate_eval(ROOT / "data/eval/stage_a_english_dev_v1.jsonl", "stage_a_english_dev_v1", "development")
        locked, more = READY.validate_eval(ROOT / "data/eval/stage_a_english_locked_v1.jsonl", "stage_a_english_locked_v1", "heldout")
        self.assertEqual(errors + more, [])
        self.assertEqual(len(dev), 48)
        self.assertEqual(len(locked), 48)
        self.assertTrue(all(row["training_allowed"] is False for row in dev + locked))

    def test_balanced_categories(self) -> None:
        rows = READY.load_jsonl(ROOT / "data/eval/stage_a_english_dev_v1.jsonl") + READY.load_jsonl(ROOT / "data/eval/stage_a_english_locked_v1.jsonl")
        counts = {}
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        self.assertEqual(set(counts.values()), {12})

    def test_source_manifest_targets_and_freshness(self) -> None:
        manifest, errors = READY.validate_manifest(ROOT / "data/foundation/manifests/stage_a_sources_v1_candidate.json")
        self.assertEqual(errors, [])
        self.assertEqual(manifest["external_target_tokens"], 45_000_000)
        self.assertEqual(manifest["total_target_tokens"], 50_000_000)
        self.assertEqual(len(manifest["sources"]), 10)
        self.assertTrue(all(source["training_allowed"] is False for source in manifest["sources"]))

    def test_normalization_and_similarity(self) -> None:
        self.assertEqual(READY.normalize("A  B!"), "a b")
        self.assertEqual(READY.jaccard({1,2}, {2,3}), 1/3)

    def test_duplicate_eval_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.jsonl"
            row={"id":"same","language":"text","category":"x","difficulty":"basic","prompt":"P","scoring":"choice","accepted_answers":["A"],"rationale":"R","suite":"s","split":"heldout","training_allowed":False,"source_family":"x"}
            path.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n",encoding="utf-8")
            _, errors=READY.validate_eval(path,"s","heldout")
            self.assertTrue(any("duplicate id" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
