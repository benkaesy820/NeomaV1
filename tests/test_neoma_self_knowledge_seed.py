from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT_PATH = ROOT / "scripts" / "build_neoma_self_knowledge_seed.py"
SPEC = importlib.util.spec_from_file_location("build_neoma_self_knowledge_seed", SCRIPT_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

CARD_PATH = ROOT / "data" / "foundation" / "manifests" / "neoma_model_card_v0_1_candidate.json"


class NeomaSelfKnowledgeSeedTests(unittest.TestCase):
    def test_model_card_builds_fifty_non_training_records(self) -> None:
        records = builder.build_records(builder.load_card(CARD_PATH))
        self.assertEqual(len(records), 50)
        self.assertTrue(all(row["training_allowed"] is False for row in records))
        self.assertTrue(all(row["component_id"] == "neoma_self_knowledge" for row in records))
        joined = "\n".join(row["text"] for row in records).lower()
        self.assertIn("cannot be the sole approver", joined)
        self.assertIn("random weights", joined)
        self.assertIn("unless", joined)
        self.assertIn("missing from empty", joined)

    def test_generation_writes_jsonl_with_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "seed.jsonl"
            records = builder.build_records(builder.load_card(CARD_PATH))
            builder.atomic_write_jsonl(out, records)
            loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(loaded), 50)
        self.assertTrue(all(row["content_sha256"] for row in loaded))


if __name__ == "__main__":
    unittest.main()
