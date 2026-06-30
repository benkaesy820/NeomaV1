from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_stage_a_english_alternatives as validator


class StageAEnglishAlternativesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.external = json.loads(validator.DEFAULT_EXTERNAL.read_text(encoding="utf-8"))
        cls.internal = json.loads(validator.DEFAULT_INTERNAL.read_text(encoding="utf-8"))

    def test_committed_plans_validate(self) -> None:
        report = validator.validate(validator.DEFAULT_EXTERNAL, validator.DEFAULT_INTERNAL)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["external_source_count"], 3)
        self.assertEqual(report["external_target_tokens"], 12_000_000)
        self.assertEqual(report["internal_component_count"], 6)
        self.assertEqual(report["internal_seed_target_tokens"], 60_000)

    def test_wikimedia_completion_status_uses_dumpstatus_job_value(self) -> None:
        controls = self.external["common_controls"]
        self.assertEqual(controls["required_articles_multistream_status"], "done")

    def test_gptnl_override_is_rejected(self) -> None:
        plan = copy.deepcopy(self.external)
        plan["gptnl_policy"]["allow_manual_download"] = True
        errors = validator.validate_external(plan)
        self.assertIn("GPT-NL manual download must be false", errors)

    def test_training_permission_is_rejected(self) -> None:
        plan = copy.deepcopy(self.external)
        plan["sources"][0]["training_allowed"] = True
        errors = validator.validate_external(plan)
        self.assertTrue(any("training_allowed must be false" in error for error in errors))

    def test_moving_or_old_snapshot_is_rejected(self) -> None:
        plan = copy.deepcopy(self.external)
        plan["sources"][0]["snapshot"] = "latest"
        plan["sources"][0]["archive_url"] = plan["sources"][0]["archive_url"].replace("20260601", "latest")
        errors = validator.validate_external(plan)
        self.assertTrue(any("snapshot must be 20260601" in error for error in errors))

    def test_internal_seed_total_is_enforced(self) -> None:
        plan = copy.deepcopy(self.internal)
        plan["components"][0]["target_tokens"] += 1
        errors = validator.validate_internal(plan)
        self.assertTrue(any("expected 60000" in error for error in errors))

    def test_self_knowledge_has_grounding_rule(self) -> None:
        component = next(item for item in self.internal["components"] if item["component_id"] == "neoma_self_knowledge")
        joined = " ".join(component["verification"]).lower()
        self.assertIn("model-card", joined)
        self.assertFalse(component["training_allowed"])


if __name__ == "__main__":
    unittest.main()
