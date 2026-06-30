from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gptnl_streaming_common as common
import stream_gptnl_english as stream
import verify_gptnl_streaming as verify
from stage_a_staging_common import atomic_write_json


class GPTNLStreamingTests(unittest.TestCase):
    def test_selected_manifest_rows_only_parquet(self) -> None:
        manifest = {
            "selected_siblings": [
                {"rfilename": "cc_english-pd/ALL_PARQUET_FILES_VALID"},
                {"rfilename": "cc_english-pd/a.parquet"},
                {"rfilename": "cc_openalex/b.parquet"},
            ]
        }
        self.assertEqual(common.selected_manifest_rows(manifest), ["cc_english-pd/a.parquet", "cc_openalex/b.parquet"])

    def test_security_summary_blocks_queued(self) -> None:
        summary = common.security_summary([
            {"security_safe": False, "security_status": "queued"},
            {"security_safe": True, "security_status": "safe"},
        ])
        self.assertFalse(summary["all_safe"])
        self.assertEqual(summary["blocked_count"], 1)

    def test_row_quality_rejects_short_menu_and_secret(self) -> None:
        reject, _ = common.row_quality("Home\nAbout\nContact", 20, 1000)
        self.assertIn("too_short_or_low_information", reject)
        reject, _ = common.row_quality("-----BEGIN PRIVATE KEY-----\n" + "word " * 100, 20, 10000)
        self.assertTrue(any(reason.startswith("possible_secret") for reason in reject))

    def test_choose_text_from_row_prefers_configured_column(self) -> None:
        text, column = common.choose_text_from_row({"body": "hello", "text": "chosen"}, ["text", "body"])
        self.assertEqual((text, column), ("chosen", "text"))

    def test_write_blocked_output_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            plan = {"source_id": "gptnl_english_2026", "repository_id": "repo"}
            manifest = {"resolved_revision": "abc", "training_allowed": False}
            rows = [{"path": "cc/a.parquet", "security_safe": False, "security_status": "queued", "training_allowed": False}]
            result = stream.write_blocked_output(output, plan, manifest, rows, False)
            self.assertEqual(result["status"], "blocked_security_queued")
            report, errors = verify.verify(output, True)
            self.assertEqual(errors, [])
            self.assertTrue(report["ok"])

    def test_plan_rejects_training_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            atomic_write_json(path, {"source_id": "gptnl_english_2026", "training_allowed": True})
            with self.assertRaises(Exception):
                stream.load_plan(path)


if __name__ == "__main__":
    unittest.main()
