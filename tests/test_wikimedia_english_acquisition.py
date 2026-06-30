from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ACQUIRE_PATH = ROOT / "scripts" / "acquire_wikimedia_english_sources.py"
VERIFY_PATH = ROOT / "scripts" / "verify_wikimedia_english_acquisitions.py"

ACQUIRE_SPEC = importlib.util.spec_from_file_location("acquire_wikimedia_english_sources", ACQUIRE_PATH)
assert ACQUIRE_SPEC and ACQUIRE_SPEC.loader
acquire = importlib.util.module_from_spec(ACQUIRE_SPEC)
ACQUIRE_SPEC.loader.exec_module(acquire)

VERIFY_SPEC = importlib.util.spec_from_file_location("verify_wikimedia_english_acquisitions", VERIFY_PATH)
assert VERIFY_SPEC and VERIFY_SPEC.loader
verify_mod = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(verify_mod)

PLAN_PATH = ROOT / "data" / "foundation" / "manifests" / "stage_a_safe_english_alternatives_v1_candidate.json"


class WikimediaEnglishAcquisitionTests(unittest.TestCase):
    def test_plan_loads_exact_three_sources(self) -> None:
        plan = acquire.load_plan(PLAN_PATH)
        self.assertFalse(plan["training_allowed"])
        self.assertEqual({source["source_id"] for source in plan["sources"]}, acquire.EXPECTED_SOURCE_IDS)
        self.assertEqual(plan["common_controls"]["required_articles_multistream_status"], "done")

    def test_sha1_manifest_parser_selects_named_files(self) -> None:
        parsed = acquire.parse_sha1sums(
            "0" * 40 + "  archive.xml.bz2\n"
            + "1" * 40 + "  archive-index.txt.bz2\n"
        )
        self.assertEqual(parsed["archive.xml.bz2"], "0" * 40)
        self.assertEqual(parsed["archive-index.txt.bz2"], "1" * 40)

    def test_url_policy_rejects_unapproved_sources(self) -> None:
        acquire.validate_url("https://dumps.wikimedia.org/simplewiki/20260601/", ["dumps.wikimedia.org"])
        with self.assertRaises(acquire.AcquisitionError):
            acquire.validate_url("http://dumps.wikimedia.org/simplewiki/20260601/", ["dumps.wikimedia.org"])
        with self.assertRaises(acquire.AcquisitionError):
            acquire.validate_url("https://example.com/simplewiki/20260601/", ["dumps.wikimedia.org"])

    def test_verify_detects_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "archive.xml.bz2"
            index = root / "index.txt.bz2"
            sha1_manifest = root / "sha1.txt"
            md5_manifest = root / "md5.txt"
            legal = root / "legal.html"
            for path, payload in {
                artifact: b"archive",
                index: b"index",
                sha1_manifest: b"sha1",
                md5_manifest: b"md5",
                legal: b"legal",
            }.items():
                path.write_bytes(payload)
            source_id = "simplewiki_20260601"
            manifest = {
                "source_id": source_id,
                "status": "acquired_quarantined_pending_review",
                "training_allowed": False,
                "content_extracted": False,
                "dumpstatus": {"articles_multistream_status": "done"},
                "artifacts": {
                    "archive": {
                        "path": str(artifact),
                        "sha1": hashlib.sha1(b"archive").hexdigest(),
                        "official_sha1": hashlib.sha1(b"archive").hexdigest(),
                        "sha256": "bad",
                    },
                    "index": {
                        "path": str(index),
                        "sha1": hashlib.sha1(b"index").hexdigest(),
                        "official_sha1": hashlib.sha1(b"index").hexdigest(),
                        "sha256": hashlib.sha256(b"index").hexdigest(),
                    },
                    "sha1_manifest": {"path": str(sha1_manifest), "sha256": hashlib.sha256(b"sha1").hexdigest()},
                    "md5_manifest": {"path": str(md5_manifest), "sha256": hashlib.sha256(b"md5").hexdigest()},
                    "license_page": {"path": str(legal), "sha256": hashlib.sha256(b"legal").hexdigest()},
                },
            }
            manifest_root = root / "manifests"
            manifest_root.mkdir()
            (manifest_root / f"{source_id}.wikimedia_acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
            plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
            plan["sources"] = [source for source in plan["sources"] if source["source_id"] == source_id]
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            _, errors = verify_mod.verify(plan_path, manifest_root, require_all=True)
        self.assertTrue(any("archive SHA-256 mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
