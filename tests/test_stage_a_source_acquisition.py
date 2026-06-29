from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "acquire_stage_a_sources.py"
SPEC = importlib.util.spec_from_file_location("acquire_stage_a_sources", MODULE_PATH)
assert SPEC and SPEC.loader
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)

PLAN_PATH = ROOT / "data" / "foundation" / "manifests" / "stage_a_sources_v1_acquisition_plan.json"


class StageASourceAcquisitionTests(unittest.TestCase):
    def test_plan_is_exactly_ten_quarantined_sources(self) -> None:
        plan = acquire.load_plan(PLAN_PATH)
        self.assertEqual(plan["baseline"], "8066b1f")
        self.assertFalse(plan["training_allowed"])
        self.assertEqual(len(plan["sources"]), 10)
        self.assertEqual(len({row["source_id"] for row in plan["sources"]}), 10)
        power_shell = next(row for row in plan["sources"] if row["source_id"] == "powershell_7_6_3")
        self.assertEqual(power_shell["ref"], "v7.6.3")
        self.assertTrue(all(row["training_allowed"] is False for row in plan["sources"]))

    def test_url_policy_rejects_http_credentials_and_unknown_hosts(self) -> None:
        with self.assertRaises(acquire.AcquisitionError):
            acquire.validate_url("http://example.com/a", ["example.com"])
        with self.assertRaises(acquire.AcquisitionError):
            acquire.validate_url("https://evil.example/a", ["example.com"])
        with self.assertRaises(acquire.AcquisitionError):
            acquire.validate_url("https://user:pass@example.com/a", ["example.com"])
        acquire.validate_url("https://example.com/a", ["example.com"])

    def test_archive_scan_hashes_license_and_accepts_safe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "safe.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                for name, payload in {
                    "project/LICENSE": b"test license\n",
                    "project/src/main.py": b"print('ok')\n",
                }.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            report = acquire.scan_archive(path, ["LICENSE"], 1024 * 1024)
            self.assertFalse(report["critical_security_issue"])
            self.assertEqual(report["member_count"], 2)
            self.assertEqual(
                report["license_file_hashes"][0]["sha256"],
                hashlib.sha256(b"test license\n").hexdigest(),
            )

    def test_archive_scan_flags_traversal_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe.tar"
            with tarfile.open(path, "w") as archive:
                payload = b"x"
                bad = tarfile.TarInfo("../escape.txt")
                bad.size = len(payload)
                archive.addfile(bad, io.BytesIO(payload))
                link = tarfile.TarInfo("project/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "/tmp/target"
                archive.addfile(link)
            report = acquire.scan_archive(path, ["LICENSE"], 1024 * 1024)
            self.assertTrue(report["critical_security_issue"])
            self.assertIn("../escape.txt", report["unsafe_paths"])
            self.assertIn("project/link", report["special_members"])

    def test_github_annotated_tag_resolves_to_commit(self) -> None:
        source = {
            "kind": "github_tag_archive",
            "owner": "o",
            "repo": "r",
            "ref": "v1",
            "api_base": "https://api.github.com",
            "allowed_hosts": ["api.github.com"],
        }
        responses = [
            {"object": {"type": "tag", "url": "https://api.github.com/tag/1", "sha": "a" * 40}},
            {"object": {"type": "commit", "sha": "b" * 40}},
        ]
        with patch.object(acquire, "fetch_json", side_effect=responses):
            resolved = acquire.github_resolve_commit(source, 1.0)
        self.assertEqual(resolved["resolved_commit"], "b" * 40)
        self.assertEqual(resolved["resolved_ref"], "v1")

    def test_github_head_resolves_default_branch(self) -> None:
        source = {
            "kind": "github_head_archive",
            "owner": "o",
            "repo": "r",
            "api_base": "https://api.github.com",
            "allowed_hosts": ["api.github.com"],
        }
        with patch.object(
            acquire,
            "fetch_json",
            side_effect=[{"default_branch": "main"}, {"sha": "c" * 40}],
        ):
            resolved = acquire.github_resolve_commit(source, 1.0)
        self.assertEqual(resolved, {"resolved_ref": "main", "resolved_commit": "c" * 40})

    def test_pypi_sdist_requires_one_source_distribution(self) -> None:
        source = {
            "metadata_url": "https://pypi.org/pypi/x/1/json",
            "allowed_hosts": ["pypi.org", "files.pythonhosted.org"],
        }
        response = {
            "urls": [
                {
                    "packagetype": "sdist",
                    "url": "https://files.pythonhosted.org/x.tar.gz",
                    "filename": "x.tar.gz",
                    "digests": {"sha256": "d" * 64},
                },
                {"packagetype": "bdist_wheel"},
            ]
        }
        with patch.object(acquire, "fetch_json", return_value=response):
            result = acquire.pypi_sdist(source, 1.0)
        self.assertEqual(result["published_sha256"], "d" * 64)

    def test_huggingface_warning_keeps_stream_manifest_on_hold(self) -> None:
        source = {
            "source_id": "hf",
            "repository_id": "org/repo",
            "api_url": "https://huggingface.co/api/datasets/org/repo",
            "allowed_hosts": ["huggingface.co"],
            "allowed_subsets": ["cc_english-pd"],
        }
        metadata = {
            "sha": "e" * 40,
            "securityStatus": {"unsafe": True},
            "siblings": [
                {"rfilename": "cc_english-pd/train.parquet", "lfs": {"sha256": "f" * 64}},
                {"rfilename": "other/train.parquet"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(acquire, "fetch_json", return_value=metadata):
            path = Path(tmp) / "manifest.json"
            result = acquire.acquire_huggingface_manifest(source, path, 1.0)
            stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(result["critical_security_issue"])
        self.assertEqual(stored["selected_sibling_count"], 1)
        self.assertFalse(stored["training_allowed"])

    def test_summary_never_grants_training_permission(self) -> None:
        plan = acquire.load_plan(PLAN_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = plan["sources"][0]["source_id"]
            acquire.atomic_write_json(
                root / f"{first}.acquisition.json",
                {
                    "status": "acquired_quarantined_pending_review",
                    "artifact": {"sha256": "0" * 64},
                    "resolved": {"revision": "1" * 40},
                    "security": {"hold": False},
                    "training_allowed": False,
                },
            )
            summary = acquire.build_summary(plan, root)
        self.assertFalse(summary["training_allowed"])
        self.assertTrue(all(row["training_allowed"] is False for row in summary["sources"]))


if __name__ == "__main__":
    unittest.main()
