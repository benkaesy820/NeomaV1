from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import inventory_stage_a_sources as inventory  # noqa: E402
import stage_a_staging_common as common  # noqa: E402
import stage_stage_a_sources as stage  # noqa: E402
import verify_stage_a_staging as verify  # noqa: E402

STAGING_PLAN = ROOT / "data" / "foundation" / "manifests" / "stage_a_sources_v1_staging_plan.json"
ACQUISITION_PLAN = ROOT / "data" / "foundation" / "manifests" / "stage_a_sources_v1_acquisition_plan.json"


class StageASourceStagingTests(unittest.TestCase):
    def test_staging_plan_has_ten_non_training_sources(self) -> None:
        staging, acquisition = common.load_plans(STAGING_PLAN, ACQUISITION_PLAN)
        self.assertEqual(staging["baseline"], "f378d0a")
        self.assertFalse(staging["training_allowed"])
        self.assertEqual(len(staging["sources"]), 10)
        self.assertEqual({row["source_id"] for row in staging["sources"]}, {row["source_id"] for row in acquisition["sources"]})
        self.assertTrue(all(row["training_allowed"] is False for row in staging["sources"]))
        gptnl = next(row for row in staging["sources"] if row["source_id"] == "gptnl_english_2026")
        self.assertEqual(gptnl["staging_mode"], "none")

    def test_path_rules_make_exclusions_win(self) -> None:
        policy = {
            "allowed_paths": ["Lib/"],
            "excluded_paths": ["Lib/test/**/data/"],
            "allowed_suffixes": [".py"],
            "max_file_bytes": 100,
        }
        self.assertEqual(common.classify_member("Lib/email/parser.py", 10, policy)[0], "selected")
        self.assertEqual(common.classify_member("Lib/test/json/data/sample.py", 10, policy)[0], "excluded_path")
        self.assertEqual(common.classify_member("Doc/readme.py", 10, policy)[0], "outside_allowlist")
        self.assertEqual(common.classify_member("Lib/readme.bin", 10, policy)[0], "unsupported_type")
        self.assertEqual(common.classify_member("Lib/huge.py", 101, policy)[0], "oversize_file")

    def test_archive_paths_reject_traversal(self) -> None:
        self.assertEqual(common.normalize_archive_path("./root/src/a.py"), "root/src/a.py")
        with self.assertRaises(common.StagingError):
            common.normalize_archive_path("../escape.py")
        with self.assertRaises(common.StagingError):
            common.normalize_archive_path("/absolute.py")
        with self.assertRaises(common.StagingError):
            common.normalize_archive_path("root/../../escape.py")

    def test_common_root_and_family_hints_are_deterministic(self) -> None:
        root = common.common_archive_root(["project/src/a.py", "project/tests/test_a.py"])
        self.assertEqual(root, "project")
        self.assertEqual(common.strip_root("project/src/a.py", root), "src/a.py")
        self.assertEqual(common.family_hint("x", "src/pkg/a.py"), "x:src/pkg")

    def test_tar_inventory_selects_only_allowed_regular_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "source.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                self._add_file(tar, "project/src/good.py", b"print('ok')\n")
                self._add_file(tar, "project/src/skip.bin", b"binary")
                self._add_file(tar, "project/vendor/other.py", b"print('no')\n")
                bad = tarfile.TarInfo("../escape.py")
                bad.size = 1
                tar.addfile(bad, io.BytesIO(b"x"))
                link = tarfile.TarInfo("project/src/link.py")
                link.type = tarfile.SYMTYPE
                link.linkname = "good.py"
                tar.addfile(link)
            policy = {
                "allowed_paths": ["src/"],
                "excluded_paths": ["src/generated/"],
                "allowed_suffixes": [".py"],
                "allowed_names": [],
                "max_file_bytes": 100,
                "max_selected_bytes": 1000,
            }
            rows, summary = inventory._tar_inventory(archive_path, "fixture", policy)
        selected = [row["logical_path"] for row in rows if row["selected_for_staging"]]
        self.assertEqual(selected, ["src/good.py"])
        self.assertEqual(summary["counts"]["unsafe_paths"], 1)
        self.assertEqual(summary["counts"]["special_members"], 1)

    def test_stage_pipeline_preserves_bytes_and_never_admits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            paths = self._fixture_source(base, security_hold=False)
            policy = paths["policy"]
            rows, summary = inventory.inventory_one("fixture", policy, paths["raw_root"], paths["manifest_root"])
            common.atomic_write_jsonl(paths["inventory_root"] / "fixture.inventory.jsonl", rows)
            common.atomic_write_json(paths["inventory_root"] / "fixture.inventory.summary.json", summary)
            staged = stage.stage_one(
                "fixture", policy, paths["raw_root"], paths["manifest_root"], paths["inventory_root"], paths["stage_root"], False
            )
            output = paths["stage_root"] / "fixture" / "files" / "src" / "good.py"
            self.assertEqual(output.read_bytes(), b"print('ok')\r\n")
            self.assertEqual(staged["staged_file_count"], 1)
            self.assertFalse(staged["training_allowed"])
            results, errors = verify.verify(paths["staging_plan"], paths["acquisition_plan"], paths["inventory_root"], paths["stage_root"], True)
            self.assertEqual(errors, [])
            self.assertTrue(results[0]["ok"])

    def test_staging_refuses_security_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture_source(Path(tmp), security_hold=True)
            rows, summary = inventory.inventory_one("fixture", paths["policy"], paths["raw_root"], paths["manifest_root"])
            common.atomic_write_jsonl(paths["inventory_root"] / "fixture.inventory.jsonl", rows)
            common.atomic_write_json(paths["inventory_root"] / "fixture.inventory.summary.json", summary)
            with self.assertRaises(common.StagingError):
                stage.stage_one(
                    "fixture", paths["policy"], paths["raw_root"], paths["manifest_root"], paths["inventory_root"], paths["stage_root"], False
                )

    def test_selected_rows_reject_duplicate_logical_paths(self) -> None:
        policy = {"allowed_paths": ["src/"], "excluded_paths": [], "allowed_suffixes": [".py"], "max_file_bytes": 100, "max_selected_bytes": 1000}
        row = {"source_id": "fixture", "logical_path": "src/a.py", "archive_member": "p/src/a.py", "size_bytes": 1, "selected_for_staging": True}
        with self.assertRaises(common.StagingError):
            stage.selected_rows([row, dict(row)], "fixture", policy)

    def test_verify_detects_staged_file_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture_source(Path(tmp), security_hold=False)
            rows, summary = inventory.inventory_one("fixture", paths["policy"], paths["raw_root"], paths["manifest_root"])
            common.atomic_write_jsonl(paths["inventory_root"] / "fixture.inventory.jsonl", rows)
            common.atomic_write_json(paths["inventory_root"] / "fixture.inventory.summary.json", summary)
            stage.stage_one("fixture", paths["policy"], paths["raw_root"], paths["manifest_root"], paths["inventory_root"], paths["stage_root"], False)
            output = paths["stage_root"] / "fixture" / "files" / "src" / "good.py"
            output.write_text("tampered\n", encoding="utf-8")
            _, errors = verify.verify(paths["staging_plan"], paths["acquisition_plan"], paths["inventory_root"], paths["stage_root"], True)
            self.assertTrue(any("hash mismatch" in error or "size mismatch" in error for error in errors))

    def test_metadata_only_stage_contains_no_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            target = base / "staged" / "gptnl"
            target.parent.mkdir(parents=True)
            acquisition_path = base / "acquisition.json"
            acquisition = {"artifact": {"sha256": "a" * 64}}
            common.atomic_write_json(acquisition_path, acquisition)
            inventory_path = base / "inventory.jsonl"
            common.atomic_write_jsonl(inventory_path, [])
            result = stage.stage_metadata_only(
                "gptnl", target, {"notes": "deferred"}, acquisition_path, acquisition, inventory_path, False
            )
            self.assertEqual(result["staged_file_count"], 0)
            self.assertFalse(result["training_allowed"])
            self.assertEqual((target / "files.jsonl").read_text(encoding="utf-8"), "")

    @staticmethod
    def _add_file(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    def _fixture_source(self, base: Path, security_hold: bool) -> dict[str, Path | dict]:
        raw_root = base / "raw"
        manifest_root = base / "manifests"
        inventory_root = base / "inventory"
        stage_root = base / "staged"
        for path in (raw_root, manifest_root, inventory_root, stage_root):
            path.mkdir(parents=True, exist_ok=True)
        source_dir = raw_root / "fixture"
        source_dir.mkdir()
        archive = source_dir / "fixture.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            self._add_file(tar, "project/src/good.py", b"print('ok')\r\n")
            self._add_file(tar, "project/other/skip.py", b"print('skip')\n")
        artifact_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        acquisition_manifest = {
            "source_id": "fixture",
            "status": "acquired_quarantined_pending_review",
            "artifact": {"filename": archive.name, "sha256": artifact_sha},
            "security": {"hold": security_hold},
            "training_allowed": False,
        }
        common.atomic_write_json(manifest_root / "fixture.acquisition.json", acquisition_manifest)
        policy = {
            "source_id": "fixture",
            "inventory_mode": "archive",
            "staging_mode": "allowed_archive_members",
            "allowed_paths": ["src/"],
            "excluded_paths": [],
            "allowed_suffixes": [".py"],
            "allowed_names": [],
            "max_file_bytes": 100,
            "max_selected_bytes": 1000,
            "training_allowed": False,
            "status": "staging_planned_not_admitted",
        }
        staging_plan = base / "staging_plan.json"
        acquisition_plan = base / "acquisition_plan.json"
        common.atomic_write_json(staging_plan, {"training_allowed": False, "sources": [policy]})
        common.atomic_write_json(acquisition_plan, {"training_allowed": False, "sources": [{"source_id": "fixture", "training_allowed": False}]})
        return {
            "raw_root": raw_root,
            "manifest_root": manifest_root,
            "inventory_root": inventory_root,
            "stage_root": stage_root,
            "policy": policy,
            "staging_plan": staging_plan,
            "acquisition_plan": acquisition_plan,
        }


if __name__ == "__main__":
    unittest.main()
