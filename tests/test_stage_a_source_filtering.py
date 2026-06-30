from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import filter_stage_a_sources as filtering
import stage_a_filtering_common as common
import verify_stage_a_filtering as verify
from stage_a_staging_common import StagingError, atomic_write_json, atomic_write_jsonl


class StageAFilteringTests(unittest.TestCase):
    def test_decode_utf8_bom_utf16_and_cp1252(self) -> None:
        utf8 = common.decode_text(b"\xef\xbb\xbfhello\r\nworld\r\n")
        self.assertEqual(utf8.text, "hello\r\nworld\r\n")
        self.assertEqual(utf8.encoding, "utf-8-sig")
        self.assertTrue(utf8.had_bom)
        self.assertEqual(utf8.newline_style, "crlf")

        utf16 = common.decode_text("hello\nworld".encode("utf-16"))
        self.assertEqual(utf16.text, "hello\nworld")
        self.assertEqual(utf16.encoding, "utf-16")

        cp1252 = common.decode_text("café — test".encode("cp1252"))
        self.assertEqual(cp1252.text, "café — test")
        self.assertEqual(cp1252.encoding, "cp1252")

    def test_decode_rejects_binary_looking_payload(self) -> None:
        with self.assertRaises(StagingError):
            common.decode_text(b"abc\x00def\x00ghi")
        with self.assertRaises(StagingError):
            common.decode_text(b"abc\x01\x02\x03def", max_control_ratio=0.01)

    def test_quality_filter_rejects_generated_and_repeated_templates(self) -> None:
        generated = "# This file is auto-generated. Do not edit.\n" + "value = 1\n" * 10
        rejected, _ = common.quality_findings("src/generated.py", generated, {"min_chars": 20, "min_tokens": 5})
        self.assertIn("generated_file_marker", rejected)

        repeated = "\n".join(["same template line"] * 30)
        rejected, _ = common.quality_findings("doc/repeated.md", repeated, {"min_chars": 20, "min_tokens": 5})
        self.assertIn("repeated_template_or_snapshot", rejected)

    def test_path_filter_rejects_vendor_and_generated_paths(self) -> None:
        self.assertIsNotNone(common.path_quality_reason("vendor/library/file.py", {}))
        self.assertIsNotNone(common.path_quality_reason("docs/generated/page.md", {}))
        self.assertIsNone(common.path_quality_reason("testing/fixtures/useful_example.py", {"allow_path_parts": ["fixtures"]}))

    def test_document_family_pairs_related_files(self) -> None:
        self.assertEqual(
            common.source_family("cpython_3_14_6", "Lib/json/__init__.py")[0],
            common.source_family("cpython_3_14_6", "Lib/test/test_json.py")[0],
        )
        self.assertEqual(
            common.source_family("node_24_18_0", "doc/api/fs.md")[0],
            common.source_family("node_24_18_0", "lib/fs.js")[0],
        )
        self.assertEqual(
            common.source_family("postgresql_18_4", "src/test/regress/sql/transactions.sql")[0],
            common.source_family("postgresql_18_4", "src/test/regress/expected/transactions.out")[0],
        )

    def test_duplicate_fingerprints_detect_exact_and_near(self) -> None:
        first = common.record_fingerprints("def add(a, b):\n    return a + b\n" * 4)
        second = common.record_fingerprints("def add(a, b):\n    return a + b\n" * 4)
        left = {"record_id": "a", **first}
        right = {"record_id": "b", **second}
        action, reason = common.choose_duplicate_status(right, left)
        self.assertEqual(action, "reject")
        self.assertIn("duplicate", reason)

    def test_eval_leakage_is_critical_and_instruction_overlap_is_review(self) -> None:
        eval_text = "Return the first value only when every input is valid and preserve the original order exactly."
        instruction_text = (
            "Explain why the smallest safe change preserves existing behavior and avoids rewriting unrelated modules. "
            "State the assumption clearly and add one focused regression test before changing anything else."
        )
        protected = [
            common.ProtectedItem("eval:x", "evaluation", eval_text, common.normalize_text(eval_text), common.shingle_hashes(eval_text)),
            common.ProtectedItem("instruction:y", "instruction", instruction_text, common.normalize_text(instruction_text), common.shingle_hashes(instruction_text)),
        ]
        findings = common.leakage_findings(f"Header. {eval_text} Footer. {instruction_text}", protected)
        severities = {(row["source_kind"], row["severity"]) for row in findings}
        self.assertIn(("evaluation", "critical"), severities)
        self.assertIn(("instruction", "review"), severities)

    def test_end_to_end_filtering_keeps_everything_non_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            plan = json.loads(paths["filtering_plan"].read_text(encoding="utf-8"))
            summary = filtering.execute_filtering(
                paths["repo_root"], plan, ["gpt_fixture", "source_a", "source_b"], paths["stage_root"], paths["filter_root"], False
            )
            self.assertFalse(summary["training_allowed"])
            self.assertEqual(summary["deferred_source_ids"], ["gpt_fixture"])

            candidates = common.load_jsonl(paths["filter_root"] / "source_a" / "candidates.jsonl")
            review_rows = common.load_jsonl(paths["filter_root"] / "source_a" / "review_queue.jsonl")
            rejected = common.load_jsonl(paths["filter_root"] / "source_a" / "rejections.jsonl")
            all_rows = candidates + review_rows + rejected
            self.assertTrue(all(row["training_allowed"] is False for row in all_rows))
            self.assertTrue(any("protected_evaluation_leakage" in row.get("rejection_reasons", []) for row in rejected))
            self.assertTrue(any(any(reason.startswith("exact_normalized_duplicate_of:") for reason in row.get("rejection_reasons", [])) for row in rejected))
            self.assertTrue(any("protected_instruction_or_partial_eval_overlap" in row.get("review_reasons", []) for row in review_rows))

            results, errors = verify.verify(paths["filtering_plan"], paths["staging_plan"], paths["filter_root"], True)
            self.assertEqual(errors, [])
            self.assertTrue(all(row["ok"] for row in results))

    def test_verifier_detects_filtered_tampering_and_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            plan = json.loads(paths["filtering_plan"].read_text(encoding="utf-8"))
            filtering.execute_filtering(
                paths["repo_root"], plan, ["gpt_fixture", "source_a", "source_b"], paths["stage_root"], paths["filter_root"], False
            )
            candidates = common.load_jsonl(paths["filter_root"] / "source_a" / "candidates.jsonl")
            self.assertTrue(candidates)
            candidate_path = paths["filter_root"] / "source_a" / candidates[0]["filtered_relative_path"]
            candidate_path.write_text("tampered\n", encoding="utf-8")
            extra = paths["filter_root"] / "source_a" / "files" / "extra.py"
            extra.write_text("print('extra')\n", encoding="utf-8")
            _, errors = verify.verify(paths["filtering_plan"], paths["staging_plan"], paths["filter_root"], True)
            self.assertTrue(any("hash mismatch" in error for error in errors))
            self.assertTrue(any("file set mismatch" in error for error in errors))

    def test_protected_loader_includes_eval_accepted_and_legacy_instruction_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "eval").mkdir(parents=True)
            (root / "data" / "incoming").mkdir(parents=True)
            (root / "data" / "raw").mkdir(parents=True)
            (root / "data" / "eval" / "eval.jsonl").write_text(
                json.dumps({"id": "e", "prompt": "This evaluation wording is deliberately long enough to become protected text."}) + "\n",
                encoding="utf-8",
            )
            (root / "data" / "incoming" / "phase3_5b_batch01_accepted.jsonl").write_text(
                json.dumps({"id": "a", "answer": "This accepted answer is deliberately long enough to be protected from silent foundation overlap."}) + "\n",
                encoding="utf-8",
            )
            (root / "data" / "raw" / "code_instruction_seed.txt").write_text(
                "<instruction>This cleaned legacy instruction is deliberately long enough to remain protected during filtering.</instruction>",
                encoding="utf-8",
            )
            items = common.load_protected_items(root)
            kinds = {item.source_kind for item in items}
            ids = {item.protected_id for item in items}
            self.assertIn("evaluation", kinds)
            self.assertIn("instruction", kinds)
            self.assertTrue(any(value.startswith("code_instruction_seed.txt:legacy_") for value in ids))

    def test_filtering_plan_rejects_training_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filtering_plan = root / "filter.json"
            staging_plan = root / "stage.json"
            atomic_write_json(filtering_plan, {"training_allowed": True, "sources": []})
            atomic_write_json(staging_plan, {"training_allowed": False, "sources": []})
            with self.assertRaises(StagingError):
                filtering.load_filtering_plan(filtering_plan, staging_plan)

    def test_dry_run_does_not_require_local_staged_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp), create_staged=False)
            plan = json.loads(paths["filtering_plan"].read_text(encoding="utf-8"))
            report = filtering.dry_run_summary(plan, ["gpt_fixture", "source_a"], paths["repo_root"])
            self.assertEqual(report["deferred"], ["gpt_fixture"])
            self.assertEqual(report["will_decode"], ["source_a"])
            self.assertFalse(report["training_allowed"])

    def _fixture(self, base: Path, create_staged: bool = True) -> dict[str, Path]:
        repo_root = base / "repo"
        stage_root = base / "staged"
        filter_root = base / "filtered"
        (repo_root / "data" / "eval").mkdir(parents=True)
        (repo_root / "data" / "incoming").mkdir(parents=True)
        eval_prompt = "Return the first value only when every input is valid and preserve the original order exactly."
        with (repo_root / "data" / "eval" / "stage_a_eval.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": "eval1", "prompt": eval_prompt, "answer": "B"}) + "\n")
        instruction_answer = (
            "Explain why the smallest safe change preserves existing behavior and avoids rewriting unrelated modules. "
            "State the assumption clearly and add one focused regression test before changing anything else."
        )
        with (repo_root / "data" / "incoming" / "phase3_5b_batch99_accepted.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": "ins1", "instruction": "Review this patch.", "answer": instruction_answer}) + "\n")

        policies = [
            {"source_id": "gpt_fixture", "filtering_mode": "deferred", "defer_reason": "test", "training_allowed": False},
            {
                "source_id": "source_a", "filtering_mode": "staged_files", "training_allowed": False,
                "min_chars": 20, "min_tokens": 5, "max_decoded_chars": 100000, "allow_cp1252": True,
                "reject_path_parts": [], "allow_path_parts": [], "reject_path_regexes": [], "reject_names": [],
            },
            {
                "source_id": "source_b", "filtering_mode": "staged_files", "training_allowed": False,
                "min_chars": 20, "min_tokens": 5, "max_decoded_chars": 100000, "allow_cp1252": True,
                "reject_path_parts": [], "allow_path_parts": [], "reject_path_regexes": [], "reject_names": [],
            },
        ]
        staging_sources = [
            {"source_id": row["source_id"], "staging_mode": "none" if row["source_id"] == "gpt_fixture" else "allowed_archive_members", "training_allowed": False}
            for row in policies
        ]
        filtering_plan = base / "filtering_plan.json"
        staging_plan = base / "staging_plan.json"
        atomic_write_json(filtering_plan, {"training_allowed": False, "sources": policies})
        atomic_write_json(staging_plan, {"training_allowed": False, "sources": staging_sources})

        if create_staged:
            self._write_staged_source(stage_root, "gpt_fixture", {})
            good = (
                "def preserve_order(values):\n"
                "    result = []\n"
                "    for value in values:\n"
                "        if value not in result:\n"
                "            result.append(value)\n"
                "    return result\n"
            ).encode("utf-8")
            generated = ("# This file is generated. Do not edit.\n" + "value = 1\n" * 8).encode("utf-8")
            instruction = instruction_answer.encode("utf-8")
            self._write_staged_source(stage_root, "source_a", {
                "src/good.py": good,
                "src/dup.py": good,
                "docs/eval.md": ("Before. " + eval_prompt + " After.").encode("utf-8"),
                "docs/instruction.md": instruction,
                "src/generated.py": generated,
                "src/binary.py": b"abc\x00def\x00ghi",
            })
            self._write_staged_source(stage_root, "source_b", {
                "src/other.py": b"def multiply(a, b):\n    # Return the product without modifying either input.\n    return a * b\n",
            })
        return {
            "repo_root": repo_root,
            "stage_root": stage_root,
            "filter_root": filter_root,
            "filtering_plan": filtering_plan,
            "staging_plan": staging_plan,
        }

    def _write_staged_source(self, stage_root: Path, source_id: str, files: dict[str, bytes]) -> None:
        source_root = stage_root / source_id
        rows = []
        total = 0
        for logical_path, payload in sorted(files.items()):
            output = source_root / "files" / logical_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            total += len(payload)
            rows.append({
                "source_id": source_id,
                "logical_path": logical_path,
                "relative_staged_path": f"files/{logical_path}",
                "size_bytes": len(payload),
                "sha256": digest,
                "language_hint": "python" if logical_path.endswith(".py") else "documentation",
                "training_allowed": False,
            })
        source_root.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(source_root / "files.jsonl", rows)
        atomic_write_json(source_root / "staging_manifest.json", {
            "source_id": source_id,
            "staged_file_count": len(rows),
            "staged_bytes": total,
            "training_allowed": False,
        })


if __name__ == "__main__":
    unittest.main()
