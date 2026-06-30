from __future__ import annotations

import hashlib
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

BUILD_SPEC = importlib.util.spec_from_file_location("build_stage_a_tokenizer_sample", SCRIPTS / "build_stage_a_tokenizer_sample.py")
assert BUILD_SPEC and BUILD_SPEC.loader
builder = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(builder)

VERIFY_SPEC = importlib.util.spec_from_file_location("verify_stage_a_tokenizer_sample", SCRIPTS / "verify_stage_a_tokenizer_sample.py")
assert VERIFY_SPEC and VERIFY_SPEC.loader
verifier = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(verifier)


class StageATokenizerAdmissionTests(unittest.TestCase):
    def test_candidate_build_approval_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            filtered = repo / "data" / "foundation" / "filtered"
            self._make_repo(repo)
            plan_path = repo / "data" / "foundation" / "manifests" / "plan.json"
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(json.dumps(self._plan()), encoding="utf-8")
            candidate = repo / "data" / "foundation" / "approved" / "candidate"
            manifest = builder.build_candidate(repo, plan_path, filtered, candidate, True, False)
            self.assertFalse(manifest["tokenizer_training_allowed"])
            rows = verifier.load_jsonl(candidate / "records.jsonl")
            self.assertEqual(sum(row["group_id"] == "frozen_stage_b" for row in rows), 2)
            self.assertTrue(all(row["training_allowed"] is False for row in rows))
            self.assertNotIn("self_stale", {row["record_id"] for row in rows})
            summary, errors = verifier.verify(candidate, repo, require_approved=False)
            self.assertEqual(errors, [])
            self.assertFalse(summary["approved"])

            decision = repo / "decision.json"
            decision.write_text(json.dumps({
                "schema_version": "1.0",
                "review_id": "test_review",
                "candidate_manifest_sha256": hashlib.sha256((candidate / "manifest.json").read_bytes()).hexdigest(),
                "status": "approved",
                "reviewer": "Leo",
                "reviewed_utc": "2026-06-30T00:00:00Z",
                "approved_for_tokenizer_comparison": True,
                "model_training_allowed": False,
                "excluded_record_ids": [],
            }), encoding="utf-8")
            approved = repo / "data" / "foundation" / "approved" / "approved"
            approved_manifest = builder.approve_candidate(candidate, decision, approved, False)
            self.assertTrue(approved_manifest["tokenizer_training_allowed"])
            self.assertFalse(approved_manifest["model_training_allowed"])
            summary, errors = verifier.verify(approved, repo, require_approved=True)
            self.assertEqual(errors, [])
            self.assertTrue(summary["approved"])

    def test_approval_is_bound_to_exact_candidate_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "manifest.json").write_text(json.dumps({
                "status": "tokenizer_sample_candidate_pending_review",
                "tokenizer_training_allowed": False,
                "model_training_allowed": False,
                "minimum_proxy_tokens_after_review": 1,
                "maximum_proxy_tokens_after_review": 10,
            }), encoding="utf-8")
            (candidate / "records.jsonl").write_text("", encoding="utf-8")
            decision = root / "decision.json"
            decision.write_text(json.dumps({
                "status": "approved",
                "approved_for_tokenizer_comparison": True,
                "model_training_allowed": False,
                "reviewer": "Leo",
                "reviewed_utc": "2026-06-30T00:00:00Z",
                "candidate_manifest_sha256": "wrong",
            }), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "not bound"):
                builder.approve_candidate(candidate, decision, root / "approved", False)

    def test_verifier_detects_text_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            filtered = repo / "data" / "foundation" / "filtered"
            self._make_repo(repo)
            plan_path = repo / "data" / "foundation" / "manifests" / "plan.json"
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(json.dumps(self._plan()), encoding="utf-8")
            candidate = repo / "data" / "foundation" / "approved" / "candidate"
            builder.build_candidate(repo, plan_path, filtered, candidate, True, False)
            row = verifier.load_jsonl(candidate / "records.jsonl")[0]
            (candidate / row["text_relative_path"]).write_text("tampered\n", encoding="utf-8")
            _, errors = verifier.verify(candidate, repo, require_approved=False)
            self.assertTrue(any("hash mismatch" in error for error in errors))


    def test_stage_b_eval_overlap_is_excluded_from_tokenizer_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "data" / "eval").mkdir(parents=True)
            (repo / "data" / "incoming").mkdir(parents=True)
            (repo / "data" / "raw").mkdir(parents=True)
            (repo / "data" / "eval" / "protected.jsonl").write_text(
                json.dumps({
                    "id": "protected_prompt",
                    "prompt": "Create a TypeScript helper that requests JSON and aborts after a timeout.",
                }) + "\n",
                encoding="utf-8",
            )
            (repo / "data" / "raw" / "stage_b.txt").write_text(
                "<instruction>Create a TypeScript helper that requests JSON and aborts after a timeout.</instruction>\n"
                "<answer>Use AbortController and clear the timer in a finally block.</answer>\n"
                "<instruction>Return a copied list.</instruction>\n"
                "<answer>Use a new list so the caller's data remains unchanged.</answer>\n",
                encoding="utf-8",
            )
            policy = {"input_paths": ["data/raw/stage_b.txt"]}
            selected, excluded, total = builder.load_stage_b(
                repo,
                policy,
                builder.evaluation_index(repo),
            )
            self.assertEqual(total, 2)
            self.assertEqual(len(selected), 1)
            self.assertEqual(len(excluded), 1)
            self.assertIn("protected_evaluation_overlap", excluded[0]["reason"])

    def test_parameter_count_formula_matches_phase3_sizes(self) -> None:
        # Keep this test independent of the optional tokenizers package by
        # checking the documented closed-form values directly.
        architecture = {"n_layers": 4, "d_model": 192, "n_heads": 4, "n_kv_heads": 2, "d_ff": 576}
        d = architecture["d_model"]
        head = d // architecture["n_heads"]
        kv = architecture["n_kv_heads"] * head
        base = architecture["n_layers"] * (2 * d + d * d + 2 * d * kv + d * d + 3 * d * architecture["d_ff"]) + d
        self.assertEqual(base + 2000 * d, 2_155_200)
        self.assertEqual(base + 4000 * d, 2_539_200)
        self.assertEqual(base + 8000 * d, 3_307_200)

    def _make_repo(self, repo: Path) -> None:
        (repo / "data" / "eval").mkdir(parents=True)
        (repo / "data" / "incoming").mkdir(parents=True)
        (repo / "data" / "raw").mkdir(parents=True)
        stage_b = (
            "<instruction>Return one useful value.</instruction>\n<answer>Use the first valid value and preserve order.</answer>\n"
            "<instruction>Explain a safe change.</instruction>\n<answer>Validate the path before writing the file.</answer>\n"
        )
        (repo / "data" / "raw" / "stage_b.txt").write_text(stage_b, encoding="utf-8")

        repo_source = repo / "data" / "foundation" / "filtered" / "repo_source"
        (repo_source / "files").mkdir(parents=True)
        repo_text = "A small Python function validates input before returning a copied collection. It has a focused test and a clear docstring."
        repo_file = repo_source / "files" / "sample.py"
        repo_file.write_text(repo_text, encoding="utf-8")
        repo_row = {
            "record_id": "repo:sample.py",
            "source_id": "repo_source",
            "status": "filtered_candidate_not_admitted",
            "training_allowed": False,
            "review_reasons": [],
            "rejection_reasons": [],
            "family_id": "repo:sample",
            "logical_path": "src/sample.py",
            "filtered_relative_path": "files/sample.py",
            "filtered_sha256": hashlib.sha256(repo_file.read_bytes()).hexdigest(),
            "token_count_proxy": 30,
        }
        (repo_source / "candidates.jsonl").write_text(json.dumps(repo_row) + "\n", encoding="utf-8")

        wiki_source = repo / "data" / "foundation" / "filtered" / "wikimedia_english_20260601" / "wiki_source"
        wiki_source.mkdir(parents=True)
        wiki_row = {
            "record_id": "wiki:1",
            "source_id": "wiki_source",
            "status": "filtered_candidate_not_admitted",
            "training_allowed": False,
            "review_reasons": [],
            "rejection_reasons": [],
            "family_id": "wiki:family:one",
            "component": "simple_general_english",
            "quality_score": 60,
            "title": "Clear explanation",
            "page_id": "1",
            "revision_id": "10",
            "revision_timestamp": "2026-06-01T00:00:00Z",
            "segment_index": 1,
            "text": "A clear explanation connects a condition to its consequence. The example uses ordinary English and keeps the steps in a sensible order.",
        }
        (wiki_source / "candidates.jsonl").write_text(json.dumps(wiki_row) + "\n", encoding="utf-8")

        self_path = repo / "data" / "foundation" / "internal_seed" / "self.jsonl"
        self_path.parent.mkdir(parents=True)
        self_rows = [
            {"id": "self_stable", "text": "Neoma should separate known facts from assumptions.", "family_id": "self", "content_sha256": "a", "factual_basis": ["honesty"], "training_allowed": False},
            {"id": "self_stale", "text": "The tokenizer has not been selected yet.", "family_id": "self", "content_sha256": "b", "factual_basis": ["temporary"], "training_allowed": False},
        ]
        self_path.write_text("".join(json.dumps(row) + "\n" for row in self_rows), encoding="utf-8")

    def _plan(self) -> dict:
        return {
            "schema_version": "1.0",
            "baseline": builder.BASELINE,
            "training_allowed": False,
            "sample_budget": {"target_proxy_tokens": 200, "minimum_proxy_tokens_after_review": 1, "maximum_proxy_tokens_after_review": 1000},
            "protected_data_policy": {"stage_b_expected_record_count": 2},
            "selection_groups": [
                {"group_id": "frozen_stage_b", "target_proxy_tokens": 100, "selection_mode": "include_all_frozen_records", "family_cap": 0, "input_paths": ["data/raw/stage_b.txt"]},
                {"group_id": "repository_sources", "target_proxy_tokens": 100, "selection_mode": "clean_candidates_ranked_with_family_caps", "family_cap": 2, "source_quotas": {"repo_source": 100}, "minimum_record_tokens": 1, "maximum_record_tokens": 1000},
                {"group_id": "wikimedia_english", "target_proxy_tokens": 100, "selection_mode": "highest_quality_clean_candidates_with_topic_diversity", "family_cap": 1, "source_quotas": {"wiki_source": 100}, "minimum_quality_score": 20},
                {"group_id": "neoma_self_knowledge", "target_proxy_tokens": 100, "selection_mode": "explicit_stable_id_allowlist", "family_cap": 0, "input_path": "data/foundation/internal_seed/self.jsonl", "admitted_ids": ["self_stable"], "deferred_ids": {"self_stale": "temporary"}},
            ],
        }


if __name__ == "__main__":
    unittest.main()
