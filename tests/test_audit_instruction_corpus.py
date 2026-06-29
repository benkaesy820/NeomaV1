from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_instruction_corpus.py"
SPEC = importlib.util.spec_from_file_location("audit_instruction_corpus", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class AuditInstructionCorpusTests(unittest.TestCase):
    def make_repo(self, root: Path) -> None:
        (root / "data" / "incoming").mkdir(parents=True)
        (root / "data" / "raw").mkdir(parents=True)
        (root / "data" / "eval").mkdir(parents=True)
        row = {
            "id": "b01_py_parse_name",
            "language": "python",
            "category": "validation",
            "difficulty": "basic",
            "instruction": "Write parse_name that rejects blank input.",
            "constraints": ["Trim whitespace."],
            "answer": "def parse_name(value: str) -> str:\n    value = value.strip()\n    if not value:\n        raise ValueError('blank')\n    return value",
            "bad_code": "",
            "reasoning": "",
            "edge_cases": ["Whitespace-only input."],
            "quality_notes": ["Small boundary lesson."],
        }
        incoming = root / "data" / "incoming" / "phase3_5b_batch01_accepted.jsonl"
        incoming.write_text(json.dumps(row) + "\n", encoding="utf-8")
        rendered = AUDIT.render_jsonl_row(row)
        (root / "data" / "raw" / "phase3_5b_batch01_accepted.txt").write_text(rendered + "\n", encoding="utf-8")
        legacy = """<instruction>\nExplain why exact validation matters.\n</instruction>\n<constraints>\n</constraints>\n<answer>\nExact checks reject partial matches.\n</answer>\n"""
        (root / "data" / "raw" / "code_instruction_seed.txt").write_text(legacy, encoding="utf-8")
        (root / "data" / "raw" / "code_basics.txt").write_text("<file path=\"x.py\">\nprint('x')\n</file>\n", encoding="utf-8")
        eval_row = {
            "id": "heldout_other",
            "language": "python",
            "category": "function",
            "prompt": "Write a function that adds two integers.",
            "required_terms": ["def add_two"],
            "forbidden_terms": [],
            "test_code": "assert add_two(1, 2) == 3",
            "expected_behavior_notes": "Return the exact sum.",
        }
        for name in ("code_prompts.jsonl", "phase3_5b_heldout_v1.jsonl"):
            (root / "data" / "eval" / name).write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

    def test_render_matches_importer_shape(self) -> None:
        row = {
            "instruction": "Do X.",
            "constraints": ["A", "B"],
            "bad_code": "bad()",
            "reasoning": "Because.",
            "answer": "good()",
            "edge_cases": ["Empty."],
            "quality_notes": ["Focused."],
        }
        rendered = AUDIT.render_jsonl_row(row)
        self.assertIn("<instruction>\nDo X.\n</instruction>", rendered)
        self.assertIn("<bad_code>\nbad()\n</bad_code>", rendered)
        self.assertIn("Edge cases:\n- Empty.", rendered)

    def test_loads_accepted_and_legacy_without_double_counting_rendered_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_repo(root)
            candidate = root / "data" / "incoming" / "phase3_5b_batch02_candidates.jsonl"
            candidate.write_text(json.dumps({"id": "must_not_count"}) + "\n", encoding="utf-8")
            accepted, integrity = AUDIT.load_accepted_records(root)
            legacy, foundation = AUDIT.load_legacy_records(root)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(legacy), 1)
            self.assertEqual(len(foundation), 1)
            self.assertTrue(integrity[0]["raw_matches_rendered_jsonl"])

    def test_duplicate_detection_finds_exact_instruction(self) -> None:
        base = AUDIT.AuditRecord(
            record_id="a", source_kind="accepted_jsonl", source_path="a", language="python", category="function",
            difficulty="basic", instruction="Return a copied list.", constraints=(), answer="def f(x):\n    return list(x)",
            bad_code="", reasoning="", edge_cases=(), quality_notes=(), rendered="Return a copied list.",
        )
        other = AUDIT.AuditRecord(**{**base.__dict__, "record_id": "b", "source_path": "b"})
        flags = AUDIT.duplicate_flags([base, other])
        self.assertTrue(any(flag.field == "instruction" and flag.reason == "exact normalized match" for flag in flags))

    def test_leakage_detects_exact_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_path = root / "eval.jsonl"
            eval_path.write_text(json.dumps({"id": "e1", "prompt": "Write exact secret parser.", "required_terms": [], "forbidden_terms": []}) + "\n", encoding="utf-8")
            record = AUDIT.AuditRecord(
                record_id="r1", source_kind="accepted_jsonl", source_path="x", language="python", category="function",
                difficulty="basic", instruction="Write exact secret parser.", constraints=(), answer="pass", bad_code="",
                reasoning="", edge_cases=(), quality_notes=(), rendered="Write exact secret parser.\npass",
            )
            flags = AUDIT.leakage_flags([record], [eval_path])
            self.assertTrue(any(flag.severity == "critical" and flag.field == "prompt" for flag in flags))

    def test_meaningful_eval_terms_suppresses_generic_words(self) -> None:
        terms = AUDIT.meaningful_eval_terms(
            {
                "required_terms": ["specific", "Number.isInteger", "safe redirect URL"],
                "forbidden_terms": ["parameter"],
            }
        )
        self.assertNotIn("specific", terms)
        self.assertNotIn("parameter", terms)
        self.assertIn("Number.isInteger", terms)
        self.assertIn("safe redirect URL", terms)

    def test_capability_tags_support_overlapping_coverage(self) -> None:
        record = AUDIT.AuditRecord(
            record_id="r", source_kind="accepted_jsonl", source_path="x", language="python", category="data",
            difficulty="basic", instruction="Validate and transform a JSON mapping.", constraints=(), answer="return value",
            bad_code="", reasoning="", edge_cases=(), quality_notes=(), rendered="Validate and transform a JSON mapping.",
        )
        tags = set(AUDIT.capability_tags(record))
        self.assertIn("functions_and_data", tags)
        self.assertIn("validation", tags)

    def test_curriculum_decision_uses_requested_context(self) -> None:
        lengths = [{"full_record_tokens": 150} for _ in range(50)]
        decision = AUDIT.curriculum_decision(300, [], [], lengths, [], 128)
        self.assertEqual(decision["status"], "context_length_cleanup_required")
        self.assertEqual(decision["expected_context"], 128)

    def test_missing_requested_tokenizer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(SystemExit):
                AUDIT.resolve_tokenizer(root, "missing.json")

    def test_human_review_queue_has_blank_reviewer_fields(self) -> None:
        duplicate = AUDIT.DuplicateFlag("a", "b", "answer", 1.0, "exact normalized match", "a.jsonl", "b.jsonl")
        rows = AUDIT.human_review_queue([duplicate], [], [], 256, "proxy_lexical_tokens")
        self.assertEqual(rows[0]["priority"], "critical")
        self.assertEqual(rows[0]["reviewer_decision"], "")
        self.assertEqual(rows[0]["action"], "")

    def test_audit_outputs_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            self.make_repo(root)
            first = root / "reports1"
            second = root / "reports2"
            AUDIT.audit(root, first, None, 256)
            AUDIT.audit(root, second, None, 256)
            for name in (
                "phase3_5b_corpus_audit.json",
                "phase3_5b_corpus_audit.md",
                "phase3_5b_near_duplicates.csv",
                "phase3_5b_eval_leakage_flags.csv",
                "phase3_5b_length_outliers.csv",
                "phase3_5b_record_inventory.csv",
                "phase3_5b_human_review_queue.csv",
                "phase3_5b_curriculum_gaps.md",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
