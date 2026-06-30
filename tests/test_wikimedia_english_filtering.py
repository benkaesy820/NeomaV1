from __future__ import annotations

import bz2
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
FILTER_PATH = ROOT / "scripts" / "filter_wikimedia_english_sources.py"
VERIFY_PATH = ROOT / "scripts" / "verify_wikimedia_english_filtering.py"

FILTER_SPEC = importlib.util.spec_from_file_location("filter_wikimedia_english_sources", FILTER_PATH)
assert FILTER_SPEC and FILTER_SPEC.loader
wiki_filter = importlib.util.module_from_spec(FILTER_SPEC)
FILTER_SPEC.loader.exec_module(wiki_filter)

VERIFY_SPEC = importlib.util.spec_from_file_location("verify_wikimedia_english_filtering", VERIFY_PATH)
assert VERIFY_SPEC and VERIFY_SPEC.loader
wiki_verify = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(wiki_verify)


class WikimediaEnglishFilteringTests(unittest.TestCase):
    def test_wikitext_cleanup_removes_templates_refs_and_keeps_link_labels(self) -> None:
        raw = """
{{Infobox}}
'''Python''' is a [[programming language|programming language]].
It is used before tests are run because examples can explain behavior.<ref>note</ref>

== Uses ==
* It can preserve order.
"""
        cleaned, reasons = wiki_filter.clean_wikitext(raw)
        self.assertNotIn("Infobox", cleaned)
        self.assertNotIn("ref", cleaned)
        self.assertIn("programming language", cleaned)
        self.assertEqual(reasons, [])

    def test_segment_builder_respects_token_bounds(self) -> None:
        paragraphs = [
            "This paragraph explains a useful condition before a file is changed. It has enough words to matter.",
            "Another paragraph gives an example because the result should preserve order and avoid mutation.",
            "A final paragraph connects the requirement to a consequence after validation succeeds.",
        ]
        segments = wiki_filter.make_segments(paragraphs, min_tokens=20, max_tokens=80, max_segments=2)
        self.assertTrue(segments)
        self.assertLessEqual(len(segments), 2)

    def test_end_to_end_filtering_keeps_candidates_non_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            manifest_root = base / "manifests"
            raw_root = base / "raw"
            filter_root = base / "filtered"
            review_out = base / "review.json"
            (repo / "data" / "eval").mkdir(parents=True)
            (repo / "data" / "incoming").mkdir(parents=True)
            source_id = "simplewiki_20260601"
            archive = raw_root / source_id / "simple.xml.bz2"
            archive.parent.mkdir(parents=True)
            xml = self._xml_dump()
            archive.write_bytes(bz2.compress(xml.encode("utf-8")))
            manifest_root.mkdir()
            (manifest_root / f"{source_id}.wikimedia_acquisition.json").write_text(
                json.dumps({
                    "source_id": source_id,
                    "status": "acquired_quarantined_pending_review",
                    "training_allowed": False,
                    "dumpstatus": {"articles_multistream_status": "done"},
                    "artifacts": {"archive": {"path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}},
                }),
                encoding="utf-8",
            )
            plan = {
                "training_allowed": False,
                "common_controls": {"required_articles_multistream_status": "done"},
                "sources": [
                    {
                        "source_id": source_id,
                        "name": "Simple English Wikipedia",
                        "snapshot": "20260601",
                        "training_allowed": False,
                    }
                ],
            }
            plan_path = base / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            report = wiki_filter.run_filtering(
                repo_root=repo,
                plan_path=plan_path,
                manifest_root=manifest_root,
                filter_root=filter_root,
                review_out=review_out,
                selected_ids=[source_id],
                execute=True,
                force=False,
                page_limit=None,
                rejection_sample_limit=10,
            )
            self.assertFalse(report["training_allowed"])
            candidates = wiki_verify.load_jsonl(filter_root / source_id / "candidates.jsonl")
            self.assertTrue(candidates)
            self.assertTrue(all(row["training_allowed"] is False for row in candidates))
            _, errors = wiki_verify.verify(filter_root, require_all=False)
            self.assertEqual(errors, [])

    def _xml_dump(self) -> str:
        text = (
            "A careful program validates input before writing a file. "
            "It preserves existing data unless the requested change is safe. "
            "This means an empty value can be valid while a missing value is different.\n\n"
            "For example, a command may read configuration, check limits, and then return a result. "
            "The result should explain what happened because the caller needs a useful consequence.\n\n"
            "When a test describes the expected behavior, the implementation can follow the same order. "
            "The original collection remains unchanged, and a new collection contains the selected values."
        )
        return f"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
<page>
<title>Careful programming</title>
<ns>0</ns>
<id>1</id>
<revision><id>10</id><timestamp>2026-06-01T00:00:00Z</timestamp><text>{text}</text></revision>
</page>
<page>
<title>Redirect page</title>
<ns>0</ns>
<id>2</id>
<revision><id>20</id><timestamp>2026-06-01T00:00:00Z</timestamp><text>#REDIRECT [[Other]]</text></revision>
</page>
</mediawiki>"""


if __name__ == "__main__":
    unittest.main()
