#!/usr/bin/env python3
"""Verify local Wikimedia English filtering outputs without reading raw dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SOURCE_IDS = ("simplewiki_20260601", "enwikibooks_20260601", "enwikiversity_20260601")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify(filter_root: Path, require_all: bool) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    summary_path = filter_root / "stage_a_wikimedia_filtering_summary.json"
    if not summary_path.exists():
        if require_all:
            errors.append(f"missing filtering summary: {summary_path}")
        return results, errors
    summary = load_json(summary_path)
    if summary.get("training_allowed") is not False:
        errors.append("summary training_allowed must be false")
    for source_id in EXPECTED_SOURCE_IDS:
        source_dir = filter_root / source_id
        manifest_path = source_dir / "filtering_manifest.json"
        if not manifest_path.exists():
            if require_all:
                errors.append(f"missing source filtering manifest: {source_id}")
            results.append({"source_id": source_id, "status": "missing"})
            continue
        manifest = load_json(manifest_path)
        if manifest.get("training_allowed") is not False:
            errors.append(f"{source_id}: manifest training_allowed must be false")
        candidates = load_jsonl(source_dir / "candidates.jsonl")
        review = load_jsonl(source_dir / "review_queue.jsonl")
        rejections = load_jsonl(source_dir / "rejections_sample.jsonl")
        families = load_jsonl(source_dir / "families.jsonl")
        for bucket, rows in (("candidate", candidates), ("review", review), ("rejection", rejections)):
            for row in rows:
                if row.get("training_allowed") is not False:
                    errors.append(f"{source_id}: {bucket} row grants training permission: {row.get('record_id')}")
                text = row.get("text")
                if isinstance(text, str) and row.get("decoded_sha256") != sha256_text(text):
                    errors.append(f"{source_id}: decoded_sha256 mismatch: {row.get('record_id')}")
                if row.get("status") == "filtered_candidate_not_admitted" and row.get("rejection_reasons"):
                    errors.append(f"{source_id}: candidate has rejection reasons: {row.get('record_id')}")
        if len(candidates) != manifest.get("candidate_count"):
            errors.append(f"{source_id}: candidate count mismatch")
        if len(review) != manifest.get("review_count"):
            errors.append(f"{source_id}: review count mismatch")
        if len(families) != manifest.get("family_count"):
            errors.append(f"{source_id}: family count mismatch")
        results.append({
            "source_id": source_id,
            "status": manifest.get("status"),
            "candidate_count": len(candidates),
            "review_count": len(review),
            "rejection_sample_count": len(rejections),
            "family_count": len(families),
            "training_allowed": False,
        })
    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-root", type=Path, default=Path("data/foundation/filtered/wikimedia_english_20260601"))
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results, errors = verify(args.filter_root, args.require_all)
    report = {
        "schema_version": "1.0",
        "filter_root": str(args.filter_root),
        "require_all": args.require_all,
        "results": results,
        "errors": errors,
        "ok": not errors,
        "training_allowed": False,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
