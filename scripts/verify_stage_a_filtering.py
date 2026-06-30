#!/usr/bin/env python3
"""Verify Stage A filtered-candidate outputs without granting training permission."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from stage_a_filtering_common import StagingError, atomic_write_json, load_jsonl, safe_destination, sha256_file
from stage_a_staging_common import load_json
from filter_stage_a_sources import load_filtering_plan


BASELINE = "2c74c34"


def verify_source(source: dict[str, Any], filter_root: Path) -> tuple[dict[str, Any], list[str]]:
    source_id = source["source_id"]
    errors: list[str] = []
    source_root = filter_root / source_id
    required = {
        "manifest": source_root / "filtering_manifest.json",
        "candidates": source_root / "candidates.jsonl",
        "review": source_root / "review_queue.jsonl",
        "rejections": source_root / "rejections.jsonl",
        "families": source_root / "families.jsonl",
    }
    for path in required.values():
        if not path.is_file():
            errors.append(f"missing required file: {path}")
    if errors:
        return {"source_id": source_id, "ok": False, "training_allowed": False}, errors

    manifest = load_json(required["manifest"])
    candidates = load_jsonl(required["candidates"])
    review = load_jsonl(required["review"])
    rejected = load_jsonl(required["rejections"])
    families = load_jsonl(required["families"])
    if manifest.get("training_allowed") is not False:
        errors.append("filtering manifest grants training permission")
    if manifest.get("source_id") != source_id:
        errors.append("source ID mismatch")
    if source.get("filtering_mode") == "deferred":
        if candidates or review or rejected or families:
            errors.append("deferred source emitted content rows")
        if manifest.get("status") != "deferred_not_filtered_not_admitted":
            errors.append("deferred source has wrong status")
        return {
            "source_id": source_id,
            "ok": not errors,
            "status": manifest.get("status"),
            "candidate_count": 0,
            "review_count": 0,
            "rejected_count": 0,
            "training_allowed": False,
        }, errors

    expected_counts = {
        "candidate_count": len(candidates),
        "review_count": len(review),
        "rejected_count": len(rejected),
        "family_count": len(families),
    }
    for key, expected in expected_counts.items():
        if int(manifest.get(key, -1)) != expected:
            errors.append(f"{key} mismatch: {manifest.get(key)} != {expected}")

    all_rows = [("candidate", row) for row in candidates] + [("review", row) for row in review] + [("rejected", row) for row in rejected]
    ids: set[str] = set()
    filtered_paths: set[str] = set()
    normalized_hashes: dict[str, str] = {}
    expected_family_members: dict[str, set[str]] = {}
    total_bytes = 0
    for bucket, row in all_rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            errors.append("row missing record_id")
            continue
        if record_id in ids:
            errors.append(f"duplicate record_id: {record_id}")
        ids.add(record_id)
        if row.get("source_id") != source_id:
            errors.append(f"source mismatch: {record_id}")
        if row.get("training_allowed") is not False:
            errors.append(f"record grants training permission: {record_id}")
        status = row.get("status")
        if bucket == "candidate" and status != "filtered_candidate_not_admitted":
            errors.append(f"candidate status mismatch: {record_id}")
        if bucket == "review" and status != "human_review_required_not_admitted":
            errors.append(f"review status mismatch: {record_id}")
        if bucket == "rejected" and status != "rejected_not_admitted":
            errors.append(f"rejection status mismatch: {record_id}")
        if status == "rejected_not_admitted":
            if row.get("filtered_relative_path"):
                errors.append(f"rejected row has filtered file: {record_id}")
            if not row.get("rejection_reasons"):
                errors.append(f"rejected row lacks reason: {record_id}")
            continue

        relative = row.get("filtered_relative_path")
        if not isinstance(relative, str) or not relative.startswith("files/"):
            errors.append(f"non-rejected row lacks valid filtered path: {record_id}")
            continue
        if relative in filtered_paths:
            errors.append(f"duplicate filtered path: {relative}")
        filtered_paths.add(relative)
        path = safe_destination(source_root, relative)
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing/nonregular filtered file: {relative}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != row.get("filtered_sha256"):
            errors.append(f"filtered hash mismatch: {relative}")
        total_bytes += path.stat().st_size
        normalized_hash = row.get("normalized_sha256")
        if isinstance(normalized_hash, str):
            previous = normalized_hashes.get(normalized_hash)
            if previous:
                errors.append(f"surviving exact normalized duplicate: {record_id} and {previous}")
            normalized_hashes[normalized_hash] = record_id
        family_id = row.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            errors.append(f"missing family_id: {record_id}")
        else:
            expected_family_members.setdefault(family_id, set()).add(record_id)
        for leakage in row.get("leakage_findings", []):
            if leakage.get("severity") == "critical":
                errors.append(f"critical eval leakage survived filtering: {record_id}")

    actual_files = {
        path.relative_to(source_root).as_posix()
        for path in (source_root / "files").rglob("*")
        if path.is_file()
    } if (source_root / "files").exists() else set()
    if actual_files != filtered_paths:
        errors.append(f"filtered file set mismatch; missing={sorted(filtered_paths - actual_files)[:10]}, extra={sorted(actual_files - filtered_paths)[:10]}")
    if total_bytes != int(manifest.get("filtered_bytes", -1)):
        errors.append("filtered byte total mismatch")

    family_ids: set[str] = set()
    for family in families:
        if family.get("training_allowed") is not False:
            errors.append(f"family grants training permission: {family.get('family_id')}")
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            errors.append("family row missing family_id")
            continue
        if family_id in family_ids:
            errors.append(f"duplicate family_id: {family_id}")
        family_ids.add(family_id)
        members = set(family.get("member_ids", []))
        if members != expected_family_members.get(family_id, set()):
            errors.append(f"family membership mismatch: {family_id}")
        if int(family.get("member_count", -1)) != len(members):
            errors.append(f"family count mismatch: {family_id}")
    if family_ids != set(expected_family_members):
        errors.append("family set mismatch")

    return {
        "source_id": source_id,
        "ok": not errors,
        **expected_counts,
        "filtered_bytes": total_bytes,
        "training_allowed": False,
    }, errors


def verify(filtering_plan_path: Path, staging_plan_path: Path, filter_root: Path, require_all: bool) -> tuple[list[dict[str, Any]], list[str]]:
    filtering, _ = load_filtering_plan(filtering_plan_path, staging_plan_path)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in filtering["sources"]:
        source_root = filter_root / source["source_id"]
        if not source_root.exists() and not require_all:
            results.append({"source_id": source["source_id"], "ok": None, "status": "not_filtered", "training_allowed": False})
            continue
        result, source_errors = verify_source(source, filter_root)
        results.append(result)
        errors.extend(f"{source['source_id']}: {message}" for message in source_errors)

    summary_path = filter_root / "stage_a_sources_v1_filtering_summary.json"
    if require_all and not summary_path.is_file():
        errors.append(f"missing global summary: {summary_path}")
    if summary_path.is_file():
        summary = load_json(summary_path)
        if summary.get("training_allowed") is not False:
            errors.append("global summary grants training permission")
        if require_all:
            expected_ids = [row["source_id"] for row in filtering["sources"]]
            if summary.get("selected_source_ids") != expected_ids:
                errors.append("global summary does not cover every planned source in order")
        totals = Counter()
        for result in results:
            if result.get("ok") is None:
                continue
            for key in ("candidate_count", "review_count", "rejected_count", "family_count"):
                totals[key] += int(result.get(key, 0))
        for key, expected in totals.items():
            if int(summary.get("counts", {}).get(key, -1)) != expected:
                errors.append(f"global {key} mismatch")
    return results, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtering-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_filtering_plan.json"))
    parser.add_argument("--staging-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_staging_plan.json"))
    parser.add_argument("--filter-root", type=Path, default=Path("data/foundation/filtered"))
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    results, errors = verify(args.filtering_plan, args.staging_plan, args.filter_root, args.require_all)
    report = {
        "schema_version": "1.0",
        "baseline": BASELINE,
        "require_all": args.require_all,
        "results": results,
        "errors": errors,
        "ok": not errors,
        "training_allowed": False,
    }
    if args.out:
        atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True) + "\n", end="")
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StagingError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
