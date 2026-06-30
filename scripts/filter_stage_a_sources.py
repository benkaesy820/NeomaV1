#!/usr/bin/env python3
"""Decode and filter locally staged Stage A source files without admitting training data.

The command is a review-candidate builder. It verifies staged bytes, decodes text,
applies deterministic quality/security rules, constructs source-local document
families, checks protected evaluation/instruction overlap, and emits explicit
candidate/review/rejection manifests. Every output keeps training_allowed=false.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
from typing import Any

from stage_a_filtering_common import (
    StagingError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    build_protected_index,
    choose_duplicate_status,
    decode_text,
    leakage_findings,
    load_jsonl,
    load_protected_items,
    manifest_digest,
    normalized_utf8_text,
    path_quality_reason,
    quality_findings,
    record_fingerprints,
    safe_destination,
    safe_record_path,
    sha256_file,
    simhash_bands,
    source_family,
)
from stage_a_staging_common import load_json, selected_source_ids, utc_now


BASELINE = "2c74c34"


def load_filtering_plan(filtering_plan_path: Path, staging_plan_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    filtering = load_json(filtering_plan_path)
    staging = load_json(staging_plan_path)
    if filtering.get("training_allowed") is not False or staging.get("training_allowed") is not False:
        raise StagingError("filtering and staging plans must keep training_allowed=false")
    filter_sources = filtering.get("sources")
    staging_sources = staging.get("sources")
    if not isinstance(filter_sources, list) or not isinstance(staging_sources, list):
        raise StagingError("filtering/staging sources must be arrays")
    filter_ids = [row.get("source_id") for row in filter_sources]
    staging_ids = [row.get("source_id") for row in staging_sources]
    if len(filter_ids) != len(set(filter_ids)) or len(staging_ids) != len(set(staging_ids)):
        raise StagingError("duplicate source IDs in filtering/staging plan")
    if filter_ids != staging_ids:
        raise StagingError("filtering source order/IDs must match staging plan")
    for row in filter_sources:
        source_id = row.get("source_id")
        if row.get("training_allowed") is not False:
            raise StagingError(f"filter source grants training permission: {source_id}")
        mode = row.get("filtering_mode")
        if mode not in {"deferred", "staged_files"}:
            raise StagingError(f"{source_id}: invalid filtering_mode {mode!r}")
        for pattern in row.get("reject_path_regexes", []):
            try:
                re.compile(str(pattern))
            except re.error as exc:
                raise StagingError(f"{source_id}: invalid reject_path_regex {pattern!r}: {exc}") from exc
        for key in ("min_chars", "min_tokens", "max_decoded_chars", "max_line_chars"):
            if key in row and int(row[key]) < 0:
                raise StagingError(f"{source_id}: {key} must be non-negative")
    return filtering, staging


def _verify_staged_source(source_id: str, stage_root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    source_root = stage_root / source_id
    manifest_path = source_root / "staging_manifest.json"
    files_path = source_root / "files.jsonl"
    if not manifest_path.is_file() or not files_path.is_file():
        raise StagingError(f"{source_id}: staged manifest/files list missing")
    manifest = load_json(manifest_path)
    rows = load_jsonl(files_path)
    if manifest.get("training_allowed") is not False:
        raise StagingError(f"{source_id}: staging manifest grants training permission")
    if manifest.get("source_id") != source_id:
        raise StagingError(f"{source_id}: staging source ID mismatch")
    if int(manifest.get("staged_file_count", -1)) != len(rows):
        raise StagingError(f"{source_id}: staged file count mismatch")
    for row in rows:
        if row.get("training_allowed") is not False:
            raise StagingError(f"{source_id}: staged row grants training permission")
    return source_root, manifest, rows


def _base_record(source_id: str, logical_path: str, staging_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_id": f"{source_id}:{logical_path}",
        "source_id": source_id,
        "logical_path": logical_path,
        "language_hint": staging_row.get("language_hint", "unknown"),
        "staged_sha256": staging_row.get("sha256"),
        "staged_size_bytes": staging_row.get("size_bytes"),
        "training_allowed": False,
    }


def _inspect_one(
    source_id: str,
    source_policy: dict[str, Any],
    source_root: Path,
    staging_row: dict[str, Any],
    protected: list[Any],
) -> tuple[dict[str, Any], str | None]:
    logical_path = staging_row.get("logical_path")
    relative_path = staging_row.get("relative_staged_path")
    if not isinstance(logical_path, str) or not isinstance(relative_path, str):
        raise StagingError(f"{source_id}: staged row is missing logical/relative path")
    record = _base_record(source_id, logical_path, staging_row)
    staged_path = safe_destination(source_root, relative_path)
    if not staged_path.is_file() or staged_path.is_symlink():
        record.update(status="rejected_not_admitted", rejection_reasons=["missing_or_nonregular_staged_file"])
        return record, None
    actual_size = staged_path.stat().st_size
    actual_hash = sha256_file(staged_path)
    if actual_size != staging_row.get("size_bytes") or actual_hash != staging_row.get("sha256"):
        record.update(status="rejected_not_admitted", rejection_reasons=["staged_file_integrity_mismatch"], actual_sha256=actual_hash, actual_size_bytes=actual_size)
        return record, None

    path_reason = path_quality_reason(logical_path, source_policy)
    if path_reason:
        record.update(status="rejected_not_admitted", rejection_reasons=[path_reason])
        return record, None

    payload = staged_path.read_bytes()
    try:
        decoded = decode_text(
            payload,
            allow_cp1252=bool(source_policy.get("allow_cp1252", True)),
            max_control_ratio=float(source_policy.get("max_control_ratio", 0.001)),
        )
    except StagingError as exc:
        record.update(status="rejected_not_admitted", rejection_reasons=[f"decode_rejected:{exc}"])
        return record, None

    text = normalized_utf8_text(decoded.text)
    reject_reasons, review_reasons = quality_findings(logical_path, text, source_policy)
    leakage = leakage_findings(text, protected)
    critical_leakage = [row for row in leakage if row["severity"] == "critical"]
    review_leakage = [row for row in leakage if row["severity"] == "review"]
    if critical_leakage:
        reject_reasons.append("protected_evaluation_leakage")
    if review_leakage:
        review_reasons.append("protected_instruction_or_partial_eval_overlap")

    family_id, family_rule = source_family(source_id, logical_path)
    fingerprints = record_fingerprints(text)
    record.update(
        encoding=decoded.encoding,
        had_bom=decoded.had_bom,
        newline_style=decoded.newline_style,
        control_ratio=round(decoded.control_ratio, 8),
        normalized_newlines=True,
        family_id=family_id,
        family_rule=family_rule,
        leakage_findings=leakage,
        review_reasons=sorted(set(review_reasons)),
        rejection_reasons=sorted(set(reject_reasons)),
        **fingerprints,
    )
    if reject_reasons:
        record["status"] = "rejected_not_admitted"
        return record, None
    if review_reasons:
        record["status"] = "human_review_required_not_admitted"
    else:
        record["status"] = "filtered_candidate_not_admitted"
    return record, text


def _apply_global_dedup(records_with_text: list[tuple[dict[str, Any], str | None]]) -> None:
    exact: dict[str, dict[str, Any]] = {}
    templates: dict[str, dict[str, Any]] = {}
    bands: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for record, text in records_with_text:
        if text is None or record.get("status") == "rejected_not_admitted":
            continue
        prior: dict[str, Any] | None = exact.get(record["normalized_sha256"])
        if prior is None:
            prior = templates.get(record["template_sha256"])
        possible: dict[str, dict[str, Any]] = {}
        if prior is None:
            simhash = int(record["simhash64"], 16)
            for band in simhash_bands(simhash):
                for candidate in bands.get(band, []):
                    possible[candidate["record_id"]] = candidate
            for candidate in sorted(possible.values(), key=lambda row: row["record_id"]):
                action, reason = choose_duplicate_status(record, candidate)
                if action != "keep":
                    prior = candidate
                    if action == "reject":
                        record["status"] = "rejected_not_admitted"
                        record.setdefault("rejection_reasons", []).append(reason)
                    else:
                        record["status"] = "human_review_required_not_admitted"
                        record.setdefault("review_reasons", []).append(reason)
                    record["duplicate_of"] = candidate["record_id"]
                    record["duplicate_class"] = action
                    break
        else:
            action, reason = choose_duplicate_status(record, prior)
            record["status"] = "rejected_not_admitted" if action == "reject" else "human_review_required_not_admitted"
            key = "rejection_reasons" if action == "reject" else "review_reasons"
            record.setdefault(key, []).append(reason)
            record["duplicate_of"] = prior["record_id"]
            record["duplicate_class"] = action

        record["rejection_reasons"] = sorted(set(record.get("rejection_reasons", [])))
        record["review_reasons"] = sorted(set(record.get("review_reasons", [])))
        if record["status"] == "rejected_not_admitted":
            continue
        exact.setdefault(record["normalized_sha256"], record)
        templates.setdefault(record["template_sha256"], record)
        simhash = int(record["simhash64"], 16)
        for band in simhash_bands(simhash):
            bands[band].append(record)


def _atomic_replace_directory(temp: Path, target: Path, force: bool) -> None:
    if target.exists():
        if not force:
            raise StagingError(f"output exists; review it and rerun with --force: {target}")
        backup = target.with_name(target.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(target, backup)
        try:
            os.replace(temp, target)
        except Exception:
            os.replace(backup, target)
            raise
        shutil.rmtree(backup)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, target)


def execute_filtering(
    repo_root: Path,
    filtering_plan: dict[str, Any],
    selected: list[str],
    stage_root: Path,
    filter_root: Path,
    force: bool,
) -> dict[str, Any]:
    selected_set = set(selected)
    source_map = {row["source_id"]: row for row in filtering_plan["sources"]}
    protected = build_protected_index(load_protected_items(repo_root))
    inspected: list[tuple[dict[str, Any], str | None]] = []
    staging_inputs: list[Path] = []
    deferred: list[str] = []

    for source_id in selected:
        policy = source_map[source_id]
        if policy.get("filtering_mode") == "deferred":
            deferred.append(source_id)
            continue
        source_root, _, rows = _verify_staged_source(source_id, stage_root)
        staging_inputs.extend([source_root / "staging_manifest.json", source_root / "files.jsonl"])
        for row in sorted(rows, key=lambda item: str(item.get("logical_path", ""))):
            inspected.append(_inspect_one(source_id, policy, source_root, row, protected))

    _apply_global_dedup(inspected)
    parent = filter_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=filter_root.name + ".tmp-", dir=parent))
    try:
        by_source: dict[str, list[tuple[dict[str, Any], str | None]]] = defaultdict(list)
        for pair in inspected:
            by_source[pair[0]["source_id"]].append(pair)

        source_summaries: list[dict[str, Any]] = []
        for source_id in selected:
            policy = source_map[source_id]
            source_dir = temp / source_id
            source_dir.mkdir(parents=True, exist_ok=True)
            if source_id in deferred:
                manifest = {
                    "schema_version": "1.0",
                    "baseline": BASELINE,
                    "source_id": source_id,
                    "status": "deferred_not_filtered_not_admitted",
                    "reason": policy.get("defer_reason", "source deferred by filtering plan"),
                    "candidate_count": 0,
                    "review_count": 0,
                    "rejected_count": 0,
                    "training_allowed": False,
                }
                atomic_write_jsonl(source_dir / "candidates.jsonl", [])
                atomic_write_jsonl(source_dir / "review_queue.jsonl", [])
                atomic_write_jsonl(source_dir / "rejections.jsonl", [])
                atomic_write_jsonl(source_dir / "families.jsonl", [])
                atomic_write_json(source_dir / "filtering_manifest.json", manifest)
                source_summaries.append(manifest)
                continue

            rows = by_source.get(source_id, [])
            candidates: list[dict[str, Any]] = []
            review: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            families: dict[str, list[str]] = defaultdict(list)
            copied_bytes = 0
            for record, text in rows:
                status = record["status"]
                if status == "rejected_not_admitted":
                    rejected.append(record)
                    continue
                if text is None:
                    raise StagingError(f"{record['record_id']}: non-rejected record has no decoded text")
                output = safe_record_path(temp, source_id, record["logical_path"])
                atomic_write_text(output, text)
                relative = output.relative_to(temp / source_id).as_posix()
                record["filtered_relative_path"] = relative
                record["filtered_sha256"] = sha256_file(output)
                copied_bytes += output.stat().st_size
                families[record["family_id"]].append(record["record_id"])
                if status == "human_review_required_not_admitted":
                    review.append(record)
                else:
                    candidates.append(record)

            family_rows = [
                {
                    "schema_version": "1.0",
                    "source_id": source_id,
                    "family_id": family_id,
                    "member_count": len(member_ids),
                    "member_ids": sorted(member_ids),
                    "status": "family_candidate_not_admitted",
                    "training_allowed": False,
                }
                for family_id, member_ids in sorted(families.items())
            ]
            atomic_write_jsonl(source_dir / "candidates.jsonl", candidates)
            atomic_write_jsonl(source_dir / "review_queue.jsonl", review)
            atomic_write_jsonl(source_dir / "rejections.jsonl", rejected)
            atomic_write_jsonl(source_dir / "families.jsonl", family_rows)
            reason_counts = Counter(reason for record in rejected for reason in record.get("rejection_reasons", []))
            review_counts = Counter(reason for record in review for reason in record.get("review_reasons", []))
            manifest = {
                "schema_version": "1.0",
                "baseline": BASELINE,
                "generated_utc": utc_now(),
                "source_id": source_id,
                "status": "filtered_candidates_not_admitted",
                "input_file_count": len(rows),
                "candidate_count": len(candidates),
                "review_count": len(review),
                "rejected_count": len(rejected),
                "family_count": len(family_rows),
                "filtered_bytes": copied_bytes,
                "rejection_reason_counts": dict(sorted(reason_counts.items())),
                "review_reason_counts": dict(sorted(review_counts.items())),
                "protected_item_count": len(protected.items),
                "training_allowed": False,
            }
            atomic_write_json(source_dir / "filtering_manifest.json", manifest)
            source_summaries.append(manifest)

        summary_counts = Counter()
        for row in source_summaries:
            summary_counts["candidate_count"] += int(row.get("candidate_count", 0))
            summary_counts["review_count"] += int(row.get("review_count", 0))
            summary_counts["rejected_count"] += int(row.get("rejected_count", 0))
            summary_counts["family_count"] += int(row.get("family_count", 0))
        summary = {
            "schema_version": "1.0",
            "baseline": BASELINE,
            "generated_utc": utc_now(),
            "selected_source_ids": selected,
            "source_count": len(selected),
            "deferred_source_ids": deferred,
            "dedup_scope": "selected_sources_global",
            "protected_item_count": len(protected.items),
            "staging_input_manifest_digest": manifest_digest(staging_inputs) if staging_inputs else None,
            "counts": dict(summary_counts),
            "sources": source_summaries,
            "status": "filtered_candidates_not_admitted",
            "training_allowed": False,
        }
        atomic_write_json(temp / "stage_a_sources_v1_filtering_summary.json", summary)
        _atomic_replace_directory(temp, filter_root, force)
        return summary
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def dry_run_summary(filtering_plan: dict[str, Any], selected: list[str], repo_root: Path) -> dict[str, Any]:
    source_map = {row["source_id"]: row for row in filtering_plan["sources"]}
    protected = build_protected_index(load_protected_items(repo_root))
    return {
        "schema_version": "1.0",
        "baseline": BASELINE,
        "mode": "dry_run",
        "selected_sources": [
            {
                "source_id": source_id,
                "filtering_mode": source_map[source_id].get("filtering_mode"),
                "training_allowed": False,
            }
            for source_id in selected
        ],
        "protected_item_count": len(protected.items),
        "will_decode": [source_id for source_id in selected if source_map[source_id].get("filtering_mode") != "deferred"],
        "deferred": [source_id for source_id in selected if source_map[source_id].get("filtering_mode") == "deferred"],
        "training_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--filtering-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_filtering_plan.json"))
    parser.add_argument("--staging-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_staging_plan.json"))
    parser.add_argument("--stage-root", type=Path, default=Path("data/foundation/staged"))
    parser.add_argument("--filter-root", type=Path, default=Path("data/foundation/filtered"))
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    filtering, _ = load_filtering_plan(args.filtering_plan, args.staging_plan)
    selected = selected_source_ids(filtering, args.source, args.all)
    if not selected:
        raise StagingError("select at least one source with --source or --all")
    if args.execute:
        report = execute_filtering(args.repo_root.resolve(), filtering, selected, args.stage_root, args.filter_root, args.force)
    else:
        report = dry_run_summary(filtering, selected, args.repo_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True) + "\n", end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StagingError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
