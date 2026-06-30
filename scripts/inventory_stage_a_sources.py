#!/usr/bin/env python3
"""Inventory quarantined Stage A sources against explicit allowed-path rules.

This command never grants training permission and never extracts archive content.
It writes local-only inventory JSON/JSONL files for Leo review.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tarfile
from typing import Any
import zipfile

from stage_a_staging_common import (
    StagingError,
    atomic_write_json,
    atomic_write_jsonl,
    classify_member,
    common_archive_root,
    family_hint,
    language_hint,
    load_json,
    load_plans,
    normalize_archive_path,
    selected_source_ids,
    sha256_file,
    strip_root,
    utc_now,
)


def acquisition_manifest(manifest_root: Path, source_id: str) -> dict[str, Any]:
    path = manifest_root / f"{source_id}.acquisition.json"
    value = load_json(path)
    if value.get("source_id") != source_id:
        raise StagingError(f"acquisition manifest source mismatch: {path}")
    if value.get("training_allowed") is not False:
        raise StagingError(f"acquisition manifest grants training permission: {path}")
    if not str(value.get("status", "")).startswith("acquired_") and not str(value.get("status", "")).startswith("stream_manifest_"):
        raise StagingError(f"source is not acquired: {source_id}: {value.get('status')}")
    return value


def artifact_path(raw_root: Path, source_id: str, manifest: dict[str, Any]) -> Path:
    filename = manifest.get("artifact", {}).get("filename")
    if not isinstance(filename, str) or not filename:
        raise StagingError(f"acquisition manifest has no artifact filename: {source_id}")
    path = raw_root / source_id / filename
    if not path.is_file():
        raise StagingError(f"quarantined artifact missing: {path}")
    expected = manifest.get("artifact", {}).get("sha256")
    actual = sha256_file(path)
    if not isinstance(expected, str) or actual.lower() != expected.lower():
        raise StagingError(f"artifact hash mismatch for {source_id}: {actual} != {expected}")
    return path


def _tar_inventory(path: Path, source_id: str, policy: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        root = common_archive_root(member.name for member in members)
        counts: Counter[str] = Counter()
        selected_bytes = 0
        special: list[str] = []
        unsafe: list[str] = []
        for member in members:
            counts["archive_members"] += 1
            if member.isdir():
                counts["directories"] += 1
                continue
            if not member.isfile():
                counts["special_members"] += 1
                special.append(member.name)
                continue
            counts["regular_files"] += 1
            try:
                archive_path = normalize_archive_path(member.name)
                logical = strip_root(archive_path, root)
                if not logical:
                    continue
            except StagingError:
                counts["unsafe_paths"] += 1
                unsafe.append(member.name)
                continue
            classification, reason, selected = classify_member(logical, int(member.size), policy)
            counts[classification] += 1
            if selected:
                selected_bytes += int(member.size)
            rows.append({
                "schema_version": "1.0",
                "source_id": source_id,
                "archive_member": archive_path,
                "archive_root": root,
                "logical_path": logical,
                "size_bytes": int(member.size),
                "classification": classification,
                "reason": reason,
                "selected_for_staging": selected,
                "language_hint": language_hint(logical),
                "family_hint": family_hint(source_id, logical),
                "training_allowed": False,
            })
    summary = {
        "archive_type": "tar",
        "archive_root": root,
        "counts": dict(sorted(counts.items())),
        "selected_bytes": selected_bytes,
        "selected_byte_limit": int(policy.get("max_selected_bytes", 0)),
        "selected_within_limit": selected_bytes <= int(policy.get("max_selected_bytes", 0)),
        "unsafe_path_samples": unsafe[:50],
        "special_member_samples": special[:50],
    }
    return sorted(rows, key=lambda row: row["logical_path"]), summary


def _zip_inventory(path: Path, source_id: str, policy: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        root = common_archive_root(info.filename for info in infos)
        counts: Counter[str] = Counter()
        selected_bytes = 0
        unsafe: list[str] = []
        for info in infos:
            counts["archive_members"] += 1
            if info.is_dir():
                counts["directories"] += 1
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                counts["special_members"] += 1
                continue
            counts["regular_files"] += 1
            try:
                archive_path = normalize_archive_path(info.filename)
                logical = strip_root(archive_path, root)
                if not logical:
                    continue
            except StagingError:
                counts["unsafe_paths"] += 1
                unsafe.append(info.filename)
                continue
            classification, reason, selected = classify_member(logical, int(info.file_size), policy)
            counts[classification] += 1
            if selected:
                selected_bytes += int(info.file_size)
            rows.append({
                "schema_version": "1.0",
                "source_id": source_id,
                "archive_member": archive_path,
                "archive_root": root,
                "logical_path": logical,
                "size_bytes": int(info.file_size),
                "classification": classification,
                "reason": reason,
                "selected_for_staging": selected,
                "language_hint": language_hint(logical),
                "family_hint": family_hint(source_id, logical),
                "training_allowed": False,
            })
    summary = {
        "archive_type": "zip",
        "archive_root": root,
        "counts": dict(sorted(counts.items())),
        "selected_bytes": selected_bytes,
        "selected_byte_limit": int(policy.get("max_selected_bytes", 0)),
        "selected_within_limit": selected_bytes <= int(policy.get("max_selected_bytes", 0)),
        "unsafe_path_samples": unsafe[:50],
        "special_member_samples": [],
    }
    return sorted(rows, key=lambda row: row["logical_path"]), summary


def inventory_huggingface(path: Path, source_id: str, manifest: dict[str, Any], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    siblings = payload.get("selected_siblings", [])
    if not isinstance(siblings, list):
        raise StagingError("GPT-NL stream manifest selected_siblings must be an array")
    rows: list[dict[str, Any]] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = item.get("rfilename")
        if not isinstance(name, str) or not name:
            continue
        size = item.get("size")
        lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
        rows.append({
            "schema_version": "1.0",
            "source_id": source_id,
            "remote_member": name,
            "logical_path": name,
            "size_bytes": int(size) if isinstance(size, int) else None,
            "remote_sha256": lfs.get("sha256") or lfs.get("oid"),
            "classification": "metadata_only_deferred",
            "reason": "Work Packet 12 does not stream GPT-NL rows",
            "selected_for_staging": False,
            "language_hint": "english_dataset_shard",
            "family_hint": family_hint(source_id, name),
            "training_allowed": False,
        })
    warnings = payload.get("security_warnings", [])
    summary = {
        "archive_type": "huggingface_stream_manifest",
        "resolved_revision": payload.get("resolved_revision") or manifest.get("resolved", {}).get("revision"),
        "selected_sibling_count": len(rows),
        "security_warning_count": len(warnings) if isinstance(warnings, list) else 0,
        "security_hold": bool(manifest.get("security", {}).get("hold")),
        "rows_downloaded": 0,
        "staging_mode": policy.get("staging_mode"),
    }
    return sorted(rows, key=lambda row: row["logical_path"]), summary


def inventory_one(
    source_id: str,
    policy: dict[str, Any],
    raw_root: Path,
    manifest_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_root / f"{source_id}.acquisition.json"
    manifest = acquisition_manifest(manifest_root, source_id)
    artifact = artifact_path(raw_root, source_id, manifest)
    mode = policy.get("inventory_mode")
    if mode == "metadata_only":
        rows, detail = inventory_huggingface(artifact, source_id, manifest, policy)
    elif tarfile.is_tarfile(artifact):
        rows, detail = _tar_inventory(artifact, source_id, policy)
    elif zipfile.is_zipfile(artifact):
        rows, detail = _zip_inventory(artifact, source_id, policy)
    else:
        raise StagingError(f"unsupported archive format: {artifact}")
    summary = {
        "schema_version": "1.0",
        "source_id": source_id,
        "baseline": "f378d0a",
        "generated_utc": utc_now(),
        "artifact_filename": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "acquisition_manifest_sha256": sha256_file(manifest_path),
        "acquisition_status": manifest.get("status"),
        "security_hold": bool(manifest.get("security", {}).get("hold")),
        "policy_status": policy.get("status"),
        "inventory": detail,
        "inventory_record_count": len(rows),
        "training_allowed": False,
        "status": "inventoried_not_staged_not_admitted",
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_staging_plan.json"))
    parser.add_argument("--acquisition-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_acquisition_plan.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/foundation/sources/raw/quarantine"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/foundation/sources/manifests"))
    parser.add_argument("--inventory-root", type=Path, default=Path("data/foundation/sources/inventory"))
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true", help="write local inventory reports")
    args = parser.parse_args(argv)

    staging, _ = load_plans(args.staging_plan, args.acquisition_plan)
    selected = selected_source_ids(staging, args.source, args.all)
    if not selected:
        print("Planned Stage A source inventories:")
        for row in staging["sources"]:
            print(f"- {row['source_id']}: {row['inventory_mode']} / {row['staging_mode']}")
        print("\nDry by default. Use --all --execute or --source ID --execute.")
        return 0
    policies = {row["source_id"]: row for row in staging["sources"]}
    if not args.execute:
        print("Dry run; no inventory files will be written.")
        for source_id in selected:
            print(f"- would inventory {source_id}")
        return 0

    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    args.inventory_root.mkdir(parents=True, exist_ok=True)
    for source_id in selected:
        print(f"Inventorying {source_id}...", flush=True)
        try:
            rows, summary = inventory_one(source_id, policies[source_id], args.raw_root, args.manifest_root)
            atomic_write_jsonl(args.inventory_root / f"{source_id}.inventory.jsonl", rows)
            atomic_write_json(args.inventory_root / f"{source_id}.inventory.summary.json", summary)
            summaries.append(summary)
            print(f"  {summary['inventory_record_count']} inventory records")
        except (StagingError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            failures.append(f"{source_id}: {exc}")
            print(f"  FAILED: {exc}", file=sys.stderr)
    atomic_write_json(args.inventory_root / "stage_a_sources_v1_inventory_summary.json", {
        "schema_version": "1.0",
        "baseline": "f378d0a",
        "generated_utc": utc_now(),
        "source_count": len(summaries),
        "sources": summaries,
        "failures": failures,
        "training_allowed": False,
        "status": "inventory_complete_not_staged_not_admitted" if not failures else "inventory_incomplete",
    })
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StagingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
