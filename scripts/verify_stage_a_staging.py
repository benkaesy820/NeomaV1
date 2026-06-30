#!/usr/bin/env python3
"""Verify local Stage A inventories and staged extraction without admitting data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage_a_staging_common import StagingError, atomic_write_json, load_json, load_plans, safe_destination, sha256_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise StagingError(f"{path}:{index}: expected object")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingError(f"cannot read {path}: {exc}") from exc
    return rows


def verify_source(source: dict[str, Any], inventory_root: Path, stage_root: Path) -> tuple[dict[str, Any], list[str]]:
    source_id = source["source_id"]
    errors: list[str] = []
    inventory_path = inventory_root / f"{source_id}.inventory.jsonl"
    inventory_summary_path = inventory_root / f"{source_id}.inventory.summary.json"
    source_root = stage_root / source_id
    staging_manifest_path = source_root / "staging_manifest.json"
    files_manifest_path = source_root / "files.jsonl"
    for path in (inventory_path, inventory_summary_path, staging_manifest_path, files_manifest_path):
        if not path.is_file():
            errors.append(f"missing required file: {path}")
    if errors:
        return {"source_id": source_id, "ok": False, "training_allowed": False}, errors

    inventory_summary = load_json(inventory_summary_path)
    staging = load_json(staging_manifest_path)
    inventory_rows = load_jsonl(inventory_path)
    file_rows = load_jsonl(files_manifest_path)
    if inventory_summary.get("training_allowed") is not False:
        errors.append("inventory summary grants training permission")
    if staging.get("training_allowed") is not False:
        errors.append("staging manifest grants training permission")
    if staging.get("source_id") != source_id:
        errors.append("staging source_id mismatch")
    if staging.get("inventory_sha256") != sha256_file(inventory_path):
        errors.append("inventory hash mismatch")
    if int(staging.get("staged_file_count", -1)) != len(file_rows):
        errors.append("staged file count mismatch")

    total_bytes = 0
    listed: set[str] = set()
    for row in file_rows:
        if row.get("training_allowed") is not False:
            errors.append(f"file grants training permission: {row.get('logical_path')}")
            continue
        relative = row.get("relative_staged_path")
        logical = row.get("logical_path")
        if not isinstance(relative, str) or not isinstance(logical, str):
            errors.append("file row missing paths")
            continue
        expected_relative = f"files/{logical}"
        if relative != expected_relative:
            errors.append(f"relative path mismatch: {relative} != {expected_relative}")
        if relative in listed:
            errors.append(f"duplicate staged path: {relative}")
        listed.add(relative)
        path = safe_destination(source_root, relative)
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or non-regular staged file: {relative}")
            continue
        actual_size = path.stat().st_size
        total_bytes += actual_size
        if actual_size != row.get("size_bytes"):
            errors.append(f"size mismatch: {relative}")
        if sha256_file(path) != row.get("sha256"):
            errors.append(f"hash mismatch: {relative}")

    if total_bytes != int(staging.get("staged_bytes", -1)):
        errors.append("staged byte total mismatch")
    actual_files = {
        path.relative_to(source_root).as_posix()
        for path in (source_root / "files").rglob("*")
        if path.is_file()
    } if (source_root / "files").exists() else set()
    if actual_files != listed:
        missing = sorted(listed - actual_files)[:20]
        extra = sorted(actual_files - listed)[:20]
        errors.append(f"staged file set mismatch; missing={missing}, extra={extra}")

    selected_inventory = sum(1 for row in inventory_rows if row.get("selected_for_staging"))
    if source.get("staging_mode") == "none":
        if file_rows or staging.get("staged_file_count") != 0:
            errors.append("metadata-only source unexpectedly staged files")
    elif selected_inventory != len(file_rows):
        errors.append(f"selected inventory/staged count mismatch: {selected_inventory} != {len(file_rows)}")

    return {
        "source_id": source_id,
        "ok": not errors,
        "inventory_records": len(inventory_rows),
        "staged_files": len(file_rows),
        "staged_bytes": total_bytes,
        "training_allowed": False,
    }, errors


def verify(staging_plan: Path, acquisition_plan: Path, inventory_root: Path, stage_root: Path, require_all: bool) -> tuple[list[dict[str, Any]], list[str]]:
    staging, _ = load_plans(staging_plan, acquisition_plan)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in staging["sources"]:
        source_root = stage_root / source["source_id"]
        if not source_root.exists() and not require_all:
            results.append({"source_id": source["source_id"], "ok": None, "status": "not_staged", "training_allowed": False})
            continue
        result, source_errors = verify_source(source, inventory_root, stage_root)
        results.append(result)
        errors.extend(f"{source['source_id']}: {error}" for error in source_errors)
    return results, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_staging_plan.json"))
    parser.add_argument("--acquisition-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_acquisition_plan.json"))
    parser.add_argument("--inventory-root", type=Path, default=Path("data/foundation/sources/inventory"))
    parser.add_argument("--stage-root", type=Path, default=Path("data/foundation/staged"))
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    results, errors = verify(
        args.staging_plan,
        args.acquisition_plan,
        args.inventory_root,
        args.stage_root,
        args.require_all,
    )
    report = {
        "schema_version": "1.0",
        "baseline": "f378d0a",
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
