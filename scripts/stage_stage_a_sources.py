#!/usr/bin/env python3
"""Extract only inventoried allowed members into local Stage A staging.

The output is byte-preserving, source-separated, and always non-training.
No content normalization, quality admission, tokenizer work, dataset preparation,
or model training occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Callable
import zipfile

from stage_a_staging_common import (
    CHUNK_SIZE,
    StagingError,
    atomic_write_json,
    atomic_write_jsonl,
    classify_member,
    family_hint,
    language_hint,
    load_json,
    load_plans,
    normalize_archive_path,
    safe_destination,
    selected_source_ids,
    sha256_file,
    utc_now,
)


def load_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise StagingError(f"inventory row {line_number} is not an object")
            if value.get("training_allowed") is not False:
                raise StagingError(f"inventory row {line_number} grants training permission")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingError(f"cannot read inventory {path}: {exc}") from exc
    return rows


def acquisition_manifest(manifest_root: Path, source_id: str) -> tuple[Path, dict[str, Any]]:
    path = manifest_root / f"{source_id}.acquisition.json"
    value = load_json(path)
    if value.get("source_id") != source_id:
        raise StagingError(f"acquisition manifest source mismatch: {source_id}")
    if value.get("training_allowed") is not False:
        raise StagingError(f"acquisition manifest grants training permission: {source_id}")
    if bool(value.get("security", {}).get("hold")):
        raise StagingError(f"source remains on security hold: {source_id}")
    return path, value


def artifact_path(raw_root: Path, source_id: str, manifest: dict[str, Any]) -> Path:
    filename = manifest.get("artifact", {}).get("filename")
    expected = manifest.get("artifact", {}).get("sha256")
    if not isinstance(filename, str) or not isinstance(expected, str):
        raise StagingError(f"incomplete acquisition artifact metadata: {source_id}")
    path = raw_root / source_id / filename
    if not path.is_file():
        raise StagingError(f"artifact is missing: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise StagingError(f"artifact hash mismatch for {source_id}")
    return path


def copy_stream(source: BinaryIO, destination: Path, expected_size: int) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with destination.open("xb") as output:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > expected_size:
                raise StagingError(f"member emitted more bytes than declared: {destination}")
            digest.update(chunk)
            output.write(chunk)
    if size != expected_size:
        raise StagingError(f"member size mismatch: {destination}: {size} != {expected_size}")
    return digest.hexdigest(), size


def selected_rows(rows: list[dict[str, Any]], source_id: str, policy: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_logical: set[str] = set()
    total = 0
    for row in rows:
        if row.get("source_id") != source_id:
            raise StagingError(f"inventory contains another source: {row.get('source_id')}")
        if not row.get("selected_for_staging"):
            continue
        logical = row.get("logical_path")
        archive_member = row.get("archive_member")
        size = row.get("size_bytes")
        if not isinstance(logical, str) or not isinstance(archive_member, str) or not isinstance(size, int):
            raise StagingError(f"invalid selected inventory row: {row}")
        classification, _, should_select = classify_member(logical, size, policy)
        if classification != "selected" or not should_select:
            raise StagingError(f"inventory selection no longer matches policy: {logical}")
        if logical in seen_logical:
            raise StagingError(f"duplicate staged logical path: {logical}")
        seen_logical.add(logical)
        total += size
        selected.append(row)
    limit = int(policy.get("max_selected_bytes", 0))
    if total > limit:
        raise StagingError(f"selected bytes exceed limit for {source_id}: {total} > {limit}")
    return sorted(selected, key=lambda row: row["logical_path"])


def _extract_tar(artifact: Path, rows: list[dict[str, Any]], files_root: Path, source_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected = {row["archive_member"]: row for row in rows}
    remaining = set(selected)
    with tarfile.open(artifact, "r:*") as archive:
        for member in archive:
            try:
                archive_member = normalize_archive_path(member.name)
            except StagingError:
                continue
            row = selected.get(archive_member)
            if row is None:
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise StagingError(f"refusing non-regular member: {archive_member}")
            stream = archive.extractfile(member)
            if stream is None:
                raise StagingError(f"cannot read archive member: {archive_member}")
            destination = safe_destination(files_root, row["logical_path"])
            digest, size = copy_stream(stream, destination, int(row["size_bytes"]))
            records.append(_file_record(source_id, row, digest, size))
            remaining.discard(archive_member)
        if remaining:
            missing = sorted(remaining)[:20]
            raise StagingError(f"inventoried members disappeared: {missing}")
    return records


def _zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def _extract_zip(artifact: Path, rows: list[dict[str, Any]], files_root: Path, source_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(artifact) as archive:
        by_name: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            try:
                by_name[normalize_archive_path(info.filename)] = info
            except StagingError:
                continue
        for row in rows:
            archive_member = row["archive_member"]
            info = by_name.get(archive_member)
            if info is None:
                raise StagingError(f"inventoried member disappeared: {archive_member}")
            if info.is_dir() or _zip_is_symlink(info):
                raise StagingError(f"refusing non-regular zip member: {archive_member}")
            destination = safe_destination(files_root, row["logical_path"])
            with archive.open(info, "r") as stream:
                digest, size = copy_stream(stream, destination, int(row["size_bytes"]))
            records.append(_file_record(source_id, row, digest, size))
    return records


def _file_record(source_id: str, row: dict[str, Any], digest: str, size: int) -> dict[str, Any]:
    logical = row["logical_path"]
    return {
        "schema_version": "1.0",
        "source_id": source_id,
        "archive_member": row["archive_member"],
        "logical_path": logical,
        "relative_staged_path": f"files/{logical}",
        "size_bytes": size,
        "sha256": digest,
        "language_hint": row.get("language_hint") or language_hint(logical),
        "family_hint": row.get("family_hint") or family_hint(source_id, logical),
        "status": "staged_unreviewed",
        "training_allowed": False,
    }


def _atomic_commit_directory(temporary: Path, target: Path, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise StagingError(f"staged source already exists; use --force after review: {target}")
    backup = target.with_name(target.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except Exception:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def stage_metadata_only(
    source_id: str,
    target: Path,
    policy: dict[str, Any],
    acquisition_path: Path,
    acquisition: dict[str, Any],
    inventory_path: Path,
    force: bool,
) -> dict[str, Any]:
    temp = Path(tempfile.mkdtemp(prefix=f".{source_id}.stage-", dir=target.parent))
    try:
        manifest = {
            "schema_version": "1.0",
            "source_id": source_id,
            "baseline": "f378d0a",
            "generated_utc": utc_now(),
            "mode": "metadata_only",
            "artifact_sha256": acquisition["artifact"]["sha256"],
            "acquisition_manifest_sha256": sha256_file(acquisition_path),
            "inventory_sha256": sha256_file(inventory_path),
            "staged_file_count": 0,
            "staged_bytes": 0,
            "status": "metadata_only_rows_not_downloaded_not_admitted",
            "training_allowed": False,
            "notes": policy.get("notes"),
        }
        atomic_write_json(temp / "staging_manifest.json", manifest)
        atomic_write_jsonl(temp / "files.jsonl", [])
        _atomic_commit_directory(temp, target, force)
        return manifest
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def stage_one(
    source_id: str,
    policy: dict[str, Any],
    raw_root: Path,
    manifest_root: Path,
    inventory_root: Path,
    stage_root: Path,
    force: bool,
) -> dict[str, Any]:
    acquisition_path, acquisition = acquisition_manifest(manifest_root, source_id)
    artifact = artifact_path(raw_root, source_id, acquisition)
    inventory_path = inventory_root / f"{source_id}.inventory.jsonl"
    inventory_summary_path = inventory_root / f"{source_id}.inventory.summary.json"
    rows = load_inventory(inventory_path)
    inventory_summary = load_json(inventory_summary_path)
    if inventory_summary.get("training_allowed") is not False:
        raise StagingError(f"inventory summary grants training permission: {source_id}")
    if inventory_summary.get("artifact_sha256") != acquisition["artifact"]["sha256"]:
        raise StagingError(f"inventory was produced from a different artifact: {source_id}")
    target = safe_destination(stage_root, source_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if policy.get("staging_mode") == "none":
        return stage_metadata_only(source_id, target, policy, acquisition_path, acquisition, inventory_path, force)

    chosen = selected_rows(rows, source_id, policy)
    temp = Path(tempfile.mkdtemp(prefix=f".{source_id}.stage-", dir=target.parent))
    try:
        files_root = temp / "files"
        files_root.mkdir(parents=True)
        if tarfile.is_tarfile(artifact):
            records = _extract_tar(artifact, chosen, files_root, source_id)
        elif zipfile.is_zipfile(artifact):
            records = _extract_zip(artifact, chosen, files_root, source_id)
        else:
            raise StagingError(f"unsupported archive format: {artifact}")
        records.sort(key=lambda row: row["logical_path"])
        total_bytes = sum(int(row["size_bytes"]) for row in records)
        manifest = {
            "schema_version": "1.0",
            "source_id": source_id,
            "baseline": "f378d0a",
            "generated_utc": utc_now(),
            "mode": "allowed_archive_members",
            "artifact_filename": artifact.name,
            "artifact_sha256": acquisition["artifact"]["sha256"],
            "acquisition_manifest_sha256": sha256_file(acquisition_path),
            "inventory_sha256": sha256_file(inventory_path),
            "inventory_summary_sha256": sha256_file(inventory_summary_path),
            "staged_file_count": len(records),
            "staged_bytes": total_bytes,
            "source_resolved": acquisition.get("resolved", {}),
            "status": "staged_unreviewed_not_admitted",
            "training_allowed": False,
            "content_normalized": False,
            "quality_filtered": False,
        }
        atomic_write_jsonl(temp / "files.jsonl", records)
        atomic_write_json(temp / "staging_manifest.json", manifest)
        _atomic_commit_directory(temp, target, force)
        return manifest
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_staging_plan.json"))
    parser.add_argument("--acquisition-plan", type=Path, default=Path("data/foundation/manifests/stage_a_sources_v1_acquisition_plan.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/foundation/sources/raw/quarantine"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/foundation/sources/manifests"))
    parser.add_argument("--inventory-root", type=Path, default=Path("data/foundation/sources/inventory"))
    parser.add_argument("--stage-root", type=Path, default=Path("data/foundation/staged"))
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    staging, _ = load_plans(args.staging_plan, args.acquisition_plan)
    selected = selected_source_ids(staging, args.source, args.all)
    if not selected:
        print("Planned Stage A staging sources:")
        for row in staging["sources"]:
            print(f"- {row['source_id']}: {row['staging_mode']}")
        print("\nDry by default. Run inventory first, then use --all --execute.")
        return 0
    if not args.execute:
        print("Dry run; no source files will be extracted.")
        for source_id in selected:
            print(f"- would stage {source_id} -> {args.stage_root / source_id}")
        return 0

    policies = {row["source_id"]: row for row in staging["sources"]}
    args.stage_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for source_id in selected:
        print(f"Staging {source_id}...", flush=True)
        try:
            summary = stage_one(
                source_id,
                policies[source_id],
                args.raw_root,
                args.manifest_root,
                args.inventory_root,
                args.stage_root,
                args.force,
            )
            summaries.append(summary)
            print(f"  {summary['status']}: {summary['staged_file_count']} files / {summary['staged_bytes']} bytes")
        except (StagingError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            failures.append(f"{source_id}: {exc}")
            print(f"  FAILED: {exc}", file=sys.stderr)
    atomic_write_json(args.stage_root / "stage_a_sources_v1_staging_summary.json", {
        "schema_version": "1.0",
        "baseline": "f378d0a",
        "generated_utc": utc_now(),
        "source_count": len(summaries),
        "sources": summaries,
        "failures": failures,
        "status": "staging_complete_not_admitted" if not failures else "staging_incomplete",
        "training_allowed": False,
    })
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StagingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
