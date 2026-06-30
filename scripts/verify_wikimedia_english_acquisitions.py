#!/usr/bin/env python3
"""Verify local Wikimedia English acquisition manifests and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024
EXPECTED_SOURCE_IDS = {
    "simplewiki_20260601",
    "enwikibooks_20260601",
    "enwikiversity_20260601",
}


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(
    plan_path: Path,
    manifest_root: Path,
    require_all: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    plan = load_json(plan_path)
    sources = plan.get("sources", [])
    errors: list[str] = []
    results: list[dict[str, Any]] = []
    ids = {source.get("source_id") for source in sources if isinstance(source, dict)}
    if ids != EXPECTED_SOURCE_IDS:
        errors.append(f"unexpected plan source ids: {sorted(ids)}")
    for source in sources:
        source_id = source["source_id"]
        manifest_path = manifest_root / f"{source_id}.wikimedia_acquisition.json"
        if not manifest_path.exists():
            if require_all:
                errors.append(f"missing acquisition manifest: {source_id}")
            results.append({"source_id": source_id, "status": "not_acquired"})
            continue
        manifest = load_json(manifest_path)
        if manifest.get("training_allowed") is not False:
            errors.append(f"{source_id}: training_allowed must remain false")
        if manifest.get("content_extracted") is not False:
            errors.append(f"{source_id}: acquisition must not extract content")
        if manifest.get("status") != "acquired_quarantined_pending_review":
            errors.append(f"{source_id}: unexpected status {manifest.get('status')}")
        if manifest.get("dumpstatus", {}).get("articles_multistream_status") != "done":
            errors.append(f"{source_id}: articles_multistream_status must be done")
        artifacts = manifest.get("artifacts", {})
        row = {"source_id": source_id, "status": manifest.get("status"), "training_allowed": False}
        for key in ("archive", "index"):
            artifact = artifacts.get(key, {})
            path = Path(artifact.get("path", ""))
            if not path.is_file():
                errors.append(f"{source_id}: missing {key} artifact: {path}")
                continue
            sha1 = hash_file(path, "sha1")
            sha256 = hash_file(path, "sha256")
            if sha1 != artifact.get("sha1") or sha1 != artifact.get("official_sha1"):
                errors.append(f"{source_id}: {key} SHA-1 mismatch")
            if sha256 != artifact.get("sha256"):
                errors.append(f"{source_id}: {key} SHA-256 mismatch")
            row[f"{key}_size_bytes"] = path.stat().st_size
            row[f"{key}_sha1"] = sha1
            row[f"{key}_sha256"] = sha256
        for key in ("sha1_manifest", "md5_manifest", "license_page"):
            artifact = artifacts.get(key, {})
            path = Path(artifact.get("path", ""))
            if not path.is_file():
                errors.append(f"{source_id}: missing {key}: {path}")
                continue
            if hash_file(path, "sha256") != artifact.get("sha256"):
                errors.append(f"{source_id}: {key} SHA-256 mismatch")
        results.append(row)
    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/foundation/manifests/stage_a_safe_english_alternatives_v1_candidate.json"),
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/foundation/sources/manifests"),
    )
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results, errors = verify(args.plan, args.manifest_root, args.require_all)
    report = {
        "schema_version": "1.0",
        "plan": str(args.plan),
        "manifest_root": str(args.manifest_root),
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
