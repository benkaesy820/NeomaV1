#!/usr/bin/env python3
"""Verify local Stage A acquisition manifests and immutable artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(plan_path: Path, manifest_root: Path, require_all: bool) -> tuple[list[dict[str, Any]], list[str]]:
    plan = load_json(plan_path)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in plan["sources"]:
        source_id = source["source_id"]
        manifest_path = manifest_root / f"{source_id}.acquisition.json"
        if not manifest_path.exists():
            if require_all:
                errors.append(f"missing acquisition manifest: {source_id}")
            results.append({"source_id": source_id, "status": "not_acquired"})
            continue
        manifest = load_json(manifest_path)
        if manifest.get("training_allowed") is not False:
            errors.append(f"training permission must remain false: {source_id}")
        status = manifest.get("status")
        if not isinstance(status, str) or not status.startswith("acquired_"):
            errors.append(f"source is not successfully acquired: {source_id} ({status})")
            results.append({"source_id": source_id, "status": status})
            continue
        artifact = manifest.get("artifact", {})
        artifact_path = Path(artifact.get("path", ""))
        expected = artifact.get("sha256")
        if not artifact_path.is_file():
            errors.append(f"artifact is missing: {source_id}: {artifact_path}")
            results.append({"source_id": source_id, "status": status, "artifact_ok": False})
            continue
        actual = sha256_file(artifact_path)
        if actual != expected:
            errors.append(f"artifact hash mismatch: {source_id}: {actual} != {expected}")
        published = artifact.get("published_sha256")
        if published and published != actual:
            errors.append(f"published hash mismatch: {source_id}")
        results.append({
            "source_id": source_id,
            "status": status,
            "artifact_ok": actual == expected,
            "sha256": actual,
            "security_hold": bool(manifest.get("security", {}).get("hold")),
            "training_allowed": False,
        })
    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/foundation/manifests/stage_a_sources_v1_acquisition_plan.json"),
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
