#!/usr/bin/env python3
"""Verify GPT-NL streaming/quarantine output remains non-training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage_a_filtering_common import StagingError, load_jsonl, safe_destination, sha256_file
from stage_a_staging_common import atomic_write_json, load_json


def verify(output_root: Path, require_output: bool) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest_path = output_root / "gptnl_streaming_manifest.json"
    if not manifest_path.exists():
        if require_output:
            errors.append(f"missing manifest: {manifest_path}")
        return {"source_id": "gptnl_english_2026", "status": "not_run", "training_allowed": False}, errors
    manifest = load_json(manifest_path)
    if manifest.get("training_allowed") is not False:
        errors.append("manifest grants training permission")
    required = ["shard_security.jsonl", "candidates.jsonl", "review_queue.jsonl", "rejections.jsonl"]
    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for name in required:
        path = output_root / name
        if not path.is_file():
            errors.append(f"missing required file: {path}")
            rows_by_file[name] = []
            continue
        rows = load_jsonl(path)
        rows_by_file[name] = rows
        for row in rows:
            if row.get("training_allowed") is not False:
                errors.append(f"{name} grants training permission")
    for row in rows_by_file.get("candidates.jsonl", []):
        rel = row.get("filtered_relative_path")
        if not isinstance(rel, str):
            errors.append(f"candidate missing filtered path: {row.get('record_id')}")
            continue
        path = safe_destination(output_root, rel)
        if not path.is_file() or path.is_symlink():
            errors.append(f"candidate file missing/nonregular: {rel}")
    if manifest.get("status") == "blocked_security_queued":
        if manifest.get("rows_downloaded") != 0 or rows_by_file.get("candidates.jsonl"):
            errors.append("blocked security run unexpectedly has downloaded rows/candidates")
    return {
        "source_id": manifest.get("source_id"),
        "status": manifest.get("status"),
        "candidate_count": len(rows_by_file.get("candidates.jsonl", [])),
        "review_count": len(rows_by_file.get("review_queue.jsonl", [])),
        "rejected_count": len(rows_by_file.get("rejections.jsonl", [])),
        "shard_security_count": len(rows_by_file.get("shard_security.jsonl", [])),
        "training_allowed": False,
        "ok": not errors,
    }, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/foundation/gptnl_streaming/gptnl_english_2026"))
    parser.add_argument("--require-output", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result, errors = verify(args.output_root, args.require_output)
    report = {"schema_version": "1.0", "baseline": "f319c02", "result": result, "errors": errors, "ok": not errors, "training_allowed": False}
    if args.out:
        atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True) + "\n", end="")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
