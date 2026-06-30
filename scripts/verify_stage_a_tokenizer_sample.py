#!/usr/bin/env python3
"""Verify candidate or tokenizer-approved Stage A sample integrity and permissions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from stage_a_filtering_common import build_protected_index, leakage_findings, load_protected_items, record_fingerprints
from stage_a_staging_common import StagingError, sha256_file


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise StagingError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise StagingError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def verify(root: Path, repo_root: Path, require_approved: bool) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest_path = root / "manifest.json"
    records_path = root / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        return {}, ["missing manifest.json or records.jsonl"]
    manifest = load_json(manifest_path)
    rows = load_jsonl(records_path)
    approved = manifest.get("status") == "approved_for_tokenizer_comparison_only"
    if require_approved and not approved:
        errors.append("sample is not approved for tokenizer comparison")
    if manifest.get("training_allowed") is not False or manifest.get("model_training_allowed") is not False:
        errors.append("sample grants generic/model training permission")
    if approved:
        if manifest.get("tokenizer_training_allowed") is not True:
            errors.append("approved sample does not grant tokenizer-only permission")
        decision_path = root / "review_decision.json"
        if not decision_path.is_file():
            errors.append("approved sample is missing review_decision.json")
        elif manifest.get("review_decision_sha256") != sha256_file(decision_path):
            errors.append("review decision hash mismatch")
    elif manifest.get("tokenizer_training_allowed") is not False:
        errors.append("candidate sample grants tokenizer permission before review")
    if int(manifest.get("record_count", -1)) != len(rows):
        errors.append("record count mismatch")
    if manifest.get("records_sha256") != sha256_file(records_path):
        errors.append("records.jsonl hash mismatch")

    eval_items = [item for item in load_protected_items(repo_root) if item.source_kind == "evaluation"]
    eval_index = build_protected_index(eval_items)
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    actual_tokens = 0
    counts = Counter()
    expected_paths: set[str] = {"manifest.json", "records.jsonl"}
    if (root / "review_sample.csv").is_file():
        expected_paths.add("review_sample.csv")
    if (root / "review_decision.json").is_file():
        expected_paths.add("review_decision.json")

    for row in rows:
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in seen_ids:
            errors.append(f"duplicate or empty record ID: {record_id!r}")
            continue
        seen_ids.add(record_id)
        if row.get("training_allowed") is not False or row.get("model_training_allowed") is not False:
            errors.append(f"{record_id}: invalid model-training permission")
        expected_tokenizer = True if approved else False
        if row.get("tokenizer_training_allowed") is not expected_tokenizer:
            errors.append(f"{record_id}: tokenizer permission mismatch")
        relative = str(row.get("text_relative_path", ""))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            errors.append(f"{record_id}: unsafe text path")
            continue
        text_path = root / relative
        expected_paths.add(relative)
        if not text_path.is_file() or text_path.is_symlink():
            errors.append(f"{record_id}: text file missing or nonregular")
            continue
        if sha256_file(text_path) != row.get("stored_sha256"):
            errors.append(f"{record_id}: stored text hash mismatch")
            continue
        text = text_path.read_text(encoding="utf-8").rstrip("\n")
        fingerprints = record_fingerprints(text)
        for field in ("decoded_sha256", "normalized_sha256", "template_sha256", "simhash64", "token_count_proxy"):
            if row.get(field) != fingerprints[field]:
                errors.append(f"{record_id}: fingerprint mismatch for {field}")
        normalized_hash = str(row.get("normalized_sha256", ""))
        if normalized_hash in seen_hashes:
            errors.append(f"{record_id}: duplicate normalized content")
        seen_hashes.add(normalized_hash)
        findings = leakage_findings(text, eval_index)
        if any(item["severity"] in {"critical", "review"} for item in findings):
            errors.append(f"{record_id}: protected evaluation leakage")
        actual_tokens += int(row.get("token_count_proxy", 0))
        counts[str(row.get("group_id", "unknown"))] += 1

    if int(manifest.get("proxy_token_count", -1)) != actual_tokens:
        errors.append("proxy token count mismatch")
    for path in root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in expected_paths:
                errors.append(f"unexpected file: {relative}")

    summary = {
        "ok": not errors,
        "root": root.as_posix(),
        "approved": approved,
        "record_count": len(rows),
        "proxy_token_count": actual_tokens,
        "counts_by_group": dict(sorted(counts.items())),
        "training_allowed": False,
        "model_training_allowed": False,
        "tokenizer_training_allowed": approved,
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/foundation/approved/stage_a_tokenizer_sample_v0_1_candidate"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--require-approved", action="store_true")
    args = parser.parse_args()
    summary, errors = verify(args.root.resolve(), args.repo_root.resolve(), args.require_approved)
    print(json.dumps({**summary, "errors": errors}, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
