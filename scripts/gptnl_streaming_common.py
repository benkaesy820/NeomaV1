#!/usr/bin/env python3
"""Shared helpers for bounded GPT-NL streaming/quarantine."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

from stage_a_filtering_common import (
    HIGH_CONFIDENCE_SECRET_PATTERNS,
    build_protected_index,
    leakage_findings,
    lexical_tokens,
    load_protected_items,
    normalize_text,
    shingle_hashes,
)
from stage_a_staging_common import StagingError, atomic_write_json, atomic_write_jsonl, utc_now


SPACE_RE = re.compile(r"\s+")
MENU_RE = re.compile(r"(?im)^(?:home|about|contact|privacy|terms|cookies|subscribe|login|sign in)\s*$")
OCR_NOISE_RE = re.compile(r"[|]{3,}|[_]{6,}|(?:\b\w\b\s*){12,}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StagingError(f"expected JSON object: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selected_manifest_rows(stream_manifest: dict[str, Any]) -> list[str]:
    rows = []
    for item in stream_manifest.get("selected_siblings", []):
        if not isinstance(item, dict):
            continue
        name = item.get("rfilename")
        if isinstance(name, str) and name.endswith(".parquet"):
            rows.append(name)
    return sorted(rows)


def shard_metadata(item: Any) -> dict[str, Any]:
    security = getattr(item, "security", None)
    lfs = getattr(item, "lfs", None)
    last_commit = getattr(item, "last_commit", None) or getattr(item, "lastCommit", None)
    return {
        "path": getattr(item, "path", None),
        "size": getattr(item, "size", None),
        "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
        "security_safe": getattr(security, "safe", None) if security else None,
        "security_status": getattr(security, "status", None) if security else None,
        "av_scan": getattr(security, "av_scan", None) if security else None,
        "pickle_import_scan": getattr(security, "pickle_import_scan", None) if security else None,
        "last_commit": getattr(last_commit, "oid", None) if last_commit else None,
        "training_allowed": False,
    }


def security_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    safe_count = 0
    blocked_count = 0
    for row in rows:
        key = str(row.get("security_status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
        if row.get("security_safe") is True and row.get("security_status") == "safe":
            safe_count += 1
        else:
            blocked_count += 1
    return {
        "status_counts": dict(sorted(counts.items())),
        "safe_count": safe_count,
        "blocked_count": blocked_count,
        "all_safe": blocked_count == 0,
    }


def normalize_row_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def row_quality(text: str, min_chars: int, max_chars: int) -> tuple[list[str], list[str]]:
    reject: list[str] = []
    review: list[str] = []
    normalized = normalize_row_text(text)
    tokens = lexical_tokens(normalized)
    if len(normalized) < min_chars or len(tokens) < 45:
        reject.append("too_short_or_low_information")
    if len(normalized) > max_chars:
        reject.append("row_too_large")
    if MENU_RE.search(normalized):
        reject.append("menu_or_navigation_noise")
    if OCR_NOISE_RE.search(normalized):
        reject.append("ocr_or_template_noise")
    if len(set(tokens)) / max(1, len(tokens)) < 0.18 and len(tokens) > 120:
        reject.append("low_vocabulary_diversity")
    for label, pattern in HIGH_CONFIDENCE_SECRET_PATTERNS:
        if pattern.search(normalized):
            reject.append(f"possible_secret:{label}")
    if "ignore previous instructions" in normalized.lower() or "system prompt" in normalized.lower():
        review.append("prompt_injection_language")
    return sorted(set(reject)), sorted(set(review))


def choose_text_from_row(row: dict[str, Any], text_columns: list[str]) -> tuple[str | None, str | None]:
    for column in text_columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value, column
    for column, value in row.items():
        if isinstance(value, str) and len(value.strip()) >= 240:
            return value, column
    return None, None


def build_row_record(
    source_id: str,
    shard_path: str,
    row_index: int,
    text: str,
    text_column: str,
    protected_index: Any,
    min_chars: int,
    max_chars: int,
) -> tuple[dict[str, Any], str]:
    normalized = normalize_row_text(text)
    rejection_reasons, review_reasons = row_quality(normalized, min_chars, max_chars)
    leakage = leakage_findings(normalized, protected_index)
    if any(item["severity"] == "critical" for item in leakage):
        rejection_reasons.append("protected_evaluation_leakage")
    if any(item["severity"] == "review" for item in leakage):
        review_reasons.append("protected_instruction_or_partial_eval_overlap")
    record_id = f"{source_id}:{shard_path}:row_{row_index}"
    record = {
        "schema_version": "1.0",
        "record_id": record_id,
        "source_id": source_id,
        "shard_path": shard_path,
        "row_index": row_index,
        "text_column": text_column,
        "char_count": len(normalized),
        "token_count_proxy": len(lexical_tokens(normalized)),
        "normalized_sha256": sha256_text(normalize_text(normalized)),
        "text_sha256": sha256_text(normalized),
        "shingle_count": len(shingle_hashes(normalized)),
        "leakage_findings": leakage,
        "rejection_reasons": sorted(set(rejection_reasons)),
        "review_reasons": sorted(set(review_reasons)),
        "training_allowed": False,
    }
    if rejection_reasons:
        record["status"] = "rejected_not_admitted"
    elif review_reasons:
        record["status"] = "human_review_required_not_admitted"
    else:
        record["status"] = "gptnl_candidate_not_admitted"
    return record, normalized


def protected_index(repo_root: Path) -> Any:
    return build_protected_index(load_protected_items(repo_root))
