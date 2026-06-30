#!/usr/bin/env python3
"""Shared helpers for the bounded Stage A 250K training probe."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from stage_a_filtering_common import build_protected_index, leakage_findings, load_protected_items
from stage_a_staging_common import StagingError, atomic_write_bytes, atomic_write_json, atomic_write_jsonl, canonical_json_bytes, sha256_file

SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
PROBE_SCOPE = "stage_a_250k_probe_v0_1_only"
BLOCKING_LEAKAGE_SEVERITIES = {"critical", "review"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise StagingError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def safe_filename(record_id: str, digest: str) -> str:
    stem = SAFE_ID_RE.sub("_", record_id).strip("._") or "record"
    return f"{stem[:100]}-{digest[:12]}.txt"


def atomic_replace_directory(source: Path, target: Path, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not force:
            raise StagingError(f"output exists; use --force after review: {target}")
        backup = Path(tempfile.mkdtemp(prefix=target.name + ".old-", dir=target.parent))
        backup.rmdir()
        os.replace(target, backup)
        try:
            os.replace(source, target)
        except Exception:
            os.replace(backup, target)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(source, target)


def stable_rank(seed: int, *parts: str) -> int:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def protected_evaluation_state(repo_root: Path) -> tuple[Any, str, int]:
    items = [item for item in load_protected_items(repo_root) if item.source_kind == "evaluation"]
    payload = [
        {
            "protected_id": item.protected_id,
            "source_kind": item.source_kind,
            "text_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
        }
        for item in sorted(items, key=lambda value: value.protected_id)
    ]
    fingerprint = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return build_protected_index(items), fingerprint, len(items)


def blocking_leakage(text: str, protected_index: Any) -> list[dict[str, Any]]:
    return [
        finding
        for finding in leakage_findings(text, protected_index)
        if str(finding.get("severity")) in BLOCKING_LEAKAGE_SEVERITIES
    ]


def verify_text_record(root: Path, row: dict[str, Any]) -> tuple[Path, str]:
    relative = row.get("text_relative_path")
    if not isinstance(relative, str) or not relative:
        raise StagingError(f"{row.get('record_id')}: missing text_relative_path")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise StagingError(f"{row.get('record_id')}: text file is missing or unsafe")
    expected = row.get("stored_sha256")
    actual = sha256_file(path)
    if expected != actual:
        raise StagingError(f"{row.get('record_id')}: text hash mismatch")
    return path, path.read_text(encoding="utf-8")


def verify_special_tokens(tokenizer: Any, expected: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for token in expected:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise StagingError(f"tokenizer is missing required special token {token}")
        if tokenizer.encode(token).ids != [token_id]:
            raise StagingError(f"special token is not atomic: {token}")
        if tokenizer.decode([token_id], skip_special_tokens=False) != token:
            raise StagingError(f"special token does not round-trip: {token}")
        result[token] = int(token_id)
    return result


def hash_uint16_file(path: Path) -> str:
    if not path.is_file():
        raise StagingError(f"missing token file: {path}")
    if path.stat().st_size % 2:
        raise StagingError(f"uint16 token file has an odd byte length: {path}")
    return sha256_file(path)


def write_jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def copy_text_records(
    rows: list[dict[str, Any]],
    source_root: Path,
    destination_root: Path,
    status: str,
    training_allowed: bool,
    model_training_allowed: bool,
    tokenizer_training_allowed: bool,
    review_id: str | None = None,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        source, _ = verify_text_record(source_root, row)
        destination = destination_root / str(row["text_relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        output = dict(row)
        output.update(
            status=status,
            training_allowed=training_allowed,
            model_training_allowed=model_training_allowed,
            tokenizer_training_allowed=tokenizer_training_allowed,
            training_scope=PROBE_SCOPE if model_training_allowed else None,
        )
        if review_id is not None:
            output["review_id"] = review_id
        output["stored_sha256"] = sha256_file(destination)
        output_rows.append(output)
    return output_rows
