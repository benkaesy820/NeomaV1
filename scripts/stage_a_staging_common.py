#!/usr/bin/env python3
"""Shared helpers for Stage A source inventory and non-training staging."""

from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable

CHUNK_SIZE = 1024 * 1024


class StagingError(RuntimeError):
    """Raised when a source cannot be inventoried or staged safely."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    atomic_write_bytes(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StagingError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_archive_path(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or path.is_absolute():
        raise StagingError(f"unsafe archive path: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise StagingError(f"unsafe archive path: {name!r}")
    return path.as_posix()


def common_archive_root(names: Iterable[str]) -> str | None:
    first_parts: set[str] = set()
    count = 0
    for name in names:
        try:
            normalized = normalize_archive_path(name)
        except StagingError:
            continue
        parts = PurePosixPath(normalized).parts
        if not parts:
            continue
        first_parts.add(parts[0])
        count += 1
        if len(first_parts) > 1:
            return None
    if count and len(first_parts) == 1:
        return next(iter(first_parts))
    return None


def strip_root(path: str, root: str | None) -> str:
    normalized = normalize_archive_path(path)
    if root:
        prefix = root.rstrip("/") + "/"
        if normalized == root:
            return ""
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return normalized


def _matches_directory_pattern(path: str, pattern: str) -> bool:
    directory_pattern = pattern.rstrip("/")
    path_obj = PurePosixPath(path)
    candidates = [path_obj.as_posix()]
    candidates.extend(parent.as_posix() for parent in path_obj.parents if parent.as_posix() != ".")
    return any(fnmatch.fnmatchcase(candidate, directory_pattern) for candidate in candidates)


def path_matches(path: str, pattern: str) -> bool:
    path = path.replace("\\", "/").lstrip("/")
    pattern = pattern.replace("\\", "/").lstrip("/")
    if pattern.endswith("/"):
        plain = pattern.rstrip("/")
        if not any(char in plain for char in "*?["):
            return path == plain or path.startswith(plain + "/")
        return _matches_directory_pattern(path, pattern)
    return fnmatch.fnmatchcase(path, pattern)


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(path_matches(path, pattern) for pattern in patterns)


def suffix_for(path: str) -> str:
    name = PurePosixPath(path).name.lower()
    if name in {"makefile", "readme", "license", "copying", "copyright"}:
        return name
    suffixes = PurePosixPath(path).suffixes
    if not suffixes:
        return ""
    compound = "".join(suffixes[-2:]).lower()
    return compound if compound == ".d.ts" else suffixes[-1].lower()


def classify_member(logical_path: str, size_bytes: int, policy: dict[str, Any]) -> tuple[str, str, bool]:
    excluded = list(policy.get("excluded_paths", []))
    allowed = list(policy.get("allowed_paths", []))
    if matches_any(logical_path, excluded):
        return "excluded_path", "matched explicit exclusion", False
    if not matches_any(logical_path, allowed):
        return "outside_allowlist", "did not match an allowed path", False
    max_file_bytes = int(policy.get("max_file_bytes", 4 * 1024 * 1024))
    if size_bytes > max_file_bytes:
        return "oversize_file", f"file exceeds {max_file_bytes} bytes", False
    suffix = suffix_for(logical_path)
    allowed_suffixes = {str(item).lower() for item in policy.get("allowed_suffixes", [])}
    allowed_names = {str(item).lower() for item in policy.get("allowed_names", [])}
    if suffix not in allowed_suffixes and PurePosixPath(logical_path).name.lower() not in allowed_names:
        return "unsupported_type", f"suffix/name {suffix or '<none>'} is not staged", False
    return "selected", "allowed path and staged file type", True


def family_hint(source_id: str, logical_path: str) -> str:
    parts = PurePosixPath(logical_path).parts
    if not parts:
        return f"{source_id}:root"
    depth = min(3, len(parts) - 1 if len(parts) > 1 else 1)
    key = "/".join(parts[:depth])
    return f"{source_id}:{key}"


def language_hint(path: str) -> str:
    suffix = suffix_for(path)
    mapping = {
        ".py": "python", ".pyi": "python", ".js": "javascript", ".mjs": "javascript",
        ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript", ".d.ts": "typescript",
        ".ps1": "powershell", ".psm1": "powershell", ".psd1": "powershell",
        ".sql": "sql", ".out": "sql_expected", ".sgml": "documentation", ".rst": "documentation",
        ".md": "documentation", ".mdx": "documentation", ".txt": "text", ".json": "configuration",
        ".toml": "configuration", ".ini": "configuration", ".yaml": "configuration", ".yml": "configuration",
        ".xml": "documentation", ".html": "documentation", ".css": "stylesheet",
    }
    return mapping.get(suffix, "other_text")


def load_plans(staging_plan_path: Path, acquisition_plan_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    staging = load_json(staging_plan_path)
    acquisition = load_json(acquisition_plan_path)
    if staging.get("training_allowed") is not False or acquisition.get("training_allowed") is not False:
        raise StagingError("all Stage A acquisition/staging plans must keep training_allowed=false")
    staging_sources = staging.get("sources")
    acquisition_sources = acquisition.get("sources")
    if not isinstance(staging_sources, list) or not isinstance(acquisition_sources, list):
        raise StagingError("staging/acquisition plan sources must be arrays")
    staging_ids = [row.get("source_id") for row in staging_sources]
    acquisition_ids = [row.get("source_id") for row in acquisition_sources]
    if len(staging_ids) != len(set(staging_ids)) or len(acquisition_ids) != len(set(acquisition_ids)):
        raise StagingError("duplicate source IDs in plans")
    if set(staging_ids) != set(acquisition_ids):
        raise StagingError("staging and acquisition plans do not contain the same source IDs")
    for row in staging_sources:
        if row.get("training_allowed") is not False:
            raise StagingError(f"staging source granted training permission: {row.get('source_id')}")
    return staging, acquisition


def selected_source_ids(plan: dict[str, Any], source_ids: list[str], select_all: bool) -> list[str]:
    ordered = [row["source_id"] for row in plan["sources"]]
    if select_all and source_ids:
        raise StagingError("use either --all or --source, not both")
    selected = ordered if select_all else list(source_ids)
    if not selected:
        return []
    unknown = sorted(set(selected) - set(ordered))
    if unknown:
        raise StagingError(f"unknown source IDs: {', '.join(unknown)}")
    return selected


def safe_destination(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve()
    destination = (root / PurePosixPath(relative_path)).resolve()
    try:
        destination.relative_to(root_resolved)
    except ValueError as exc:
        raise StagingError(f"destination escapes staging root: {relative_path}") from exc
    return destination
