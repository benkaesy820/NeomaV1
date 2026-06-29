#!/usr/bin/env python3
"""Acquire Stage A source snapshots into local quarantine without admitting content.

The command is intentionally conservative:
- dry-run unless --execute is supplied;
- HTTPS and explicit-host allowlists only;
- exact stable versions or acquisition-time commits;
- streaming SHA-256 with atomic writes;
- archive inventory and path-safety checks without extraction;
- every result remains training_allowed=false and quarantined for Leo review.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

USER_AGENT = "NeomaV1-StageA-Acquirer/1.0"
CHUNK_SIZE = 1024 * 1024
SUSPICIOUS_SUFFIXES = {
    ".7z", ".app", ".bin", ".class", ".deb", ".dll", ".dmg", ".dylib",
    ".exe", ".jar", ".msi", ".nupkg", ".pkl", ".pickle", ".pt", ".pth",
    ".rar", ".rpm", ".so", ".whl", ".zip",
}


class AcquisitionError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def validate_url(url: str, allowed_hosts: Iterable[str]) -> None:
    parsed = urlparse(url)
    allowed = {host.lower() for host in allowed_hosts}
    if parsed.scheme != "https":
        raise AcquisitionError(f"refusing non-HTTPS URL: {url}")
    if not parsed.hostname or parsed.hostname.lower() not in allowed:
        raise AcquisitionError(f"URL host is not allowlisted: {url}")
    if parsed.username or parsed.password:
        raise AcquisitionError("credentials in source URLs are forbidden")


def request_headers(url: str) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, application/octet-stream;q=0.9, */*;q=0.5"}
    if urlparse(url).hostname == "api.github.com":
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2026-03-10"
        headers["Accept"] = "application/vnd.github+json"
    if urlparse(url).hostname == "huggingface.co":
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


class SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: Iterable[str]) -> None:
        super().__init__()
        self.allowed_hosts = tuple(allowed_hosts)

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        validate_url(newurl, self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(url: str, allowed_hosts: Iterable[str], timeout: float) -> Any:
    allowed_hosts = tuple(allowed_hosts)
    validate_url(url, allowed_hosts)
    request = Request(url, headers=request_headers(url))
    opener = build_opener(SafeRedirectHandler(allowed_hosts))
    response = opener.open(request, timeout=timeout)
    final_url = response.geturl()
    validate_url(final_url, allowed_hosts)
    return response


def fetch_json(url: str, allowed_hosts: Iterable[str], timeout: float) -> Any:
    with contextlib.closing(open_url(url, allowed_hosts, timeout)) as response:
        payload = response.read()
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"invalid JSON from {url}: {exc}") from exc


def stream_download(
    url: str,
    destination: Path,
    allowed_hosts: Iterable[str],
    max_bytes: int,
    timeout: float,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    part_path.unlink(missing_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with contextlib.closing(open_url(url, allowed_hosts, timeout)) as response:
            length_header = response.headers.get("Content-Length")
            if length_header and int(length_header) > max_bytes:
                raise AcquisitionError(f"declared download size exceeds limit: {length_header} > {max_bytes}")
            with part_path.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise AcquisitionError(f"download exceeded limit: {size} > {max_bytes}")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            final_url = response.geturl()
        os.replace(part_path, destination)
        return {"sha256": digest.hexdigest(), "size_bytes": size, "final_url": final_url}
    except Exception:
        part_path.unlink(missing_ok=True)
        raise


def _safe_archive_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or path.is_absolute():
        return False
    return all(part not in {"", ".", ".."} for part in path.parts)


def _wanted_license(name: str, candidates: list[str]) -> bool:
    lowered = name.replace("\\", "/").lower()
    base = PurePosixPath(lowered).name
    for candidate in candidates:
        c = candidate.lower()
        if base == PurePosixPath(c).name or lowered.endswith("/" + c):
            return True
    return False


def scan_archive(path: Path, license_candidates: list[str], max_uncompressed_bytes: int) -> dict[str, Any]:
    members = 0
    declared_bytes = 0
    unsafe_paths: list[str] = []
    special_members: list[str] = []
    suspicious_files: list[str] = []
    license_hashes: list[dict[str, Any]] = []

    def record_common(name: str, size: int) -> None:
        nonlocal members, declared_bytes
        members += 1
        declared_bytes += max(0, size)
        if members > 500_000:
            raise AcquisitionError("archive member count exceeds 500,000")
        if declared_bytes > max_uncompressed_bytes:
            raise AcquisitionError(
                f"archive declared size exceeds limit: {declared_bytes} > {max_uncompressed_bytes}"
            )
        if not _safe_archive_name(name) and len(unsafe_paths) < 100:
            unsafe_paths.append(name)
        if PurePosixPath(name.replace("\\", "/")).suffix.lower() in SUSPICIOUS_SUFFIXES and len(suspicious_files) < 100:
            suspicious_files.append(name)

    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                record_common(member.name, member.size)
                if member.ischr() or member.isblk() or member.isfifo() or member.issym() or member.islnk():
                    if len(special_members) < 100:
                        special_members.append(member.name)
                if member.isfile() and member.size <= 4 * 1024 * 1024 and _wanted_license(member.name, license_candidates):
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        payload = extracted.read()
                        license_hashes.append({
                            "path": member.name,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "size_bytes": len(payload),
                        })
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                record_common(info.filename, info.file_size)
                if info.is_dir():
                    continue
                if info.file_size <= 4 * 1024 * 1024 and _wanted_license(info.filename, license_candidates):
                    payload = archive.read(info)
                    license_hashes.append({
                        "path": info.filename,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    })
    else:
        raise AcquisitionError(f"unsupported or invalid archive: {path.name}")

    critical = bool(unsafe_paths or special_members)
    warnings: list[str] = []
    if suspicious_files:
        warnings.append(f"archive contains {len(suspicious_files)} suspicious/binary file names; review before extraction")
    if not license_hashes:
        warnings.append("no expected license file was found in the archive")
    return {
        "member_count": members,
        "declared_uncompressed_bytes": declared_bytes,
        "unsafe_paths": unsafe_paths,
        "special_members": special_members,
        "suspicious_files_sample": suspicious_files,
        "license_file_hashes": sorted(license_hashes, key=lambda row: row["path"]),
        "critical_security_issue": critical,
        "warnings": warnings,
    }


def github_resolve_commit(source: dict[str, Any], timeout: float) -> dict[str, str]:
    allowed = source["allowed_hosts"]
    owner = source["owner"]
    repo = source["repo"]
    api = source["api_base"].rstrip("/")
    if source["kind"] == "github_head_archive":
        repo_meta = fetch_json(f"{api}/repos/{owner}/{repo}", allowed, timeout)
        branch = repo_meta.get("default_branch")
        if not isinstance(branch, str) or not branch:
            raise AcquisitionError("GitHub repository metadata did not provide a default branch")
        commit_meta = fetch_json(f"{api}/repos/{owner}/{repo}/commits/{quote(branch, safe='')}", allowed, timeout)
        sha = commit_meta.get("sha")
        if not isinstance(sha, str) or len(sha) != 40:
            raise AcquisitionError("GitHub commit response did not contain a full SHA")
        return {"resolved_ref": branch, "resolved_commit": sha}

    ref = source["ref"]
    ref_meta = fetch_json(f"{api}/repos/{owner}/{repo}/git/ref/tags/{quote(ref, safe='')}", allowed, timeout)
    obj = ref_meta.get("object", {})
    seen = set()
    while obj.get("type") == "tag":
        url = obj.get("url")
        if not isinstance(url, str) or url in seen:
            raise AcquisitionError("invalid or cyclic annotated Git tag")
        seen.add(url)
        tag_meta = fetch_json(url, allowed, timeout)
        obj = tag_meta.get("object", {})
    sha = obj.get("sha")
    if obj.get("type") != "commit" or not isinstance(sha, str) or len(sha) != 40:
        raise AcquisitionError("GitHub tag did not resolve to a full commit SHA")
    return {"resolved_ref": ref, "resolved_commit": sha}


def pypi_sdist(source: dict[str, Any], timeout: float) -> dict[str, str]:
    meta = fetch_json(source["metadata_url"], source["allowed_hosts"], timeout)
    candidates = [item for item in meta.get("urls", []) if item.get("packagetype") == "sdist"]
    if len(candidates) != 1:
        raise AcquisitionError(f"expected exactly one PyPI sdist, found {len(candidates)}")
    item = candidates[0]
    url = item.get("url")
    sha = item.get("digests", {}).get("sha256")
    filename = item.get("filename")
    if not all(isinstance(value, str) and value for value in (url, sha, filename)):
        raise AcquisitionError("PyPI sdist metadata is incomplete")
    return {"download_url": url, "published_sha256": sha, "published_filename": filename}


def _security_warning_walk(value: Any, prefix: str = "") -> list[str]:
    warnings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if any(term in lowered for term in ("unsafe", "malware", "pickle", "security")):
                if child not in (None, False, "", [], {}, "safe", "SAFE"):
                    warnings.append(f"{child_prefix}={child!r}")
            warnings.extend(_security_warning_walk(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            warnings.extend(_security_warning_walk(child, f"{prefix}[{index}]"))
    return warnings


def acquire_huggingface_manifest(source: dict[str, Any], artifact_path: Path, timeout: float) -> dict[str, Any]:
    metadata = fetch_json(source["api_url"], source["allowed_hosts"], timeout)
    revision = metadata.get("sha")
    if not isinstance(revision, str) or len(revision) < 7:
        raise AcquisitionError("Hugging Face metadata did not provide an immutable revision")
    siblings = metadata.get("siblings", [])
    selected: list[dict[str, Any]] = []
    allowed_subsets = tuple(source.get("allowed_subsets", []))
    for item in siblings:
        name = item.get("rfilename") if isinstance(item, dict) else None
        if not isinstance(name, str):
            continue
        if allowed_subsets and not any(subset in name for subset in allowed_subsets):
            continue
        selected.append(item)
    suspicious = [
        item.get("rfilename") for item in selected
        if isinstance(item.get("rfilename"), str)
        and PurePosixPath(item["rfilename"]).suffix.lower() in SUSPICIOUS_SUFFIXES
    ]
    warnings = sorted(set(_security_warning_walk(metadata)))
    if suspicious:
        warnings.append(f"selected file list contains suspicious extensions: {suspicious[:20]}")
    payload = {
        "schema_version": "1.0",
        "source_id": source["source_id"],
        "repository_id": source["repository_id"],
        "resolved_revision": revision,
        "allowed_subsets": list(allowed_subsets),
        "selected_sibling_count": len(selected),
        "selected_siblings": selected,
        "security_warnings": warnings,
        "training_allowed": False,
        "status": "stream_manifest_quarantined_pending_review",
    }
    atomic_write_json(artifact_path, payload)
    return {
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "resolved_revision": revision,
        "security_warnings": warnings,
        "selected_sibling_count": len(selected),
        "critical_security_issue": bool(warnings),
    }


def source_manifest_path(manifest_root: Path, source_id: str) -> Path:
    return manifest_root / f"{source_id}.acquisition.json"


def build_summary(plan: dict[str, Any], manifest_root: Path) -> dict[str, Any]:
    rows = []
    for source in plan["sources"]:
        path = source_manifest_path(manifest_root, source["source_id"])
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "source_id": source["source_id"],
                "status": data.get("status"),
                "artifact_sha256": data.get("artifact", {}).get("sha256"),
                "resolved_commit": data.get("resolved", {}).get("commit"),
                "resolved_revision": data.get("resolved", {}).get("revision"),
                "security_hold": data.get("security", {}).get("hold", False),
                "training_allowed": False,
            })
        else:
            rows.append({"source_id": source["source_id"], "status": "not_acquired", "training_allowed": False})
    return {
        "schema_version": "1.0",
        "manifest_id": "stage_a_sources_v1_acquisition_summary",
        "baseline": plan["baseline"],
        "generated_utc": utc_now(),
        "training_allowed": False,
        "sources": rows,
    }


def acquire_one(source: dict[str, Any], raw_root: Path, manifest_root: Path, timeout: float, force: bool) -> dict[str, Any]:
    source_dir = raw_root / source["source_id"]
    source_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = source_dir / source["artifact_filename"]
    manifest_path = source_manifest_path(manifest_root, source["source_id"])
    if artifact_path.exists() and not force:
        if not manifest_path.exists():
            raise AcquisitionError(f"artifact exists without an acquisition manifest: {artifact_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = existing.get("artifact", {}).get("sha256")
        actual = sha256_file(artifact_path)
        if expected != actual:
            raise AcquisitionError(f"existing artifact hash mismatch: {artifact_path}")
        if existing.get("training_allowed") is not False:
            raise AcquisitionError(f"existing manifest granted training permission unexpectedly: {manifest_path}")
        return existing
    if force:
        artifact_path.unlink(missing_ok=True)

    started = utc_now()
    resolved: dict[str, Any] = {}
    security: dict[str, Any] = {"hold": False, "warnings": []}
    archive: dict[str, Any] | None = None
    artifact_meta: dict[str, Any]

    try:
        kind = source["kind"]
        if kind == "huggingface_stream_manifest":
            hf = acquire_huggingface_manifest(source, artifact_path, timeout)
            resolved["revision"] = hf["resolved_revision"]
            security["warnings"] = hf["security_warnings"]
            security["hold"] = bool(hf["critical_security_issue"])
            artifact_meta = {
                "path": str(artifact_path),
                "filename": artifact_path.name,
                "sha256": hf["artifact_sha256"],
                "size_bytes": hf["artifact_size_bytes"],
                "download_url": source["api_url"],
            }
        else:
            download_url: str
            published_sha = source.get("expected_sha256")
            if kind == "pypi_sdist":
                pypi = pypi_sdist(source, timeout)
                download_url = pypi["download_url"]
                if pypi["published_filename"] != source["artifact_filename"]:
                    raise AcquisitionError("PyPI filename does not match the pinned plan")
                if published_sha and pypi["published_sha256"].lower() != published_sha.lower():
                    raise AcquisitionError("PyPI published checksum differs from the pinned plan")
                published_sha = pypi["published_sha256"]
            elif kind in {"github_tag_archive", "github_head_archive"}:
                git = github_resolve_commit(source, timeout)
                resolved["ref"] = git["resolved_ref"]
                resolved["commit"] = git["resolved_commit"]
                download_url = (
                    f"https://codeload.github.com/{source['owner']}/{source['repo']}/tar.gz/{git['resolved_commit']}"
                )
            elif kind == "http_archive":
                download_url = source["download_url"]
            else:
                raise AcquisitionError(f"unsupported acquisition kind: {kind}")

            downloaded = stream_download(
                download_url,
                artifact_path,
                source["allowed_hosts"],
                int(source["max_download_bytes"]),
                timeout,
            )
            if published_sha and downloaded["sha256"].lower() != str(published_sha).lower():
                artifact_path.unlink(missing_ok=True)
                raise AcquisitionError(
                    f"SHA-256 mismatch for {source['source_id']}: {downloaded['sha256']} != {published_sha}"
                )
            archive = scan_archive(
                artifact_path,
                list(source.get("license_paths", [])),
                max_uncompressed_bytes=max(int(source["max_download_bytes"]) * 40, 512 * 1024 * 1024),
            )
            security["warnings"] = archive["warnings"]
            security["hold"] = bool(archive["critical_security_issue"])
            artifact_meta = {
                "path": str(artifact_path),
                "filename": artifact_path.name,
                "sha256": downloaded["sha256"],
                "size_bytes": downloaded["size_bytes"],
                "download_url": download_url,
                "final_url": downloaded["final_url"],
                "published_sha256": published_sha,
            }

        result = {
            "schema_version": "1.0",
            "source_id": source["source_id"],
            "name": source["name"],
            "baseline": "8066b1f",
            "acquisition_started_utc": started,
            "acquisition_completed_utc": utc_now(),
            "kind": source["kind"],
            "expected_version": source.get("expected_version"),
            "resolved": resolved,
            "artifact": artifact_meta,
            "archive_inventory": archive,
            "license_declared": source.get("license"),
            "security": security,
            "training_allowed": False,
            "status": "acquired_security_hold" if security["hold"] else "acquired_quarantined_pending_review",
            "review": {"leo_decision": "", "notes": ""},
        }
        atomic_write_json(manifest_path, result)
        return result
    except Exception as exc:
        failure = {
            "schema_version": "1.0",
            "source_id": source["source_id"],
            "baseline": "8066b1f",
            "acquisition_started_utc": started,
            "acquisition_failed_utc": utc_now(),
            "training_allowed": False,
            "status": "acquisition_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_write_json(manifest_path, failure)
        raise


def load_plan(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("training_allowed") is not False:
        raise AcquisitionError("acquisition plan must keep training_allowed=false")
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise AcquisitionError("acquisition plan contains no sources")
    ids = [row.get("source_id") for row in sources]
    if len(ids) != len(set(ids)):
        raise AcquisitionError("duplicate source_id in acquisition plan")
    for row in sources:
        if row.get("training_allowed") is not False:
            raise AcquisitionError(f"source {row.get('source_id')} is not quarantined")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/foundation/manifests/stage_a_sources_v1_acquisition_plan.json"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/foundation/sources/raw/quarantine"),
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/foundation/sources/manifests"),
    )
    parser.add_argument("--source", action="append", default=[], help="source_id; may be repeated")
    parser.add_argument("--all", action="store_true", help="select all sources")
    parser.add_argument("--execute", action="store_true", help="perform network acquisition")
    parser.add_argument("--force", action="store_true", help="replace an existing local artifact")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = load_plan(args.plan)
    by_id = {row["source_id"]: row for row in plan["sources"]}
    if args.all and args.source:
        raise AcquisitionError("use either --all or --source, not both")
    selected_ids = list(by_id) if args.all else list(args.source)
    if not selected_ids:
        print("Planned sources:")
        for source in plan["sources"]:
            print(f"- {source['source_id']}: {source['name']} [{source['kind']}]")
        print("\nDry by default. Use --all --execute or --source ID --execute to acquire.")
        return 0
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise AcquisitionError(f"unknown source IDs: {', '.join(unknown)}")
    if not args.execute:
        print("Dry run; no network or filesystem artifacts will be created.")
        for source_id in selected_ids:
            source = by_id[source_id]
            print(f"- would acquire {source_id} -> {args.raw_root / source_id / source['artifact_filename']}")
        return 0

    failures = 0
    for source_id in selected_ids:
        print(f"Acquiring {source_id}...", flush=True)
        try:
            result = acquire_one(by_id[source_id], args.raw_root, args.manifest_root, args.timeout, args.force)
            print(f"  {result['status']}: {result['artifact']['sha256']}")
        except (AcquisitionError, HTTPError, URLError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)

    args.manifest_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.manifest_root / "stage_a_sources_v1_acquisition_summary.json",
        build_summary(plan, args.manifest_root),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcquisitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
