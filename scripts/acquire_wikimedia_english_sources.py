#!/usr/bin/env python3
"""Acquire exact Wikimedia English replacement dumps into local quarantine.

This performs acquisition only:
- verifies each 20260601 articles multistream job is ``done``;
- downloads the XML archive, multistream index, checksum manifests, dumpstatus,
  and legal page as inert bytes;
- verifies official SHA-1 for the archive and index;
- records local SHA-256 and provenance;
- keeps every output ``training_allowed=false``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

USER_AGENT = "NeomaV1-Wikimedia-English-Acquirer/1.0"
CHUNK_SIZE = 1024 * 1024
EXPECTED_SOURCE_IDS = {
    "simplewiki_20260601",
    "enwikibooks_20260601",
    "enwikiversity_20260601",
}


class AcquisitionError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


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


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_url(url: str, allowed_hosts: Iterable[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise AcquisitionError(f"refusing non-HTTPS URL: {url}")
    if parsed.username or parsed.password:
        raise AcquisitionError(f"refusing URL with credentials: {url}")
    if not parsed.hostname or parsed.hostname.lower() not in {host.lower() for host in allowed_hosts}:
        raise AcquisitionError(f"URL host is not allowlisted: {url}")


class SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: Iterable[str]) -> None:
        super().__init__()
        self.allowed_hosts = tuple(allowed_hosts)

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        validate_url(newurl, self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(url: str, allowed_hosts: Iterable[str], timeout: float) -> Any:
    validate_url(url, allowed_hosts)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    response = build_opener(SafeRedirectHandler(allowed_hosts)).open(request, timeout=timeout)
    validate_url(response.geturl(), allowed_hosts)
    return response


def download_bytes(url: str, allowed_hosts: Iterable[str], timeout: float) -> bytes:
    with contextlib.closing(open_url(url, allowed_hosts, timeout)) as response:
        return response.read()


def stream_download(
    url: str,
    destination: Path,
    allowed_hosts: Iterable[str],
    timeout: float,
    max_bytes: int,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    part_path.unlink(missing_ok=True)
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    size = 0
    final_url = url
    try:
        with contextlib.closing(open_url(url, allowed_hosts, timeout)) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise AcquisitionError(f"declared download exceeds limit: {length} > {max_bytes}")
            final_url = response.geturl()
            with part_path.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise AcquisitionError(f"download exceeds limit: {size} > {max_bytes}")
                    sha1.update(chunk)
                    sha256.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        os.replace(part_path, destination)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "filename": destination.name,
        "size_bytes": size,
        "sha1": sha1.hexdigest(),
        "sha256": sha256.hexdigest(),
        "download_url": url,
        "final_url": final_url,
    }


def parse_sha1sums(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 40:
            continue
        rows[parts[1].strip()] = parts[0].lower()
    return rows


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("training_allowed") is not False:
        raise AcquisitionError("safe-English plan must keep training_allowed=false")
    controls = plan.get("common_controls", {})
    if controls.get("required_articles_multistream_status") != "done":
        raise AcquisitionError("safe-English plan must require articles multistream status done")
    if controls.get("verify_official_sha1") is not True:
        raise AcquisitionError("safe-English plan must require official SHA-1 verification")
    sources = plan.get("sources")
    if not isinstance(sources, list):
        raise AcquisitionError("safe-English plan sources must be a list")
    ids = {source.get("source_id") for source in sources if isinstance(source, dict)}
    if ids != EXPECTED_SOURCE_IDS:
        raise AcquisitionError(f"unexpected Wikimedia source ids: {sorted(ids)}")
    for source in sources:
        if source.get("training_allowed") is not False:
            raise AcquisitionError(f"{source.get('source_id')}: training_allowed must be false")
        if source.get("snapshot") != "20260601":
            raise AcquisitionError(f"{source.get('source_id')}: snapshot must be 20260601")
    return plan


def _local_artifact(path: Path, expected_sha1: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    actual_sha1 = hash_file(path, "sha1")
    if actual_sha1 != expected_sha1:
        raise AcquisitionError(f"existing artifact SHA-1 mismatch: {path}")
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha1": actual_sha1,
        "sha256": hash_file(path, "sha256"),
        "reused_existing": True,
    }


def _write_text_artifact(path: Path, payload: bytes) -> dict[str, Any]:
    atomic_write_bytes(path, payload)
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def acquire_one(
    source: dict[str, Any],
    raw_root: Path,
    manifest_root: Path,
    timeout: float,
    max_bytes: int,
    force: bool,
) -> dict[str, Any]:
    source_id = source["source_id"]
    host = "dumps.wikimedia.org"
    allowed_hosts = [host]
    source_dir = raw_root / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{source_id}.wikimedia_acquisition.json"
    started = utc_now()

    dumpstatus_url = source["upstream_index"].rstrip("/") + "/dumpstatus.json"
    dumpstatus_payload = download_bytes(dumpstatus_url, allowed_hosts, timeout)
    dumpstatus = json.loads(dumpstatus_payload.decode("utf-8"))
    job = dumpstatus.get("jobs", {}).get("articlesmultistreamdump", {})
    if job.get("status") != "done":
        raise AcquisitionError(f"{source_id}: articlesmultistreamdump status is {job.get('status')!r}, expected 'done'")

    sha1_payload = download_bytes(source["sha1_url"], allowed_hosts, timeout)
    sha1_rows = parse_sha1sums(sha1_payload.decode("utf-8"))
    archive_name = Path(urlparse(source["archive_url"]).path).name
    index_name = Path(urlparse(source["index_url"]).path).name
    missing = [name for name in (archive_name, index_name) if name not in sha1_rows]
    if missing:
        raise AcquisitionError(f"{source_id}: official SHA-1 manifest missing {missing}")

    md5_payload = download_bytes(source["md5_url"], allowed_hosts, timeout)
    legal_payload = download_bytes(source["license_info_url"], allowed_hosts, timeout)
    _write_text_artifact(source_dir / Path(urlparse(dumpstatus_url).path).name, dumpstatus_payload)
    sha1_artifact = _write_text_artifact(source_dir / Path(urlparse(source["sha1_url"]).path).name, sha1_payload)
    md5_artifact = _write_text_artifact(source_dir / Path(urlparse(source["md5_url"]).path).name, md5_payload)
    legal_artifact = _write_text_artifact(source_dir / "wikimedia-dump-legal.html", legal_payload)

    archive_path = source_dir / archive_name
    index_path = source_dir / index_name
    if force:
        archive_path.unlink(missing_ok=True)
        index_path.unlink(missing_ok=True)

    archive = _local_artifact(archive_path, sha1_rows[archive_name])
    if archive is None:
        archive = stream_download(source["archive_url"], archive_path, allowed_hosts, timeout, max_bytes)
        if archive["sha1"] != sha1_rows[archive_name]:
            archive_path.unlink(missing_ok=True)
            raise AcquisitionError(f"{source_id}: archive SHA-1 mismatch after download")

    index = _local_artifact(index_path, sha1_rows[index_name])
    if index is None:
        index = stream_download(source["index_url"], index_path, allowed_hosts, timeout, max_bytes)
        if index["sha1"] != sha1_rows[index_name]:
            index_path.unlink(missing_ok=True)
            raise AcquisitionError(f"{source_id}: index SHA-1 mismatch after download")

    archive["official_sha1"] = sha1_rows[archive_name]
    index["official_sha1"] = sha1_rows[index_name]
    result = {
        "schema_version": "1.0",
        "source_id": source_id,
        "name": source["name"],
        "snapshot": source["snapshot"],
        "baseline": "82b994f",
        "acquisition_started_utc": started,
        "acquisition_completed_utc": utc_now(),
        "dumpstatus": {
            "url": dumpstatus_url,
            "artifact_path": str(source_dir / "dumpstatus.json"),
            "articles_multistream_status": job.get("status"),
            "articles_multistream_updated": job.get("updated"),
            "files": sorted(job.get("files", {}).keys()),
        },
        "artifacts": {
            "archive": archive,
            "index": index,
            "sha1_manifest": sha1_artifact,
            "md5_manifest": md5_artifact,
            "license_page": legal_artifact,
        },
        "official_sha1_verified": True,
        "local_sha256_recorded": True,
        "license_declared": source.get("license"),
        "license_status": source.get("license_status"),
        "status": "acquired_quarantined_pending_review",
        "training_allowed": False,
        "content_extracted": False,
        "review": {"leo_decision": "", "notes": ""},
    }
    atomic_write_json(manifest_path, result)
    return result


def build_summary(plan: dict[str, Any], manifest_root: Path) -> dict[str, Any]:
    sources = []
    for source in plan["sources"]:
        path = manifest_root / f"{source['source_id']}.wikimedia_acquisition.json"
        if not path.exists():
            sources.append({"source_id": source["source_id"], "status": "not_acquired", "training_allowed": False})
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        archive = manifest.get("artifacts", {}).get("archive", {})
        index = manifest.get("artifacts", {}).get("index", {})
        sources.append({
            "source_id": manifest.get("source_id"),
            "snapshot": manifest.get("snapshot"),
            "status": manifest.get("status"),
            "articles_multistream_status": manifest.get("dumpstatus", {}).get("articles_multistream_status"),
            "archive_size_bytes": archive.get("size_bytes"),
            "archive_sha1": archive.get("sha1"),
            "archive_sha256": archive.get("sha256"),
            "index_size_bytes": index.get("size_bytes"),
            "index_sha1": index.get("sha1"),
            "index_sha256": index.get("sha256"),
            "training_allowed": False,
        })
    return {
        "schema_version": "1.0",
        "review_id": "stage_a_safe_english_acquisition_results_v1",
        "baseline": "82b994f",
        "generated_utc": utc_now(),
        "scope": "Wikimedia safe-English source acquisition and immutable hashing only",
        "raw_artifacts_committed": False,
        "generated_local_manifests_committed": False,
        "training_allowed": False,
        "sources": sources,
        "all_sources_acquired": all(row["status"] == "acquired_quarantined_pending_review" for row in sources),
        "review_decision": {
            "status": "acquisition_complete_quarantined_pending_filtering",
            "reason": "Exact 20260601 Wikimedia XML dumps and multistream indexes were acquired, official SHA-1 values verified, local SHA-256 values recorded, and no content was extracted or admitted.",
        },
        "next_step": "Parse and filter bounded review samples from the three Wikimedia dumps; keep every page training_allowed=false until a later admission packet.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/foundation/manifests/stage_a_safe_english_alternatives_v1_candidate.json"),
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
    parser.add_argument(
        "--review-out",
        type=Path,
        default=Path("data/reviews/stage_a_safe_english_acquisition_results_v1.json"),
    )
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-bytes-per-file", type=int, default=2_000_000_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = load_plan(args.plan)
    by_id = {source["source_id"]: source for source in plan["sources"]}
    if args.all and args.source:
        raise AcquisitionError("use either --all or --source, not both")
    selected = list(by_id) if args.all else list(args.source)
    if not selected:
        print("Planned Wikimedia English sources:")
        for source_id in by_id:
            print(f"- {source_id}")
        print("\nDry by default. Use --all --execute to acquire.")
        return 0
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise AcquisitionError(f"unknown source ids: {unknown}")
    if not args.execute:
        print("Dry run; no network artifacts will be created.")
        for source_id in selected:
            print(f"- would acquire {source_id}")
        return 0

    args.manifest_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for source_id in selected:
        print(f"Acquiring {source_id}...", flush=True)
        try:
            result = acquire_one(
                by_id[source_id],
                args.raw_root,
                args.manifest_root,
                args.timeout,
                args.max_bytes_per_file,
                args.force,
            )
            archive = result["artifacts"]["archive"]
            index = result["artifacts"]["index"]
            print(f"  archive {archive['size_bytes']} bytes sha256={archive['sha256']}", flush=True)
            print(f"  index   {index['size_bytes']} bytes sha256={index['sha256']}", flush=True)
        except (AcquisitionError, HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"  FAILED: {exc}", flush=True)

    summary = build_summary(plan, args.manifest_root)
    atomic_write_json(args.manifest_root / "stage_a_safe_english_acquisition_summary.json", summary)
    atomic_write_json(args.review_out, summary)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcquisitionError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
