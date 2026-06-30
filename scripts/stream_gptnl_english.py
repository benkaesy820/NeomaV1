#!/usr/bin/env python3
"""Inspect and, when safe, stream bounded GPT-NL English rows into quarantine.

The default execution path resolves Hugging Face shard security metadata and
blocks row streaming unless all selected shards are marked safe. No output is
training data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from gptnl_streaming_common import (
    atomic_write_bytes,
    build_row_record,
    canonical_json_bytes,
    choose_text_from_row,
    load_json,
    protected_index,
    security_summary,
    selected_manifest_rows,
    shard_metadata,
)
from stage_a_staging_common import StagingError, atomic_write_json, atomic_write_jsonl, utc_now


BASELINE = "f319c02"


def load_plan(path: Path) -> dict[str, Any]:
    plan = load_json(path)
    if plan.get("training_allowed") is not False:
        raise StagingError("GPT-NL streaming plan must keep training_allowed=false")
    if plan.get("source_id") != "gptnl_english_2026":
        raise StagingError("this tool only handles gptnl_english_2026")
    return plan


def inspect_shards(plan: dict[str, Any], stream_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise StagingError("huggingface_hub is required for GPT-NL shard metadata inspection") from exc
    repo = plan["repository_id"]
    revision = stream_manifest["resolved_revision"]
    selected = set(selected_manifest_rows(stream_manifest))
    api = HfApi()
    rows: list[dict[str, Any]] = []
    max_per_subset = int(plan.get("inspection", {}).get("max_shards_per_subset", 50))
    for subset in plan.get("selected_subsets", []):
        count = 0
        for item in api.list_repo_tree(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            path_in_repo=subset,
            recursive=False,
            expand=True,
        ):
            path = getattr(item, "path", "")
            if not isinstance(path, str) or not path.endswith(".parquet") or path not in selected:
                continue
            row = shard_metadata(item)
            row["subset"] = subset
            row["training_allowed"] = False
            rows.append(row)
            count += 1
            if count >= max_per_subset:
                break
    return rows


def _atomic_replace_directory(temp: Path, target: Path, force: bool) -> None:
    if target.exists():
        if not force:
            raise StagingError(f"output exists; review it and rerun with --force: {target}")
        backup = target.with_name(target.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        target.replace(backup)
        try:
            temp.replace(target)
        except Exception:
            backup.replace(target)
            raise
        shutil.rmtree(backup)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp.replace(target)


def write_blocked_output(
    output_root: Path,
    plan: dict[str, Any],
    stream_manifest: dict[str, Any],
    shard_rows: list[dict[str, Any]],
    force: bool,
) -> dict[str, Any]:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp-", dir=output_root.parent))
    try:
        summary = security_summary(shard_rows)
        manifest = {
            "schema_version": "1.0",
            "baseline": BASELINE,
            "source_id": plan["source_id"],
            "repository_id": plan["repository_id"],
            "resolved_revision": stream_manifest.get("resolved_revision"),
            "generated_utc": utc_now(),
            "status": "blocked_security_queued" if not summary["all_safe"] else "ready_for_bounded_streaming",
            "security": summary,
            "shards_inspected": len(shard_rows),
            "rows_downloaded": 0,
            "candidate_count": 0,
            "review_count": 0,
            "rejected_count": 0,
            "training_allowed": False,
            "notes": [
                "Default policy requires Hugging Face shard security_status=safe before row streaming.",
                "No GPT-NL text rows were downloaded by this blocked inspection run.",
            ],
        }
        atomic_write_jsonl(temp / "shard_security.jsonl", shard_rows)
        atomic_write_jsonl(temp / "candidates.jsonl", [])
        atomic_write_jsonl(temp / "review_queue.jsonl", [])
        atomic_write_jsonl(temp / "rejections.jsonl", [])
        atomic_write_json(temp / "gptnl_streaming_manifest.json", manifest)
        _atomic_replace_directory(temp, output_root, force)
        return manifest
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def stream_rows_if_allowed(
    repo_root: Path,
    output_root: Path,
    plan: dict[str, Any],
    stream_manifest: dict[str, Any],
    shard_rows: list[dict[str, Any]],
    force: bool,
    allow_queued_security: bool,
) -> dict[str, Any]:
    summary = security_summary(shard_rows)
    if not summary["all_safe"] and not allow_queued_security:
        return write_blocked_output(output_root, plan, stream_manifest, shard_rows, force)
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise StagingError("row streaming requires pyarrow plus huggingface_hub") from exc

    budget = plan["streaming_budget"]
    max_rows = int(budget["max_rows"])
    max_candidates = int(budget["max_candidate_rows"])
    min_chars = int(budget["min_chars"])
    max_chars = int(budget["max_chars"])
    text_columns = list(plan.get("text_columns", []))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp-", dir=output_root.parent))
    protected = protected_index(repo_root)
    candidates: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows_read = 0
    try:
        for shard in shard_rows:
            if len(candidates) >= max_candidates or rows_read >= max_rows:
                break
            shard_path = shard["path"]
            local = hf_hub_download(
                repo_id=plan["repository_id"],
                repo_type="dataset",
                revision=stream_manifest["resolved_revision"],
                filename=shard_path,
                local_dir=temp / "shards",
            )
            table = pq.read_table(local)
            for batch in table.to_batches(max_chunksize=512):
                for row in batch.to_pylist():
                    rows_read += 1
                    text, column = choose_text_from_row(row, text_columns)
                    if text is None or column is None:
                        continue
                    record, normalized = build_row_record(
                        plan["source_id"], shard_path, rows_read, text, column, protected, min_chars, max_chars
                    )
                    if record["normalized_sha256"] in seen:
                        record["status"] = "rejected_not_admitted"
                        record.setdefault("rejection_reasons", []).append("duplicate_row_text")
                    else:
                        seen.add(record["normalized_sha256"])
                    if record["status"] == "gptnl_candidate_not_admitted":
                        rel = Path("files") / f"row_{len(candidates)+1:08d}.txt"
                        atomic_write_bytes(temp / rel, normalized.encode("utf-8"))
                        record["filtered_relative_path"] = rel.as_posix()
                        candidates.append(record)
                    elif record["status"] == "human_review_required_not_admitted":
                        review.append(record)
                    else:
                        rejected.append(record)
                    if len(candidates) >= max_candidates or rows_read >= max_rows:
                        break
                if len(candidates) >= max_candidates or rows_read >= max_rows:
                    break
        manifest = {
            "schema_version": "1.0",
            "baseline": BASELINE,
            "source_id": plan["source_id"],
            "status": "streamed_candidates_not_admitted",
            "generated_utc": utc_now(),
            "security": summary,
            "rows_downloaded": rows_read,
            "candidate_count": len(candidates),
            "review_count": len(review),
            "rejected_count": len(rejected),
            "training_allowed": False,
        }
        atomic_write_jsonl(temp / "shard_security.jsonl", shard_rows)
        atomic_write_jsonl(temp / "candidates.jsonl", candidates)
        atomic_write_jsonl(temp / "review_queue.jsonl", review)
        atomic_write_jsonl(temp / "rejections.jsonl", rejected)
        atomic_write_json(temp / "gptnl_streaming_manifest.json", manifest)
        _atomic_replace_directory(temp, output_root, force)
        return manifest
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def dry_run(plan: dict[str, Any], stream_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "baseline": BASELINE,
        "source_id": plan["source_id"],
        "mode": "dry_run",
        "selected_subset_count": len(plan.get("selected_subsets", [])),
        "selected_shards_in_manifest": len(selected_manifest_rows(stream_manifest)),
        "requires_security_status": plan["required_security_status"],
        "training_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, default=Path("data/foundation/manifests/stage_a_gptnl_streaming_plan.json"))
    parser.add_argument("--stream-manifest", type=Path, default=Path("data/foundation/sources/raw/quarantine/gptnl_english_2026/gptnl_selected_subsets_stream_manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/foundation/gptnl_streaming/gptnl_english_2026"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stream-rows", action="store_true", help="attempt bounded row streaming after security inspection")
    parser.add_argument("--allow-queued-security", action="store_true", help="explicit override for queued Hugging Face security status")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    plan = load_plan(args.plan)
    stream_manifest = load_json(args.stream_manifest)
    if stream_manifest.get("training_allowed") is not False:
        raise StagingError("GPT-NL stream manifest must keep training_allowed=false")
    if not args.execute:
        print(json.dumps(dry_run(plan, stream_manifest), indent=2, sort_keys=True) + "\n", end="")
        return 0

    shard_rows = inspect_shards(plan, stream_manifest)
    if args.stream_rows:
        manifest = stream_rows_if_allowed(
            args.repo_root.resolve(), args.output_root, plan, stream_manifest, shard_rows, args.force, args.allow_queued_security
        )
    else:
        manifest = write_blocked_output(args.output_root, plan, stream_manifest, shard_rows, args.force)
    print(json.dumps(manifest, indent=2, sort_keys=True) + "\n", end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StagingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
