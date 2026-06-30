#!/usr/bin/env python3
"""Build and explicitly approve the bounded Stage A 250K model-training slice.

The source is the already reviewed tokenizer-only sample. This script never
changes that source's permissions. It derives a separate approximately 250K-token slice,
rechecks evaluation leakage, and requires a hash-bound Leo review before the
slice can be used for the Stage A 250K probe.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

try:
    from tokenizers import Tokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'tokenizers' package is required for the Stage A 250K slice.") from exc

from stage_a_staging_common import StagingError, atomic_write_json, atomic_write_jsonl, load_json, sha256_file, utc_now
from stage_a_probe_common import (
    PROBE_SCOPE,
    atomic_replace_directory,
    blocking_leakage,
    copy_text_records,
    load_jsonl,
    protected_evaluation_state,
    safe_filename,
    stable_rank,
    verify_special_tokens,
    verify_text_record,
)
from train_tokenizer import special_tokens

BASELINE = "321f0f2"
DEFAULT_PLAN = Path("data/foundation/manifests/stage_a_250k_probe_v0_1_plan.json")
DEFAULT_SOURCE_ROOT = Path("data/foundation/approved/stage_a_tokenizer_sample_v0_1")
DEFAULT_TOKENIZER = Path("data/foundation/tokenizers/stage_a_v0_1/stage_a_bpe_8000.json")
DEFAULT_CANDIDATE_ROOT = Path("data/foundation/approved/stage_a_250k_probe_v0_1_candidate")
DEFAULT_APPROVED_ROOT = Path("data/foundation/approved/stage_a_250k_probe_v0_1")


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("baseline") != BASELINE:
        raise StagingError(f"250K plan baseline must be {BASELINE}")
    if plan.get("status") != "review_only_not_admitted":
        raise StagingError("250K plan must remain review-only")
    permissions = plan.get("permissions", {})
    if any(permissions.get(key) is not False for key in (
        "training_allowed_before_review",
        "model_training_allowed_before_review",
        "tokenizer_training_allowed",
    )):
        raise StagingError("250K plan grants permission before review")
    budget = plan.get("token_budget", {})
    minimum = int(budget.get("minimum_actual_tokens", 0))
    target = int(budget.get("target_actual_tokens", 0))
    maximum = int(budget.get("maximum_actual_tokens", 0))
    if not 235000 <= minimum <= target <= maximum <= 265000:
        raise StagingError("250K token budget must stay within 235K-265K")
    groups = plan.get("groups")
    if not isinstance(groups, list) or not groups:
        raise StagingError("250K plan requires selection groups")
    group_ids = {str(row.get("group_id")) for row in groups}
    required = {"repository_sources", "wikimedia_english", "neoma_self_knowledge"}
    if group_ids != required:
        raise StagingError(f"250K groups must be exactly {sorted(required)}")
    if "frozen_stage_b" in group_ids:
        raise StagingError("Stage B instruction records cannot enter Stage A 250K-probe training")


def load_source_sample(source_root: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = source_root / "manifest.json"
    records_path = source_root / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise StagingError("approved tokenizer-only sample is missing")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "approved_for_tokenizer_comparison_only":
        raise StagingError("source sample is not approved for tokenizer comparison")
    if manifest.get("tokenizer_training_allowed") is not True:
        raise StagingError("source sample lacks tokenizer-only approval")
    if manifest.get("training_allowed") is not False or manifest.get("model_training_allowed") is not False:
        raise StagingError("source tokenizer sample unexpectedly grants model training")
    expected_manifest = str(plan["source_sample"]["expected_manifest_sha256"])
    if sha256_file(manifest_path) != expected_manifest:
        raise StagingError("approved tokenizer sample manifest hash differs from the reviewed source")
    if manifest.get("records_sha256") != sha256_file(records_path):
        raise StagingError("approved tokenizer sample records hash mismatch")
    expected_records = str(plan["source_sample"]["expected_records_sha256"])
    if sha256_file(records_path) != expected_records:
        raise StagingError("approved tokenizer sample records differ from the reviewed source")
    rows = load_jsonl(records_path)
    if len(rows) != int(manifest.get("record_count", -1)):
        raise StagingError("approved tokenizer sample record count mismatch")
    return manifest, rows


def load_tokenizer(path: Path, plan: dict[str, Any]) -> tuple[Tokenizer, dict[str, int]]:
    if not path.is_file():
        raise StagingError(f"missing provisional 8K tokenizer: {path}")
    expected_hash = str(plan["tokenizer"]["sha256"])
    if sha256_file(path) != expected_hash:
        raise StagingError("provisional tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(path))
    expected_vocab = int(plan["tokenizer"]["vocab_size"])
    if tokenizer.get_vocab_size() != expected_vocab:
        raise StagingError(
            f"provisional tokenizer vocab mismatch: expected {expected_vocab}, got {tokenizer.get_vocab_size()}"
        )
    return tokenizer, verify_special_tokens(tokenizer, special_tokens("code"))


def group_policy(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["group_id"]): row for row in plan["groups"]}


def source_targets(plan: dict[str, Any]) -> dict[str, int]:
    targets: dict[str, int] = {}
    for group in plan["groups"]:
        for source_id, target in group.get("source_targets", {}).items():
            if source_id in targets:
                raise StagingError(f"duplicate source target: {source_id}")
            targets[str(source_id)] = int(target)
    return targets


def candidate_rows(
    repo_root: Path,
    source_root: Path,
    rows: list[dict[str, Any]],
    tokenizer: Tokenizer,
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policies = group_policy(plan)
    protected_index, protected_fingerprint, protected_count = protected_evaluation_state(repo_root)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise StagingError("provisional tokenizer is missing <eos>")

    allowed_groups = set(policies)
    output: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    for row in rows:
        group_id = str(row.get("group_id"))
        if group_id not in allowed_groups:
            exclusions["group_not_allowed_for_stage_a_probe"] += 1
            continue
        if group_id == "frozen_stage_b":
            raise StagingError("frozen Stage B text reached 250K candidate selection")
        if row.get("model_training_allowed") is not False:
            raise StagingError(f"{row.get('record_id')}: source row unexpectedly grants model training")
        path, text = verify_text_record(source_root, row)
        findings = blocking_leakage(text, protected_index)
        if findings:
            exclusions["protected_evaluation_overlap"] += 1
            continue
        text_hash = sha256_file(path)
        if text_hash in seen_hashes:
            exclusions["exact_duplicate"] += 1
            continue
        seen_hashes.add(text_hash)
        token_count = len(tokenizer.encode(text).ids) + 1
        policy = policies[group_id]
        minimum = int(policy.get("minimum_record_tokens", 1))
        maximum = int(policy.get("maximum_record_tokens", 10**9))
        if token_count < minimum:
            exclusions["record_too_short"] += 1
            continue
        if token_count > maximum:
            exclusions["record_too_long"] += 1
            continue
        copied = dict(row)
        copied.update(
            actual_token_count=token_count,
            tokenizer_sha256=str(plan["tokenizer"]["sha256"]),
            probe_selection_rank=stable_rank(
                int(plan["selection_seed"]),
                group_id,
                str(row.get("source_id")),
                str(row.get("family_id")),
                str(row.get("record_id")),
            ),
            leakage_findings=[],
            source_permission="tokenizer_only_source_rederived_for_separate_review",
            status="stage_a_250k_candidate_pending_review",
            training_allowed=False,
            model_training_allowed=False,
            tokenizer_training_allowed=False,
            training_scope=None,
        )
        output.append(copied)
    return output, {
        "protected_evaluation_fingerprint": protected_fingerprint,
        "protected_evaluation_item_count": protected_count,
        "exclusions": dict(exclusions),
    }


def sorted_pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("selection_score", 0)),
            int(row.get("actual_token_count", 0)),
            int(row.get("probe_selection_rank", 0)),
            str(row.get("record_id")),
        ),
    )


def select_rows(rows: list[dict[str, Any]], plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policies = group_policy(plan)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(row)
    for source_id in by_source:
        by_source[source_id] = sorted_pool(by_source[source_id])

    targets = source_targets(plan)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    family_counts: Counter[tuple[str, str]] = Counter()
    source_tokens: Counter[str] = Counter()
    group_tokens: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    def can_add(row: dict[str, Any]) -> bool:
        record_id = str(row["record_id"])
        if record_id in selected_ids:
            return False
        group_id = str(row["group_id"])
        policy = policies[group_id]
        family_key = (group_id, str(row.get("family_id") or record_id))
        cap = int(policy.get("family_cap", 1))
        if cap > 0 and family_counts[family_key] >= cap:
            return False
        maximum = int(policy["maximum_actual_tokens"])
        if group_tokens[group_id] + int(row["actual_token_count"]) > maximum:
            return False
        total_max = int(plan["token_budget"]["maximum_actual_tokens"])
        if sum(group_tokens.values()) + int(row["actual_token_count"]) > total_max:
            return False
        return True

    def add(row: dict[str, Any]) -> None:
        record_id = str(row["record_id"])
        group_id = str(row["group_id"])
        source_id = str(row["source_id"])
        family_key = (group_id, str(row.get("family_id") or record_id))
        selected.append(row)
        selected_ids.add(record_id)
        family_counts[family_key] += 1
        tokens = int(row["actual_token_count"])
        source_tokens[source_id] += tokens
        group_tokens[group_id] += tokens

    for source_id, target in targets.items():
        for row in by_source.get(source_id, []):
            if source_tokens[source_id] >= target:
                break
            if can_add(row):
                add(row)
            else:
                skipped["source_quota_candidate_blocked"] += 1

    all_remaining = sorted_pool([row for row in rows if str(row["record_id"]) not in selected_ids])
    target_total = int(plan["token_budget"]["target_actual_tokens"])
    for row in all_remaining:
        if sum(group_tokens.values()) >= target_total:
            break
        if can_add(row):
            add(row)
        else:
            skipped["fill_candidate_blocked"] += 1

    for group_id, policy in policies.items():
        minimum = int(policy["minimum_actual_tokens"])
        if group_tokens[group_id] < minimum:
            raise StagingError(
                f"{group_id} selected {group_tokens[group_id]} tokens; minimum is {minimum}"
            )
    total = sum(group_tokens.values())
    budget = plan["token_budget"]
    if not int(budget["minimum_actual_tokens"]) <= total <= int(budget["maximum_actual_tokens"]):
        raise StagingError(f"250K slice has {total} actual tokens outside the allowed range")

    minimum_sources = int(plan["diversity"]["minimum_distinct_sources"])
    if len(source_tokens) < minimum_sources:
        raise StagingError(f"250K slice uses {len(source_tokens)} sources; minimum is {minimum_sources}")

    selected = sorted(
        selected,
        key=lambda row: (str(row["group_id"]), str(row["source_id"]), str(row["record_id"])),
    )
    return selected, {
        "actual_token_count": total,
        "tokens_by_group": dict(sorted(group_tokens.items())),
        "tokens_by_source": dict(sorted(source_tokens.items())),
        "families_by_group": dict(sorted(Counter(group for group, _ in family_counts).items())),
        "skipped": dict(skipped),
    }


def write_review_csv(path: Path, rows: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    sample_size = int(plan["review"]["stratified_review_rows"])
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["group_id"])].append(row)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < min(sample_size, len(rows)):
        progress = False
        for group_id in sorted(buckets):
            bucket = buckets[group_id]
            if bucket:
                chosen.append(bucket.pop(0))
                progress = True
                if len(chosen) >= sample_size:
                    break
        if not progress:
            break
    fields = [
        "record_id", "group_id", "source_id", "family_id", "origin_path",
        "actual_token_count", "selection_score", "decision", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in chosen:
            writer.writerow({
                **{field: row.get(field, "") for field in fields},
                "decision": "",
                "notes": "",
            })


def build_candidate(
    repo_root: Path,
    plan_path: Path,
    source_root: Path,
    tokenizer_path: Path,
    out_root: Path,
    execute: bool,
    force: bool,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    validate_plan(plan)
    source_manifest, source_rows = load_source_sample(source_root, plan)
    tokenizer, special_ids = load_tokenizer(tokenizer_path, plan)
    rows, protected = candidate_rows(repo_root, source_root, source_rows, tokenizer, plan)
    selected, selection = select_rows(rows, plan)

    summary = {
        "schema_version": "1.0",
        "manifest_id": "stage_a_250k_probe_v0_1_candidate",
        "baseline": BASELINE,
        "status": "stage_a_250k_candidate_pending_review",
        "would_write": out_root.as_posix(),
        "record_count": len(selected),
        **selection,
        **protected,
        "source_sample_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "source_sample_records_sha256": source_manifest["records_sha256"],
        "tokenizer_path": tokenizer_path.as_posix(),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": special_ids,
        "training_allowed": False,
        "model_training_allowed": False,
        "tokenizer_training_allowed": False,
        "training_scope": None,
    }
    if not execute:
        return summary

    parent = out_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=out_root.name + ".tmp-", dir=parent))
    try:
        output_rows: list[dict[str, Any]] = []
        for row in selected:
            source, text = verify_text_record(source_root, row)
            filename = safe_filename(str(row["record_id"]), str(row["stored_sha256"]))
            relative = Path("texts") / str(row["group_id"]) / str(row["source_id"]) / filename
            destination = temp / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            output = dict(row)
            output["source_text_relative_path"] = row["text_relative_path"]
            output["text_relative_path"] = relative.as_posix()
            output["stored_sha256"] = sha256_file(destination)
            output_rows.append(output)
        atomic_write_jsonl(temp / "records.jsonl", output_rows)
        write_review_csv(temp / "review_sample.csv", output_rows, plan)
        manifest = dict(summary)
        manifest.update(
            generated_utc=utc_now(),
            plan_path=plan_path.as_posix(),
            plan_sha256=sha256_file(plan_path),
            records_sha256=sha256_file(temp / "records.jsonl"),
            review_sample_sha256=sha256_file(temp / "review_sample.csv"),
            next_step=(
                "Review the stratified sample and candidate manifest. Approve only by binding "
                "a Leo review decision to the exact candidate manifest SHA-256."
            ),
        )
        atomic_write_json(temp / "manifest.json", manifest)
        atomic_replace_directory(temp, out_root, force)
        return manifest
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def approve_candidate(
    candidate_root: Path,
    decision_path: Path,
    approved_root: Path,
    plan_path: Path,
    force: bool,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    validate_plan(plan)
    manifest_path = candidate_root / "manifest.json"
    records_path = candidate_root / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise StagingError("250K candidate is missing manifest or records")
    candidate = load_json(manifest_path)
    if candidate.get("status") != "stage_a_250k_candidate_pending_review":
        raise StagingError("candidate status is not reviewable")
    if any(candidate.get(key) is not False for key in (
        "training_allowed", "model_training_allowed", "tokenizer_training_allowed"
    )):
        raise StagingError("candidate already grants permission")
    if candidate.get("records_sha256") != sha256_file(records_path):
        raise StagingError("candidate records hash mismatch")

    decision = load_json(decision_path)
    if decision.get("status") != "approved":
        raise StagingError("review decision is not approved")
    if decision.get("approved_for_stage_a_250k_probe") is not True:
        raise StagingError("review decision does not approve the 250K probe")
    if decision.get("training_scope") != PROBE_SCOPE:
        raise StagingError(f"review decision must use training_scope={PROBE_SCOPE}")
    if decision.get("tokenizer_training_allowed") is not False:
        raise StagingError("250K slice approval must not grant tokenizer training")
    if not str(decision.get("reviewer", "")).strip() or not decision.get("reviewed_utc"):
        raise StagingError("review decision requires reviewer and reviewed_utc")
    if decision.get("candidate_manifest_sha256") != sha256_file(manifest_path):
        raise StagingError("review decision is not bound to the current candidate manifest")

    rows = load_jsonl(records_path)
    excluded = {str(value) for value in decision.get("excluded_record_ids", [])}
    known = {str(row["record_id"]) for row in rows}
    unknown = sorted(excluded - known)
    if unknown:
        raise StagingError(f"review decision excludes unknown records: {unknown}")
    approved_rows = [row for row in rows if str(row["record_id"]) not in excluded]
    total_tokens = sum(int(row["actual_token_count"]) for row in approved_rows)
    budget = plan["token_budget"]
    if not int(budget["minimum_actual_tokens"]) <= total_tokens <= int(budget["maximum_actual_tokens"]):
        raise StagingError("review exclusions move the slice outside the allowed token range")

    parent = approved_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=approved_root.name + ".tmp-", dir=parent))
    try:
        copied = copy_text_records(
            approved_rows,
            candidate_root,
            temp,
            status="approved_for_stage_a_250k_probe_only",
            training_allowed=True,
            model_training_allowed=True,
            tokenizer_training_allowed=False,
            review_id=str(decision.get("review_id")),
        )
        atomic_write_jsonl(temp / "records.jsonl", copied)
        shutil.copyfile(decision_path, temp / "review_decision.json")
        manifest = {
            "schema_version": "1.0",
            "manifest_id": "stage_a_250k_probe_v0_1",
            "baseline": BASELINE,
            "generated_utc": utc_now(),
            "status": "approved_for_stage_a_250k_probe_only",
            "training_scope": PROBE_SCOPE,
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "review_decision_sha256": sha256_file(decision_path),
            "records_sha256": sha256_file(temp / "records.jsonl"),
            "record_count": len(copied),
            "actual_token_count": total_tokens,
            "tokenizer_path": candidate["tokenizer_path"],
            "tokenizer_sha256": candidate["tokenizer_sha256"],
            "tokenizer_vocab_size": candidate["tokenizer_vocab_size"],
            "protected_evaluation_fingerprint": candidate["protected_evaluation_fingerprint"],
            "protected_evaluation_item_count": candidate["protected_evaluation_item_count"],
            "excluded_record_ids": sorted(excluded),
            "training_allowed": True,
            "model_training_allowed": True,
            "tokenizer_training_allowed": False,
            "explicit_limit": (
                "Approved only for the bounded Work Packet 16 Stage A 250K probe. "
                "It is not the production Stage A corpus and does not authorize 500K/1M expansion."
            ),
        }
        atomic_write_json(temp / "manifest.json", manifest)
        atomic_replace_directory(temp, approved_root, force)
        return manifest
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--repo-root", type=Path, default=Path("."))
    build.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    build.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    build.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    build.add_argument("--out-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    build.add_argument("--execute", action="store_true")
    build.add_argument("--force", action="store_true")

    approve = subparsers.add_parser("approve")
    approve.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    approve.add_argument("--review-decision", type=Path, required=True)
    approve.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)
    approve.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    approve.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "build":
        result = build_candidate(
            args.repo_root.resolve(), args.plan.resolve(), args.source_root.resolve(),
            args.tokenizer.resolve(), args.out_root.resolve(), args.execute, args.force,
        )
    else:
        result = approve_candidate(
            args.candidate_root.resolve(), args.review_decision.resolve(),
            args.approved_root.resolve(), args.plan.resolve(), args.force,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
