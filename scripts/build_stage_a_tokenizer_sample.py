#!/usr/bin/env python3
"""Build a deterministic Stage A representative sample for tokenizer comparison.

The command never grants model-training permission. In build mode it writes a
candidate sample with all permissions false. In approval mode it requires a
human review decision tied to the exact candidate-manifest SHA-256 and grants
only ``tokenizer_training_allowed=true``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable

from stage_a_filtering_common import (
    build_protected_index,
    choose_duplicate_status,
    leakage_findings,
    lexical_tokens,
    load_protected_items,
    record_fingerprints,
    simhash_bands,
)
from stage_a_staging_common import StagingError, atomic_write_json, atomic_write_jsonl, sha256_file, utc_now

BASELINE = "35e6d17"
DEFAULT_PLAN = Path("data/foundation/manifests/stage_a_tokenizer_sample_v0_1_plan.json")
DEFAULT_FILTER_ROOT = Path("data/foundation/filtered")
DEFAULT_CANDIDATE_ROOT = Path("data/foundation/approved/stage_a_tokenizer_sample_v0_1_candidate")
DEFAULT_APPROVED_ROOT = Path("data/foundation/approved/stage_a_tokenizer_sample_v0_1")
INSTRUCTION_START_RE = re.compile(r"<instruction(?:\s[^>]*)?>", re.I)
CODE_EXTENSIONS = {".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".ps1", ".psm1", ".psd1", ".sql"}
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".sgml"}


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
                raise StagingError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("baseline") != BASELINE:
        raise StagingError(f"plan baseline must be {BASELINE}")
    if plan.get("training_allowed") is not False:
        raise StagingError("plan must not grant model-training permission")
    budget = plan.get("sample_budget", {})
    minimum = int(budget.get("minimum_proxy_tokens_after_review", 0))
    target = int(budget.get("target_proxy_tokens", 0))
    maximum = int(budget.get("maximum_proxy_tokens_after_review", 0))
    if not 0 < minimum <= target <= maximum:
        raise StagingError("invalid sample budget")
    group_ids = [row.get("group_id") for row in plan.get("selection_groups", [])]
    expected = {"frozen_stage_b", "repository_sources", "wikimedia_english", "neoma_self_knowledge"}
    if set(group_ids) != expected or len(group_ids) != len(expected):
        raise StagingError("selection groups do not match the required four groups")


def group_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["group_id"]): row for row in plan["selection_groups"]}


def split_instruction_file(path: Path, repo_root: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    starts = [match.start() for match in INSTRUCTION_START_RE.finditer(text)]
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        relative = path.relative_to(repo_root).as_posix()
        rows.append({
            "record_id": f"stage_b:{relative}:{index + 1:04d}",
            "source_id": "frozen_stage_b_instruction_corpus_v0_1",
            "group_id": "frozen_stage_b",
            "family_id": f"stage_b:{relative}",
            "component": "reviewed_instruction_language",
            "origin_path": relative,
            "text": chunk,
            "selection_score": 1000,
            "selection_reason": "all frozen instruction records are included for tokenizer vocabulary",
        })
    return rows


def evaluation_index(repo_root: Path) -> Any:
    protected = [item for item in load_protected_items(repo_root) if item.source_kind == "evaluation"]
    return build_protected_index(protected)


def finalize_record(row: dict[str, Any], eval_index: Any) -> dict[str, Any]:
    text = str(row["text"]).strip()
    findings = leakage_findings(text, eval_index)
    blocking = [item for item in findings if item["severity"] in {"critical", "review"}]
    if blocking:
        raise StagingError(f"{row['record_id']}: protected evaluation overlap blocks selection")
    fingerprints = record_fingerprints(text)
    return {
        "schema_version": "1.0",
        **{key: value for key, value in row.items() if key != "text"},
        "text": text,
        "leakage_findings": findings,
        "training_allowed": False,
        "model_training_allowed": False,
        "tokenizer_training_allowed": False,
        "status": "tokenizer_sample_candidate_pending_review",
        **fingerprints,
    }


def load_stage_b(
    repo_root: Path,
    policy: dict[str, Any],
    eval_index: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for relative in policy["input_paths"]:
        path = repo_root / relative
        if not path.is_file():
            raise StagingError(f"missing frozen Stage B file: {relative}")
        rows.extend(split_instruction_file(path, repo_root))
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        findings = leakage_findings(str(row["text"]), eval_index)
        blocking = [item for item in findings if item["severity"] in {"critical", "review"}]
        if blocking:
            excluded.append({
                "record_id": row["record_id"],
                "reason": "protected_evaluation_overlap_excluded_from_tokenizer_training",
                "leakage_findings": blocking,
            })
            continue
        selected.append(finalize_record(row, eval_index))
    return selected, excluded, len(rows)


def repository_score(row: dict[str, Any], family_size: int) -> int:
    path = str(row.get("logical_path", "")).lower()
    suffix = Path(path).suffix.lower()
    tokens = int(row.get("token_count_proxy", 0))
    score = 50
    if family_size >= 2:
        score += 12
    if any(part in path for part in ("/test", "tests/", "testing/", "tst/", "regress/")):
        score += 14
    if any(part in path for part in ("/doc", "docs/", "documentation/", "tutorial", "handbook", "readme")):
        score += 12
    if suffix in CODE_EXTENSIONS:
        score += 10
    elif suffix in DOC_EXTENSIONS:
        score += 8
    if 80 <= tokens <= 900:
        score += 12
    elif tokens <= 1500:
        score += 6
    if any(marker in path for marker in ("generated", "baseline", "snapshot", "fixture")):
        score -= 20
    return score


def load_repository_pool(filter_root: Path, source_id: str, policy: dict[str, Any], eval_index: Any) -> list[dict[str, Any]]:
    source_root = filter_root / source_id
    candidates_path = source_root / "candidates.jsonl"
    if not candidates_path.is_file():
        raise StagingError(f"missing filtered repository candidates: {source_id}")
    raw_rows = load_jsonl(candidates_path)
    family_sizes = Counter(str(row.get("family_id", "")) for row in raw_rows)
    minimum = int(policy.get("minimum_record_tokens", 0))
    maximum = int(policy.get("maximum_record_tokens", 10**9))
    output: list[dict[str, Any]] = []
    for source_row in raw_rows:
        if source_row.get("status") != "filtered_candidate_not_admitted":
            continue
        if source_row.get("training_allowed") is not False:
            raise StagingError(f"{source_row.get('record_id')}: source candidate unexpectedly grants training")
        if source_row.get("review_reasons") or source_row.get("rejection_reasons"):
            continue
        relative = source_row.get("filtered_relative_path")
        if not isinstance(relative, str):
            raise StagingError(f"{source_row.get('record_id')}: missing filtered_relative_path")
        text_path = source_root / relative
        if not text_path.is_file() or text_path.is_symlink():
            raise StagingError(f"{source_row.get('record_id')}: filtered text missing")
        if source_row.get("filtered_sha256") and sha256_file(text_path) != source_row["filtered_sha256"]:
            raise StagingError(f"{source_row.get('record_id')}: filtered text hash mismatch")
        text = text_path.read_text(encoding="utf-8")
        tokens = len(lexical_tokens(text))
        if tokens < minimum or tokens > maximum:
            continue
        row = {
            "record_id": str(source_row["record_id"]),
            "source_id": source_id,
            "group_id": "repository_sources",
            "family_id": str(source_row.get("family_id") or source_row["record_id"]),
            "component": "repository_code_docs_tests",
            "origin_path": str(source_row.get("logical_path", relative)),
            "source_candidate_sha256": source_row.get("normalized_sha256"),
            "text": text,
            "selection_score": repository_score(source_row, family_sizes[str(source_row.get("family_id", ""))]),
            "selection_reason": "clean repository candidate ranked for code, documentation, tests, compactness, and family relationships",
        }
        output.append(finalize_record(row, eval_index))
    return output


def load_wikimedia_pool(filter_root: Path, source_id: str, policy: dict[str, Any], eval_index: Any) -> list[dict[str, Any]]:
    path = filter_root / "wikimedia_english_20260601" / source_id / "candidates.jsonl"
    if not path.is_file():
        raise StagingError(f"missing filtered Wikimedia candidates: {source_id}")
    minimum_score = int(policy.get("minimum_quality_score", 0))
    output: list[dict[str, Any]] = []
    for source_row in load_jsonl(path):
        if source_row.get("status") != "filtered_candidate_not_admitted":
            continue
        if source_row.get("training_allowed") is not False:
            raise StagingError(f"{source_row.get('record_id')}: Wikimedia candidate grants training")
        if source_row.get("review_reasons") or source_row.get("rejection_reasons"):
            continue
        score = int(source_row.get("quality_score", 0))
        if score < minimum_score:
            continue
        row = {
            "record_id": str(source_row["record_id"]),
            "source_id": source_id,
            "group_id": "wikimedia_english",
            "family_id": str(source_row["family_id"]),
            "component": str(source_row.get("component", "english")),
            "origin_path": f"page:{source_row.get('page_id')}:revision:{source_row.get('revision_id')}:segment:{source_row.get('segment_index')}",
            "title": source_row.get("title"),
            "page_id": source_row.get("page_id"),
            "revision_id": source_row.get("revision_id"),
            "revision_timestamp": source_row.get("revision_timestamp"),
            "source_candidate_sha256": source_row.get("normalized_sha256"),
            "text": str(source_row["text"]),
            "selection_score": score,
            "selection_reason": "clean Wikimedia candidate ranked by explanation quality and selected with one-record family diversity",
        }
        output.append(finalize_record(row, eval_index))
    return output


def load_self_knowledge(repo_root: Path, policy: dict[str, Any], eval_index: Any) -> list[dict[str, Any]]:
    path = repo_root / str(policy["input_path"])
    rows_by_id = {str(row["id"]): row for row in load_jsonl(path)}
    admitted_ids = [str(value) for value in policy["admitted_ids"]]
    missing = [record_id for record_id in admitted_ids if record_id not in rows_by_id]
    if missing:
        raise StagingError(f"self-knowledge allowlist IDs missing: {missing}")
    output: list[dict[str, Any]] = []
    for record_id in admitted_ids:
        source_row = rows_by_id[record_id]
        if source_row.get("training_allowed") is not False:
            raise StagingError(f"{record_id}: source self-knowledge row grants training")
        row = {
            "record_id": record_id,
            "source_id": "neoma_model_card_v0_1",
            "group_id": "neoma_self_knowledge",
            "family_id": str(source_row.get("family_id", "neoma_self_knowledge_v0_1")),
            "component": "stable_self_knowledge",
            "origin_path": str(policy["input_path"]),
            "source_candidate_sha256": source_row.get("content_sha256"),
            "text": str(source_row["text"]),
            "selection_score": 900,
            "selection_reason": "explicit stable self-knowledge allowlist; transient and provisional facts are deferred",
            "factual_basis": source_row.get("factual_basis", []),
        }
        output.append(finalize_record(row, eval_index))
    return output


class DedupIndex:
    def __init__(self) -> None:
        self.exact: dict[str, dict[str, Any]] = {}
        self.templates: dict[str, dict[str, Any]] = {}
        self.bands: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    def duplicate_reason(self, row: dict[str, Any]) -> str | None:
        prior = self.exact.get(str(row["normalized_sha256"])) or self.templates.get(str(row["template_sha256"]))
        if prior is not None:
            action, reason = choose_duplicate_status(row, prior)
            return reason if action != "keep" else None
        possible: dict[str, dict[str, Any]] = {}
        for band in simhash_bands(int(str(row["simhash64"]), 16)):
            for candidate in self.bands.get(band, []):
                possible[str(candidate["record_id"])] = candidate
        for candidate in sorted(possible.values(), key=lambda item: str(item["record_id"])):
            action, reason = choose_duplicate_status(row, candidate)
            if action != "keep":
                return reason
        return None

    def add(self, row: dict[str, Any]) -> None:
        self.exact.setdefault(str(row["normalized_sha256"]), row)
        self.templates.setdefault(str(row["template_sha256"]), row)
        for band in simhash_bands(int(str(row["simhash64"]), 16)):
            self.bands[band].append(row)


def select_ranked(pool: Iterable[dict[str, Any]], quota: int, family_cap: int, dedup: DedupIndex) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    used = 0
    ordered = sorted(pool, key=lambda row: (-int(row.get("selection_score", 0)), int(row["token_count_proxy"]), str(row["record_id"])))
    for row in ordered:
        family = str(row["family_id"])
        if family_cap > 0 and family_counts[family] >= family_cap:
            reasons["family_cap"] += 1
            continue
        tokens = int(row["token_count_proxy"])
        if used + tokens > quota:
            reasons["source_quota"] += 1
            continue
        duplicate = dedup.duplicate_reason(row)
        if duplicate:
            reasons["cross_group_duplicate"] += 1
            continue
        selected.append(row)
        dedup.add(row)
        family_counts[family] += 1
        used += tokens
    return selected, reasons


def safe_filename(record_id: str, content_sha256: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", record_id).strip("-._")[:80] or "record"
    return f"{slug}-{content_sha256[:12]}.txt"


def atomic_replace_directory(temp: Path, target: Path, force: bool) -> None:
    if target.exists():
        if not force:
            raise StagingError(f"output exists; rerun with --force after review: {target}")
        backup = target.with_name(target.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(target, backup)
        try:
            os.replace(temp, target)
        except Exception:
            os.replace(backup, target)
            raise
        shutil.rmtree(backup)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, target)


def write_review_csv(path: Path, rows: list[dict[str, Any]], per_source: int = 8) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(row)
    sample: list[dict[str, Any]] = []
    for source_id, source_rows in sorted(by_source.items()):
        ordered = sorted(source_rows, key=lambda row: (int(row["token_count_proxy"]), str(row["record_id"])))
        if source_id == "neoma_model_card_v0_1":
            sample.extend(ordered)
            continue
        if len(ordered) <= per_source:
            sample.extend(ordered)
            continue
        positions = sorted({round(index * (len(ordered) - 1) / (per_source - 1)) for index in range(per_source)})
        sample.extend(ordered[position] for position in positions)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_id", "source_id", "group_id", "family_id", "token_count_proxy", "content_sha256", "decision", "notes"])
        for row in sample:
            writer.writerow([row["record_id"], row["source_id"], row["group_id"], row["family_id"], row["token_count_proxy"], row["decoded_sha256"], "pending", ""])


def build_candidate(repo_root: Path, plan_path: Path, filter_root: Path, out_root: Path, execute: bool, force: bool) -> dict[str, Any]:
    plan = load_json(plan_path)
    validate_plan(plan)
    groups = group_map(plan)
    if not execute:
        return {
            "ok": True,
            "mode": "dry_run",
            "baseline": BASELINE,
            "candidate_output": out_root.as_posix(),
            "training_allowed": False,
            "model_training_allowed": False,
            "tokenizer_training_allowed": False,
        }

    eval_index = evaluation_index(repo_root)
    dedup = DedupIndex()
    selected: list[dict[str, Any]] = []
    selection_notes: dict[str, Any] = {}

    stage_b, stage_b_excluded, stage_b_total = load_stage_b(
        repo_root,
        groups["frozen_stage_b"],
        eval_index,
    )
    expected_records = int(plan["protected_data_policy"]["stage_b_expected_record_count"])
    if stage_b_total != expected_records:
        raise StagingError(f"expected {expected_records} frozen Stage B records, found {stage_b_total}")
    minimum_records = int(plan["protected_data_policy"].get("stage_b_minimum_tokenizer_records", 1))
    if len(stage_b) < minimum_records:
        raise StagingError(
            f"only {len(stage_b)} Stage B records remain after evaluation-leakage exclusion; "
            f"minimum is {minimum_records}"
        )
    # Keep every non-leaking frozen record. The complete 331-record corpus is
    # still measured during benchmarking, but protected overlaps never train
    # the tokenizer.
    for row in sorted(stage_b, key=lambda item: str(item["record_id"])):
        selected.append(row)
        dedup.add(row)
    selection_notes["frozen_stage_b"] = {
        "total_frozen_records": stage_b_total,
        "selected_records": len(stage_b),
        "excluded_records": stage_b_excluded,
        "selected_proxy_tokens": sum(int(row["token_count_proxy"]) for row in stage_b),
    }

    self_rows = load_self_knowledge(repo_root, groups["neoma_self_knowledge"], eval_index)
    self_selected, self_skips = select_ranked(
        self_rows,
        int(groups["neoma_self_knowledge"]["target_proxy_tokens"]),
        0,
        dedup,
    )
    selected.extend(self_selected)
    selection_notes["neoma_self_knowledge"] = {
        "allowlisted_records": len(self_rows),
        "selected_records": len(self_selected),
        "selected_proxy_tokens": sum(int(row["token_count_proxy"]) for row in self_selected),
        "skipped": dict(self_skips),
    }

    repo_policy = groups["repository_sources"]
    for source_id, quota in repo_policy["source_quotas"].items():
        pool = load_repository_pool(filter_root, source_id, repo_policy, eval_index)
        chosen, reasons = select_ranked(pool, int(quota), int(repo_policy["family_cap"]), dedup)
        selected.extend(chosen)
        selection_notes[source_id] = {"pool_records": len(pool), "selected_records": len(chosen), "selected_proxy_tokens": sum(int(row["token_count_proxy"]) for row in chosen), "skipped": dict(reasons)}

    wiki_policy = groups["wikimedia_english"]
    for source_id, quota in wiki_policy["source_quotas"].items():
        pool = load_wikimedia_pool(filter_root, source_id, wiki_policy, eval_index)
        chosen, reasons = select_ranked(pool, int(quota), int(wiki_policy["family_cap"]), dedup)
        selected.extend(chosen)
        selection_notes[source_id] = {"pool_records": len(pool), "selected_records": len(chosen), "selected_proxy_tokens": sum(int(row["token_count_proxy"]) for row in chosen), "skipped": dict(reasons)}

    total_tokens = sum(int(row["token_count_proxy"]) for row in selected)
    budget = plan["sample_budget"]
    minimum = int(budget["minimum_proxy_tokens_after_review"])
    maximum = int(budget["maximum_proxy_tokens_after_review"])
    if not minimum <= total_tokens <= maximum:
        raise StagingError(f"candidate sample has {total_tokens} proxy tokens; required range is {minimum}..{maximum}")

    parent = out_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=out_root.name + ".tmp-", dir=parent))
    try:
        text_root = temp / "texts"
        output_rows: list[dict[str, Any]] = []
        for row in sorted(selected, key=lambda item: (str(item["group_id"]), str(item["source_id"]), str(item["record_id"]))):
            filename = safe_filename(str(row["record_id"]), str(row["decoded_sha256"]))
            relative = Path("texts") / str(row["group_id"]) / str(row["source_id"]) / filename
            destination = temp / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(str(row["text"]) + "\n", encoding="utf-8", newline="\n")
            output = dict(row)
            output.pop("text", None)
            output["text_relative_path"] = relative.as_posix()
            output["stored_sha256"] = sha256_file(destination)
            output_rows.append(output)
        atomic_write_jsonl(temp / "records.jsonl", output_rows)
        write_review_csv(temp / "review_sample.csv", output_rows)
        counts_by_group = Counter(str(row["group_id"]) for row in output_rows)
        tokens_by_group = Counter()
        tokens_by_source = Counter()
        for row in output_rows:
            tokens_by_group[str(row["group_id"])] += int(row["token_count_proxy"])
            tokens_by_source[str(row["source_id"])] += int(row["token_count_proxy"])
        manifest = {
            "schema_version": "1.0",
            "manifest_id": "stage_a_tokenizer_sample_v0_1_candidate",
            "baseline": BASELINE,
            "generated_utc": utc_now(),
            "plan_path": plan_path.relative_to(repo_root).as_posix() if plan_path.is_relative_to(repo_root) else plan_path.as_posix(),
            "plan_sha256": sha256_file(plan_path),
            "status": "tokenizer_sample_candidate_pending_review",
            "record_count": len(output_rows),
            "proxy_token_count": total_tokens,
            "minimum_proxy_tokens_after_review": minimum,
            "maximum_proxy_tokens_after_review": maximum,
            "counts_by_group": dict(sorted(counts_by_group.items())),
            "tokens_by_group": dict(sorted(tokens_by_group.items())),
            "tokens_by_source": dict(sorted(tokens_by_source.items())),
            "selection_notes": selection_notes,
            "records_sha256": sha256_file(temp / "records.jsonl"),
            "review_sample_sha256": sha256_file(temp / "review_sample.csv"),
            "training_allowed": False,
            "model_training_allowed": False,
            "tokenizer_training_allowed": False,
            "next_step": "Review the manifest and stratified review sample, then approve only by binding a review decision to this manifest file SHA-256.",
        }
        atomic_write_json(temp / "manifest.json", manifest)
        atomic_replace_directory(temp, out_root, force)
        return manifest
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def approve_candidate(candidate_root: Path, decision_path: Path, approved_root: Path, force: bool) -> dict[str, Any]:
    manifest_path = candidate_root / "manifest.json"
    records_path = candidate_root / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise StagingError("candidate tokenizer sample is missing manifest or records")
    candidate_manifest = load_json(manifest_path)
    if candidate_manifest.get("tokenizer_training_allowed") is not False or candidate_manifest.get("model_training_allowed") is not False:
        raise StagingError("candidate sample has unexpected permissions")
    decision = load_json(decision_path)
    if decision.get("status") != "approved" or decision.get("approved_for_tokenizer_comparison") is not True:
        raise StagingError("review decision is not approved for tokenizer comparison")
    if decision.get("model_training_allowed") is not False:
        raise StagingError("review decision must keep model_training_allowed=false")
    expected_hash = sha256_file(manifest_path)
    if decision.get("candidate_manifest_sha256") != expected_hash:
        raise StagingError("review decision is not bound to the current candidate manifest")
    if not str(decision.get("reviewer", "")).strip() or not decision.get("reviewed_utc"):
        raise StagingError("review decision requires reviewer and reviewed_utc")

    excluded = {str(value) for value in decision.get("excluded_record_ids", [])}
    candidate_rows = load_jsonl(records_path)
    candidate_ids = {str(row["record_id"]) for row in candidate_rows}
    unknown_exclusions = sorted(excluded - candidate_ids)
    if unknown_exclusions:
        raise StagingError(f"review decision excludes unknown record IDs: {unknown_exclusions}")
    rows = [row for row in candidate_rows if str(row["record_id"]) not in excluded]
    total_tokens = sum(int(row["token_count_proxy"]) for row in rows)
    minimum = int(candidate_manifest["minimum_proxy_tokens_after_review"])
    maximum = int(candidate_manifest["maximum_proxy_tokens_after_review"])
    if not minimum <= total_tokens <= maximum:
        raise StagingError("review exclusions move the sample outside the permitted token range")

    parent = approved_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=approved_root.name + ".tmp-", dir=parent))
    try:
        approved_rows: list[dict[str, Any]] = []
        for row in rows:
            source = candidate_root / str(row["text_relative_path"])
            if not source.is_file() or sha256_file(source) != row["stored_sha256"]:
                raise StagingError(f"candidate text integrity mismatch: {row['record_id']}")
            destination = temp / str(row["text_relative_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            approved = dict(row)
            approved.update(
                status="approved_for_tokenizer_comparison_only",
                training_allowed=False,
                model_training_allowed=False,
                tokenizer_training_allowed=True,
                review_id=decision.get("review_id"),
            )
            approved_rows.append(approved)
        atomic_write_jsonl(temp / "records.jsonl", approved_rows)
        shutil.copyfile(decision_path, temp / "review_decision.json")
        manifest = {
            "schema_version": "1.0",
            "manifest_id": "stage_a_tokenizer_sample_v0_1",
            "baseline": BASELINE,
            "generated_utc": utc_now(),
            "status": "approved_for_tokenizer_comparison_only",
            "candidate_manifest_sha256": expected_hash,
            "review_decision_sha256": sha256_file(decision_path),
            "record_count": len(approved_rows),
            "proxy_token_count": total_tokens,
            "records_sha256": sha256_file(temp / "records.jsonl"),
            "excluded_record_ids": sorted(excluded),
            "training_allowed": False,
            "model_training_allowed": False,
            "tokenizer_training_allowed": True,
            "explicit_prohibition": "This directory must not be passed to model dataset preparation or model training.",
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
    build.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    build.add_argument("--out-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    build.add_argument("--execute", action="store_true")
    build.add_argument("--force", action="store_true")

    approve = subparsers.add_parser("approve")
    approve.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    approve.add_argument("--review-decision", type=Path, required=True)
    approve.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)
    approve.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "build":
        result = build_candidate(args.repo_root.resolve(), args.plan.resolve(), args.filter_root.resolve(), args.out_root.resolve(), args.execute, args.force)
    else:
        result = approve_candidate(args.candidate_root.resolve(), args.review_decision.resolve(), args.approved_root.resolve(), args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
