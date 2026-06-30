#!/usr/bin/env python3
"""Train and benchmark 2K/4K/8K tokenizer candidates on the approved sample.

This command trains tokenizers only. It refuses candidate samples that have not
been explicitly approved for tokenizer comparison and never prepares a model
training dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from stage_a_staging_common import StagingError, atomic_write_json, sha256_file, utc_now

try:
    from tokenizers import Tokenizer
    from train_tokenizer import build_tokenizer, special_tokens
except ImportError as exc:  # pragma: no cover - exercised in the normal Neoma environment
    raise SystemExit("The 'tokenizers' package is required for tokenizer comparison.") from exc

CONTEXT_LIMITS = (128, 192, 256, 384, 512)
INSTRUCTION_START_RE = re.compile(r"<instruction(?:\s[^>]*)?>", re.I)


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



def all_stage_b_texts(repo_root: Path, plan: dict[str, Any]) -> list[tuple[str, str]]:
    policy = next(
        row for row in plan["selection_groups"] if row["group_id"] == "frozen_stage_b"
    )
    rows: list[tuple[str, str]] = []
    for relative in policy["input_paths"]:
        path = repo_root / str(relative)
        if not path.is_file():
            raise StagingError(f"missing frozen Stage B file for benchmark: {relative}")
        text = path.read_text(encoding="utf-8-sig")
        starts = [match.start() for match in INSTRUCTION_START_RE.finditer(text)]
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                rows.append((f"{relative}:{index + 1:04d}", chunk))
    expected = int(plan["protected_data_policy"]["stage_b_expected_record_count"])
    if len(rows) != expected:
        raise StagingError(f"expected {expected} Stage B benchmark records, found {len(rows)}")
    return rows


def approved_texts(root: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], str]]]:
    manifest = load_json(root / "manifest.json")
    if manifest.get("status") != "approved_for_tokenizer_comparison_only":
        raise StagingError("sample is not approved for tokenizer comparison")
    if manifest.get("tokenizer_training_allowed") is not True:
        raise StagingError("sample does not grant tokenizer-only permission")
    if manifest.get("training_allowed") is not False or manifest.get("model_training_allowed") is not False:
        raise StagingError("sample improperly grants model-training permission")
    records_path = root / "records.jsonl"
    if manifest.get("records_sha256") != sha256_file(records_path):
        raise StagingError("approved records hash mismatch")
    pairs: list[tuple[dict[str, Any], str]] = []
    for row in load_jsonl(records_path):
        if row.get("tokenizer_training_allowed") is not True or row.get("model_training_allowed") is not False:
            raise StagingError(f"{row.get('record_id')}: invalid permission")
        path = root / str(row["text_relative_path"])
        if not path.is_file() or sha256_file(path) != row["stored_sha256"]:
            raise StagingError(f"{row.get('record_id')}: text integrity mismatch")
        pairs.append((row, path.read_text(encoding="utf-8")))
    return manifest, pairs


def model_parameter_count(vocab_size: int, architecture: dict[str, Any]) -> int:
    d_model = int(architecture["d_model"])
    n_layers = int(architecture["n_layers"])
    n_heads = int(architecture["n_heads"])
    n_kv_heads = int(architecture["n_kv_heads"])
    d_ff = int(architecture["d_ff"])
    head_dim = d_model // n_heads
    kv_dim = n_kv_heads * head_dim
    per_block = (
        2 * d_model
        + d_model * d_model
        + 2 * d_model * kv_dim
        + d_model * d_model
        + 3 * d_model * d_ff
    )
    return vocab_size * d_model + n_layers * per_block + d_model


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def train_candidates(sample_root: Path, plan: dict[str, Any], out_root: Path, force: bool) -> dict[str, Any]:
    sample_manifest, pairs = approved_texts(sample_root)
    out_root.mkdir(parents=True, exist_ok=True)
    configs = plan["tokenizer_candidates"]
    results: list[dict[str, Any]] = []
    texts = [text for _, text in pairs]
    for config in configs:
        vocab_size = int(config["vocab_size"])
        path = out_root / f"stage_a_bpe_{vocab_size}.json"
        if path.exists() and not force:
            raise StagingError(f"tokenizer exists; use --force after review: {path}")
        tokenizer, trainer = build_tokenizer(
            vocab_size=vocab_size,
            min_frequency=int(config["min_frequency"]),
            max_token_length=int(config["max_token_length"]),
            preset=str(config["preset"]),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
        tokenizer.save(str(path))
        results.append({
            "vocab_size_requested": vocab_size,
            "vocab_size_actual": tokenizer.get_vocab_size(),
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "preset": config["preset"],
            "min_frequency": config["min_frequency"],
            "max_token_length": config["max_token_length"],
        })
    manifest = {
        "schema_version": "1.0",
        "manifest_id": "stage_a_tokenizer_candidates_v0_1",
        "generated_utc": utc_now(),
        "sample_manifest_sha256": sha256_file(sample_root / "manifest.json"),
        "sample_records_sha256": sample_manifest["records_sha256"],
        "sample_record_count": sample_manifest["record_count"],
        "sample_proxy_token_count": sample_manifest["proxy_token_count"],
        "tokenizers": results,
        "model_dataset_prepared": False,
        "model_training_run": False,
    }
    atomic_write_json(out_root / "training_manifest.json", manifest)
    return manifest


def benchmark_one(
    tokenizer_path: Path,
    pairs: list[tuple[dict[str, Any], str]],
    stage_b_records: list[tuple[str, str]],
    architecture: dict[str, Any],
) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    unk_id = tokenizer.token_to_id("<unk>")
    group_bytes: Counter[str] = Counter()
    group_tokens: Counter[str] = Counter()
    source_bytes: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    stage_b_lengths: list[int] = []
    roundtrip_failures: list[str] = []
    unknown_tokens = 0
    used_ids: Counter[int] = Counter()
    all_lengths: list[int] = []

    for row, text in pairs:
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            roundtrip_failures.append(str(row["record_id"]))
        if unk_id is not None:
            unknown_tokens += ids.count(unk_id)
        used_ids.update(ids)
        byte_count = len(text.encode("utf-8"))
        group = str(row["group_id"])
        source = str(row["source_id"])
        group_bytes[group] += byte_count
        group_tokens[group] += len(ids)
        source_bytes[source] += byte_count
        source_tokens[source] += len(ids)
        all_lengths.append(len(ids))
    # Measure every frozen Stage B record, including the few records excluded
    # from tokenizer training because they overlap protected eval wording.
    stage_b_roundtrip_failures: list[str] = []
    for record_id, text in stage_b_records:
        ids = tokenizer.encode(text).ids
        stage_b_lengths.append(len(ids))
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            stage_b_roundtrip_failures.append(record_id)

    special_report: dict[str, bool] = {}
    for token in special_tokens("code"):
        token_id = tokenizer.token_to_id(token)
        special_report[token] = token_id is not None and tokenizer.encode(token).ids == [token_id]
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(special_tokens("code"))
    long_token_count = sum(
        1 for token in tokenizer.get_vocab() if len(token) > 32 and token not in special_set
    )
    return {
        "tokenizer": tokenizer_path.as_posix(),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "vocab_size": vocab_size,
        "model_parameters": model_parameter_count(vocab_size, architecture),
        "embedding_parameters": vocab_size * int(architecture["d_model"]),
        "roundtrip_failure_count": len(roundtrip_failures),
        "roundtrip_failure_ids": roundtrip_failures[:50],
        "unknown_token_count": unknown_tokens,
        "special_tokens_atomic": special_report,
        "used_vocab_count": len(used_ids),
        "vocab_utilization": round(len(used_ids) / max(1, vocab_size), 6),
        "tokens_seen_once": sum(1 for count in used_ids.values() if count == 1),
        "tokens_longer_than_32_characters": long_token_count,
        "overall": {
            "record_count": len(all_lengths),
            "total_tokens": sum(all_lengths),
            "median_record_tokens": statistics.median(all_lengths) if all_lengths else 0,
            "p90_record_tokens": round(percentile(all_lengths, 0.90), 3),
            "max_record_tokens": max(all_lengths, default=0),
        },
        "bytes_per_token_by_group": {
            group: round(group_bytes[group] / max(1, group_tokens[group]), 4)
            for group in sorted(group_bytes)
        },
        "bytes_per_token_by_source": {
            source: round(source_bytes[source] / max(1, source_tokens[source]), 4)
            for source in sorted(source_bytes)
        },
        "stage_b_context": {
            "record_count": len(stage_b_lengths),
            "median": statistics.median(stage_b_lengths) if stage_b_lengths else 0,
            "p90": round(percentile(stage_b_lengths, 0.90), 3),
            "p95": round(percentile(stage_b_lengths, 0.95), 3),
            "max": max(stage_b_lengths, default=0),
            "roundtrip_failure_count": len(stage_b_roundtrip_failures),
            "roundtrip_failure_ids": stage_b_roundtrip_failures[:50],
            "fit": {
                str(limit): {
                    "fits": sum(length <= limit for length in stage_b_lengths),
                    "exceeds": sum(length > limit for length in stage_b_lengths),
                }
                for limit in CONTEXT_LIMITS
            },
        },
    }


def recommend(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in results
        if row["roundtrip_failure_count"] == 0
        and row["unknown_token_count"] == 0
        and row["stage_b_context"]["roundtrip_failure_count"] == 0
        and row["tokens_longer_than_32_characters"] == 0
        and all(row["special_tokens_atomic"].values())
    ]
    if not eligible:
        return {"status": "no_candidate_passed_hard_gates", "recommended_vocab_size": None}
    # This is a transparent provisional ranking, not an automatic final choice.
    # Reward 256-context fit, then lower parameter count, then sample compression.
    ranked = sorted(
        eligible,
        key=lambda row: (
            -int(row["stage_b_context"]["fit"]["256"]["fits"]),
            int(row["model_parameters"]),
            int(row["overall"]["total_tokens"]),
        ),
    )
    best = ranked[0]
    return {
        "status": "provisional_recommendation_requires_leo_review_and_500k_probe_confirmation",
        "recommended_vocab_size": best["vocab_size"],
        "reason": "Best hard-gate-passing balance in order: Stage B 256-token fit, smaller model size, then compression. Final selection still requires the planned small training probe.",
    }


def benchmark(
    sample_root: Path,
    plan: dict[str, Any],
    tokenizer_root: Path,
    out_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    sample_manifest, pairs = approved_texts(sample_root)
    stage_b_records = all_stage_b_texts(repo_root, plan)
    training_manifest = load_json(tokenizer_root / "training_manifest.json")
    results: list[dict[str, Any]] = []
    for row in training_manifest["tokenizers"]:
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise StagingError(f"tokenizer integrity mismatch: {path}")
        results.append(benchmark_one(path, pairs, stage_b_records, plan["comparison_architecture"]))
    report = {
        "schema_version": "1.0",
        "report_id": "stage_a_tokenizer_comparison_v0_1",
        "generated_utc": utc_now(),
        "sample_manifest_sha256": sha256_file(sample_root / "manifest.json"),
        "sample_records_sha256": sample_manifest["records_sha256"],
        "architecture": plan["comparison_architecture"],
        "results": results,
        "recommendation": recommend(results),
        "model_dataset_prepared": False,
        "model_training_run": False,
    }
    atomic_write_json(out_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "benchmark", "all"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--sample-root", type=Path, default=Path("data/foundation/approved/stage_a_tokenizer_sample_v0_1"))
        sub.add_argument("--plan", type=Path, default=Path("data/foundation/manifests/stage_a_tokenizer_sample_v0_1_plan.json"))
        sub.add_argument("--tokenizer-root", type=Path, default=Path("data/foundation/tokenizers/stage_a_v0_1"))
        sub.add_argument("--report", type=Path, default=Path("data/foundation/tokenizers/stage_a_v0_1/comparison_report.json"))
        sub.add_argument("--repo-root", type=Path, default=Path("."))
        sub.add_argument("--force", action="store_true")
    args = parser.parse_args()
    plan = load_json(args.plan)
    result: dict[str, Any] = {}
    if args.command in {"train", "all"}:
        result["training"] = train_candidates(args.sample_root, plan, args.tokenizer_root, args.force)
    if args.command in {"benchmark", "all"}:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        result["benchmark"] = benchmark(
            args.sample_root,
            plan,
            args.tokenizer_root,
            args.report,
            args.repo_root.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
