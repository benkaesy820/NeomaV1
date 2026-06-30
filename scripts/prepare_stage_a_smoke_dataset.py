#!/usr/bin/env python3
"""Prepare the explicitly approved Stage A smoke slice into uint16 train/val files.

This is a bounded model-dataset preparation step for Work Packet 15 only. It
splits by document family before tokenization, appends EOS per record, writes no
answer masks, and emits a resolved training config bound to exact hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
try:
    from tokenizers import Tokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'tokenizers' package is required for Stage A dataset preparation.") from exc

from stage_a_staging_common import StagingError, atomic_write_json, atomic_write_jsonl, load_json, sha256_file, utc_now
from stage_a_smoke_common import (
    SMOKE_SCOPE,
    atomic_replace_directory,
    load_jsonl,
    protected_evaluation_state,
    stable_rank,
    verify_special_tokens,
    verify_text_record,
)
from train_tokenizer import special_tokens

DEFAULT_APPROVED_ROOT = Path("data/foundation/approved/stage_a_smoke_probe_v0_1")
DEFAULT_TOKENIZER = Path("data/foundation/tokenizers/stage_a_v0_1/stage_a_bpe_8000.json")
DEFAULT_CONFIG_TEMPLATE = Path("configs/stage_a_smoke_probe_8k_cpu.json")
DEFAULT_OUT_ROOT = Path("data/foundation/processed/stage_a_smoke_probe_v0_1")


def validate_approved_slice(root: Path, repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    records_path = root / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise StagingError("approved Stage A smoke slice is missing")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "approved_for_stage_a_smoke_probe_only":
        raise StagingError("smoke slice has not been approved")
    if manifest.get("training_scope") != SMOKE_SCOPE:
        raise StagingError("smoke slice has the wrong training scope")
    if manifest.get("training_allowed") is not True or manifest.get("model_training_allowed") is not True:
        raise StagingError("smoke slice does not grant model-training permission")
    if manifest.get("tokenizer_training_allowed") is not False:
        raise StagingError("smoke slice must not grant tokenizer-training permission")
    if manifest.get("records_sha256") != sha256_file(records_path):
        raise StagingError("smoke slice records hash mismatch")
    rows = load_jsonl(records_path)
    if len(rows) != int(manifest.get("record_count", -1)):
        raise StagingError("smoke slice record count mismatch")
    for row in rows:
        if row.get("training_scope") != SMOKE_SCOPE:
            raise StagingError(f"{row.get('record_id')}: wrong training scope")
        if row.get("training_allowed") is not True or row.get("model_training_allowed") is not True:
            raise StagingError(f"{row.get('record_id')}: missing smoke training permission")
        if row.get("tokenizer_training_allowed") is not False:
            raise StagingError(f"{row.get('record_id')}: unexpected tokenizer permission")
        if row.get("group_id") == "frozen_stage_b":
            raise StagingError("frozen Stage B instruction data cannot enter Stage A smoke training")
        verify_text_record(root, row)
    _, current_fingerprint, current_count = protected_evaluation_state(repo_root)
    if manifest.get("protected_evaluation_fingerprint") != current_fingerprint:
        raise StagingError("protected evaluation set changed after smoke-slice approval")
    if int(manifest.get("protected_evaluation_item_count", -1)) != current_count:
        raise StagingError("protected evaluation item count changed after smoke-slice approval")
    return manifest, rows


def load_bound_tokenizer(path: Path, manifest: dict[str, Any]) -> tuple[Tokenizer, int]:
    if not path.is_file():
        raise StagingError(f"missing provisional tokenizer: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != manifest.get("tokenizer_sha256"):
        raise StagingError("tokenizer hash does not match the approved smoke slice")
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size() != int(manifest.get("tokenizer_vocab_size", -1)):
        raise StagingError("tokenizer vocabulary size differs from the approved smoke slice")
    verify_special_tokens(tokenizer, special_tokens("code"))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise StagingError("tokenizer is missing <eos>")
    return tokenizer, int(eos_id)


def encode_rows(
    root: Path,
    rows: list[dict[str, Any]],
    tokenizer: Tokenizer,
    eos_id: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, int]]:
    encoded: dict[str, list[np.ndarray]] = defaultdict(list)
    exact_counts: dict[str, int] = {}
    for row in rows:
        _, text = verify_text_record(root, row)
        ids = tokenizer.encode(text).ids
        ids.append(eos_id)
        if not ids:
            raise StagingError(f"{row.get('record_id')}: encoded to no tokens")
        array = np.asarray(ids, dtype=np.uint16)
        encoded[str(row["record_id"])].append(array)
        exact_counts[str(row["record_id"])] = int(array.size)
    return encoded, exact_counts


def split_families(
    rows: list[dict[str, Any]],
    exact_counts: dict[str, int],
    val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], dict[str, Any]]:
    if not 0.05 <= val_fraction <= 0.25:
        raise StagingError("validation fraction must stay between 5% and 25% for the smoke probe")
    by_source_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        source_id = str(row["source_id"])
        family_id = str(row.get("family_id") or row["record_id"])
        by_source_family[source_id][family_id].append(row)

    val_families: set[str] = set()
    train_families: set[str] = set()
    split_notes: dict[str, Any] = {}
    for source_id, families in sorted(by_source_family.items()):
        family_rows = list(families.items())
        family_rows.sort(key=lambda item: stable_rank(seed, source_id, item[0]))
        total_tokens = sum(
            exact_counts[str(row["record_id"])]
            for _, members in family_rows
            for row in members
        )
        target = max(1, round(total_tokens * val_fraction))
        val_tokens = 0
        # A source with only one family cannot be split safely; keep it in train.
        if len(family_rows) >= 2:
            for family_id, members in family_rows:
                if val_tokens >= target and val_families:
                    break
                val_families.add(family_id)
                val_tokens += sum(exact_counts[str(row["record_id"])] for row in members)
        for family_id, _ in family_rows:
            if family_id not in val_families:
                train_families.add(family_id)
        split_notes[source_id] = {
            "family_count": len(family_rows),
            "total_tokens": total_tokens,
            "validation_tokens": val_tokens,
            "validation_family_count": sum(family_id in val_families for family_id, _ in family_rows),
        }

    overlap = train_families & val_families
    if overlap:
        raise StagingError(f"family split overlap: {sorted(overlap)[:5]}")
    if not val_families:
        raise StagingError("no validation families were selected")
    return train_families, val_families, split_notes


def write_split(
    rows: list[dict[str, Any]],
    root: Path,
    tokenizer: Tokenizer,
    eos_id: int,
    train_families: set[str],
    val_families: set[str],
    out_root: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    metadata: list[dict[str, Any]] = []
    train_tokens = 0
    val_tokens = 0
    train_path = out_root / "train.bin"
    val_path = out_root / "val.bin"
    with train_path.open("wb") as train_handle, val_path.open("wb") as val_handle:
        for row in sorted(rows, key=lambda value: (str(value["source_id"]), str(value["family_id"]), str(value["record_id"]))):
            _, text = verify_text_record(root, row)
            ids = tokenizer.encode(text).ids
            ids.append(eos_id)
            array = np.asarray(ids, dtype=np.uint16)
            family_id = str(row.get("family_id") or row["record_id"])
            if family_id in val_families:
                split = "val"
                offset = val_tokens
                array.tofile(val_handle)
                val_tokens += int(array.size)
            elif family_id in train_families:
                split = "train"
                offset = train_tokens
                array.tofile(train_handle)
                train_tokens += int(array.size)
            else:
                raise StagingError(f"{row.get('record_id')}: family was not assigned")
            metadata.append({
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "group_id": row["group_id"],
                "family_id": family_id,
                "split": split,
                "token_offset": offset,
                "token_count": int(array.size),
                "stored_sha256": row["stored_sha256"],
                "training_scope": SMOKE_SCOPE,
            })
    return metadata, train_tokens, val_tokens


def prepare(
    repo_root: Path,
    approved_root: Path,
    tokenizer_path: Path,
    config_template_path: Path,
    out_root: Path,
    val_fraction: float,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    approved_manifest, rows = validate_approved_slice(approved_root, repo_root)
    tokenizer, eos_id = load_bound_tokenizer(tokenizer_path, approved_manifest)
    _, exact_counts = encode_rows(approved_root, rows, tokenizer, eos_id)
    exact_total = sum(exact_counts.values())
    if not 25000 <= exact_total <= 50000:
        raise StagingError(f"approved smoke slice encodes to {exact_total} tokens; required 25K-50K")
    train_families, val_families, split_notes = split_families(rows, exact_counts, val_fraction, seed)

    config_template = load_json(config_template_path)
    if config_template.get("run_name") != "stage_a_smoke_probe_8k":
        raise StagingError("unexpected smoke config template")
    if int(config_template.get("vocab_size", -1)) != tokenizer.get_vocab_size():
        raise StagingError("config template vocabulary does not match the provisional tokenizer")
    if config_template.get("train_loss_mask") or config_template.get("val_loss_mask"):
        raise StagingError("Stage A smoke training must use normal next-token loss without answer masks")

    parent = out_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=out_root.name + ".tmp-", dir=parent))
    try:
        metadata, train_tokens, val_tokens = write_split(
            rows, approved_root, tokenizer, eos_id, train_families, val_families, temp,
        )
        seq_len = int(config_template["seq_len"])
        if train_tokens <= seq_len + 1 or val_tokens <= seq_len + 1:
            raise StagingError("prepared split is too small for the configured sequence length")
        atomic_write_jsonl(temp / "records.jsonl", metadata)
        manifest = {
            "schema_version": "1.0",
            "manifest_id": "stage_a_smoke_probe_v0_1_dataset",
            "generated_utc": utc_now(),
            "status": "prepared_for_stage_a_smoke_probe_only",
            "training_scope": SMOKE_SCOPE,
            "input_manifest_sha256": sha256_file(approved_root / "manifest.json"),
            "input_records_sha256": approved_manifest["records_sha256"],
            "tokenizer_path": tokenizer_path.as_posix(),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "tokenizer_vocab_size": tokenizer.get_vocab_size(),
            "special_token_ids": verify_special_tokens(tokenizer, special_tokens("code")),
            "eos_id": eos_id,
            "record_count": len(rows),
            "family_count": len(train_families | val_families),
            "train_family_count": len(train_families),
            "val_family_count": len(val_families),
            "family_overlap_count": 0,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "total_tokens": train_tokens + val_tokens,
            "val_fraction_requested": val_fraction,
            "val_fraction_actual": val_tokens / (train_tokens + val_tokens),
            "split_seed": seed,
            "split_notes": split_notes,
            "train_bin_sha256": sha256_file(temp / "train.bin"),
            "val_bin_sha256": sha256_file(temp / "val.bin"),
            "records_sha256": sha256_file(temp / "records.jsonl"),
            "protected_evaluation_fingerprint": approved_manifest["protected_evaluation_fingerprint"],
            "protected_evaluation_item_count": approved_manifest["protected_evaluation_item_count"],
            "instruction_loss_mask": False,
            "training_allowed": True,
            "model_training_allowed": True,
            "tokenizer_training_allowed": False,
            "explicit_limit": "Prepared only for the bounded Work Packet 15 Stage A smoke probe.",
        }
        atomic_write_json(temp / "manifest.json", manifest)

        resolved = dict(config_template)
        resolved.update({
            "tokenizer_path": tokenizer_path.as_posix(),
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "train_data": (out_root / "train.bin").as_posix(),
            "val_data": (out_root / "val.bin").as_posix(),
            "dataset_manifest": (out_root / "manifest.json").as_posix(),
            # Filled after the manifest is written into its final local directory.
            "dataset_manifest_sha256": None,
            "train_loss_mask": None,
            "val_loss_mask": None,
        })
        atomic_write_json(temp / "train_config.unbound.json", resolved)
        atomic_replace_directory(temp, out_root, force)

        # Bind the final resolved config to the exact dataset manifest. A
        # separate binding sidecar records both hashes and avoids a recursive
        # manifest/config hash dependency.
        resolved["dataset_manifest_sha256"] = sha256_file(out_root / "manifest.json")
        atomic_write_json(out_root / "train_config.json", resolved)
        (out_root / "train_config.unbound.json").unlink(missing_ok=True)
        atomic_write_json(out_root / "binding.json", {
            "schema_version": "1.0",
            "dataset_manifest_sha256": sha256_file(out_root / "manifest.json"),
            "resolved_train_config_sha256": sha256_file(out_root / "train_config.json"),
            "training_scope": SMOKE_SCOPE,
        })
        return load_json(out_root / "manifest.json")
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--config-template", type=Path, default=DEFAULT_CONFIG_TEMPLATE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare(
        args.repo_root.resolve(), args.approved_root.resolve(), args.tokenizer.resolve(),
        args.config_template.resolve(), args.out_root.resolve(), args.val_fraction,
        args.seed, args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
