#!/usr/bin/env python3
"""Verify Work Packet 15 slice, dataset, and smoke-training run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
try:
    from tokenizers import Tokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'tokenizers' package is required for smoke verification.") from exc

from stage_a_staging_common import StagingError, load_json, sha256_file
from stage_a_smoke_common import SMOKE_SCOPE, load_jsonl, verify_special_tokens, verify_text_record
from train_tokenizer import special_tokens

DEFAULT_APPROVED_ROOT = Path("data/foundation/approved/stage_a_smoke_probe_v0_1")
DEFAULT_DATASET_ROOT = Path("data/foundation/processed/stage_a_smoke_probe_v0_1")
DEFAULT_RUN_ROOT = Path("runs/stage_a_smoke_probe_8k")


def verify_slice(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    records_path = root / "records.jsonl"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "approved_for_stage_a_smoke_probe_only":
        raise StagingError("smoke slice is not approved")
    if manifest.get("training_scope") != SMOKE_SCOPE:
        raise StagingError("smoke slice scope mismatch")
    if manifest.get("training_allowed") is not True or manifest.get("model_training_allowed") is not True:
        raise StagingError("smoke slice does not grant model training")
    if manifest.get("tokenizer_training_allowed") is not False:
        raise StagingError("smoke slice unexpectedly grants tokenizer training")
    if manifest.get("records_sha256") != sha256_file(records_path):
        raise StagingError("smoke slice records hash mismatch")
    rows = load_jsonl(records_path)
    if len(rows) != int(manifest["record_count"]):
        raise StagingError("smoke slice record count mismatch")
    total = 0
    families: set[str] = set()
    sources: set[str] = set()
    for row in rows:
        if row.get("group_id") == "frozen_stage_b":
            raise StagingError("Stage B data entered the Stage A smoke slice")
        if row.get("training_scope") != SMOKE_SCOPE:
            raise StagingError(f"{row.get('record_id')}: scope mismatch")
        if row.get("training_allowed") is not True or row.get("model_training_allowed") is not True:
            raise StagingError(f"{row.get('record_id')}: permission mismatch")
        if row.get("leakage_findings"):
            raise StagingError(f"{row.get('record_id')}: leakage findings are not empty")
        verify_text_record(root, row)
        total += int(row["actual_token_count"])
        families.add(str(row["family_id"]))
        sources.add(str(row["source_id"]))
    if total != int(manifest["actual_token_count"]):
        raise StagingError("smoke slice token total mismatch")
    if not 25000 <= total <= 50000:
        raise StagingError("smoke slice token total is outside 25K-50K")
    return {
        "status": "ok",
        "record_count": len(rows),
        "actual_token_count": total,
        "source_count": len(sources),
        "family_count": len(families),
        "manifest_sha256": sha256_file(manifest_path),
    }


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    binding_path = root / "binding.json"
    config_path = root / "train_config.json"
    manifest = load_json(manifest_path)
    binding = load_json(binding_path)
    config = load_json(config_path)
    if manifest.get("status") != "prepared_for_stage_a_smoke_probe_only":
        raise StagingError("smoke dataset status mismatch")
    if manifest.get("training_scope") != SMOKE_SCOPE or config.get("training_scope") != SMOKE_SCOPE:
        raise StagingError("smoke dataset/config scope mismatch")
    if manifest.get("instruction_loss_mask") is not False:
        raise StagingError("Stage A smoke dataset must not use answer masks")
    if config.get("train_loss_mask") or config.get("val_loss_mask"):
        raise StagingError("Stage A smoke config must not use answer masks")
    if binding.get("dataset_manifest_sha256") != sha256_file(manifest_path):
        raise StagingError("dataset binding manifest hash mismatch")
    if binding.get("resolved_train_config_sha256") != sha256_file(config_path):
        raise StagingError("dataset binding config hash mismatch")
    if config.get("dataset_manifest_sha256") != sha256_file(manifest_path):
        raise StagingError("resolved config is not bound to the dataset manifest")

    train_path = root / "train.bin"
    val_path = root / "val.bin"
    if sha256_file(train_path) != manifest.get("train_bin_sha256"):
        raise StagingError("train.bin hash mismatch")
    if sha256_file(val_path) != manifest.get("val_bin_sha256"):
        raise StagingError("val.bin hash mismatch")
    train = np.memmap(train_path, dtype=np.uint16, mode="r")
    val = np.memmap(val_path, dtype=np.uint16, mode="r")
    if len(train) != int(manifest["train_tokens"]) or len(val) != int(manifest["val_tokens"]):
        raise StagingError("prepared token count mismatch")

    rows = load_jsonl(root / "records.jsonl")
    train_families = {str(row["family_id"]) for row in rows if row["split"] == "train"}
    val_families = {str(row["family_id"]) for row in rows if row["split"] == "val"}
    if train_families & val_families:
        raise StagingError("document family leaked across train/validation")
    tokenizer_path = Path(str(config["tokenizer_path"]))
    if sha256_file(tokenizer_path) != manifest.get("tokenizer_sha256"):
        raise StagingError("dataset tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    special_ids = verify_special_tokens(tokenizer, special_tokens("code"))
    return {
        "status": "ok",
        "train_tokens": len(train),
        "val_tokens": len(val),
        "record_count": len(rows),
        "train_family_count": len(train_families),
        "val_family_count": len(val_families),
        "special_token_count": len(special_ids),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except ImportError as exc:  # pragma: no cover
        raise StagingError("PyTorch is required to verify checkpoints") from exc


def verify_run(root: Path, require_complete: bool = True) -> dict[str, Any]:
    report_path = root / "smoke_probe_report.json"
    if not report_path.is_file():
        raise StagingError(f"missing smoke report: {report_path}")
    report = load_json(report_path)
    expected_status = "passed" if require_complete else report.get("status")
    if report.get("status") != expected_status:
        raise StagingError(f"smoke report status is {report.get('status')!r}")
    latest_path = root / "latest.pt"
    best_path = root / "best.pt"
    if not latest_path.is_file() or not best_path.is_file():
        raise StagingError("latest/best checkpoint is missing")
    if report.get("latest_checkpoint_sha256") != sha256_file(latest_path):
        raise StagingError("latest checkpoint hash mismatch")
    if report.get("best_checkpoint_sha256") != sha256_file(best_path):
        raise StagingError("best checkpoint hash mismatch")
    latest = load_checkpoint(latest_path)
    if int(latest.get("step", -1)) != int(report["completed_step"]):
        raise StagingError("latest checkpoint step differs from report")
    if report.get("resume_verified") is not True:
        raise StagingError("resume was not verified")
    if report.get("loss_decrease_verified") is not True:
        raise StagingError("loss decrease was not verified")
    if report.get("generation_verified") is not True:
        raise StagingError("generation was not verified")
    if report.get("special_tokens_verified") is not True:
        raise StagingError("special tokens were not verified")
    if report.get("eval_leakage_count") != 0:
        raise StagingError("report indicates evaluation leakage")
    return {
        "status": "ok",
        "completed_step": report["completed_step"],
        "initial_train_loss": report["initial_train_loss"],
        "final_train_loss": report["final_train_loss"],
        "initial_val_loss": report["initial_val_loss"],
        "best_val_loss": report["best_val_loss"],
        "latest_checkpoint_sha256": report["latest_checkpoint_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--require-slice", action="store_true")
    parser.add_argument("--require-dataset", action="store_true")
    parser.add_argument("--require-run", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {}
    if args.require_slice or args.approved_root.exists():
        result["slice"] = verify_slice(args.approved_root.resolve())
    elif args.require_slice:
        raise StagingError("approved smoke slice is required")
    if args.require_dataset or args.dataset_root.exists():
        result["dataset"] = verify_dataset(args.dataset_root.resolve())
    elif args.require_dataset:
        raise StagingError("prepared smoke dataset is required")
    if args.require_run or args.run_root.exists():
        result["run"] = verify_run(args.run_root.resolve(), require_complete=args.require_run)
    elif args.require_run:
        raise StagingError("completed smoke run is required")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
