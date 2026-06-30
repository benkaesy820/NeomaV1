#!/usr/bin/env python3
"""Verify Work Packet 17 extended 250K diagnostic artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage_a_probe_common import PROBE_SCOPE
from stage_a_staging_common import StagingError, load_json, sha256_file
from verify_stage_a_250k_probe import load_checkpoint, verify_dataset, verify_slice

DEFAULT_APPROVED_ROOT = Path("data/foundation/approved/stage_a_250k_probe_v0_1")
DEFAULT_DATASET_ROOT = Path("data/foundation/processed/stage_a_250k_probe_v0_1")
DEFAULT_RUN_ROOT = Path("runs/stage_a_250k_extended_8k")
MILESTONES = (500, 1000, 1500, 2000)


def verify_run(root: Path, require_complete: bool = True) -> dict[str, Any]:
    report_path = root / "extended_probe_report.json"
    if not report_path.is_file():
        raise StagingError(f"missing extended diagnostic report: {report_path}")
    report = load_json(report_path)
    if require_complete and report.get("status") != "passed":
        raise StagingError(f"extended report status is {report.get('status')!r}")
    if int(report.get("completed_step", -1)) != 2000:
        raise StagingError("extended diagnostic did not reach 2,000 steps")
    if int(report.get("training_tokens_seen", -1)) != 1_024_000:
        raise StagingError("extended diagnostic token budget changed")
    if report.get("model_training_scope") != PROBE_SCOPE:
        raise StagingError("extended diagnostic scope mismatch")
    if report.get("capability_claim_authorized") is not False or report.get("expansion_authorized") is not False:
        raise StagingError("extended diagnostic must not authorize capability or expansion")
    for key in (
        "resume_verified",
        "loss_decrease_verified",
        "validation_loss_improved",
        "generation_verified",
        "special_tokens_verified",
    ):
        if report.get(key) is not True:
            raise StagingError(f"report did not verify {key}")
    if report.get("eval_leakage_count") != 0 or report.get("stage_b_record_count") != 0:
        raise StagingError("protected evaluation or Stage B data entered the diagnostic")
    if float(report["final_train_loss"]) >= float(report["initial_train_loss"]):
        raise StagingError("final train loss did not improve")
    if float(report["final_val_loss"]) >= float(report["initial_val_loss"]):
        raise StagingError("final validation loss did not improve")
    if not isinstance(report.get("peak_rss_bytes"), int) or int(report["peak_rss_bytes"]) <= 0:
        raise StagingError("peak RSS was not recorded")
    if not isinstance(report.get("effective_tokens_per_second"), (int, float)) or float(report["effective_tokens_per_second"]) <= 0:
        raise StagingError("training speed was not recorded")

    latest_path = root / "latest.pt"
    best_path = root / "best.pt"
    if report.get("latest_checkpoint_sha256") != sha256_file(latest_path):
        raise StagingError("latest checkpoint hash mismatch")
    if report.get("best_checkpoint_sha256") != sha256_file(best_path):
        raise StagingError("best checkpoint hash mismatch")
    latest = load_checkpoint(latest_path)
    if int(latest.get("step", -1)) != 2000:
        raise StagingError("latest checkpoint step mismatch")
    for step in MILESTONES:
        checkpoint = root / f"checkpoint_step_{step:04d}.pt"
        generation = root / f"generation_step_{step:04d}.json"
        if not checkpoint.is_file() or not generation.is_file():
            raise StagingError(f"missing milestone artifacts for step {step}")
        if load_json(generation).get("checkpoint_step") != step:
            raise StagingError(f"generation milestone mismatch for step {step}")
    if not (root / "generation_step_0000.json").is_file():
        raise StagingError("missing pretrain generation sample")
    baseline_delta = report.get("work_packet_16_delta")
    if not isinstance(baseline_delta, dict) or float(baseline_delta["tokens_seen_ratio"]) != 4.0:
        raise StagingError("Work Packet 16 comparison is missing or malformed")
    return {
        "status": "ok",
        "completed_step": report["completed_step"],
        "initial_train_loss": report["initial_train_loss"],
        "final_train_loss": report["final_train_loss"],
        "initial_val_loss": report["initial_val_loss"],
        "final_val_loss": report["final_val_loss"],
        "best_val_loss": report["best_val_loss"],
        "best_val_loss_step": report["best_val_loss_step"],
        "overfit_warning": report["overfit_warning"],
        "effective_tokens_per_second": report["effective_tokens_per_second"],
        "peak_rss_bytes": report["peak_rss_bytes"],
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
        raise StagingError("approved 250K slice is required")
    if args.require_dataset or args.dataset_root.exists():
        result["dataset"] = verify_dataset(args.dataset_root.resolve())
    elif args.require_dataset:
        raise StagingError("prepared 250K dataset is required")
    if args.require_run or args.run_root.exists():
        result["run"] = verify_run(args.run_root.resolve(), require_complete=args.require_run)
    elif args.require_run:
        raise StagingError("completed extended diagnostic run is required")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
