#!/usr/bin/env python3
"""Run the 250K Stage A extended diagnostic through 2,000 steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage_a_probe_common import PROBE_SCOPE, verify_special_tokens
from stage_a_staging_common import StagingError, atomic_write_json, load_json, sha256_file, utc_now
from train_tokenizer import special_tokens
from verify_stage_a_250k_probe import verify_dataset, verify_slice
from run_stage_a_250k_probe import (
    checkpoint_generation,
    evaluation_rows,
    initial_generation,
    load_checkpoint,
    load_probe_rows,
    safe_remove_run,
)

DEFAULT_DATASET_CONFIG = Path("data/foundation/processed/stage_a_250k_probe_v0_1/train_config.json")
DEFAULT_TEMPLATE_CONFIG = Path("configs/stage_a_250k_extended_8k_cpu.json")
DEFAULT_APPROVED_ROOT = Path("data/foundation/approved/stage_a_250k_probe_v0_1")
DEFAULT_PROBES = Path("data/eval/stage_a_250k_probe_prompts_v0_1.jsonl")
DEFAULT_BASELINE_REPORT = Path("runs/stage_a_250k_probe_8k/probe_report.json")
DEFAULT_RESOLVED_CONFIG = Path("data/foundation/processed/stage_a_250k_probe_v0_1/train_config_extended_2000.json")
MILESTONES = (500, 1000, 1500, 2000)


def run_command(command: list[str], cwd: Path, log_path: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise StagingError(
            f"command failed with exit code {completed.returncode}; see {log_path}: {' '.join(command)}"
        )
    return completed, duration


def powershell_command(config: Path, milestone: int, first_phase: bool) -> str:
    display = config.as_posix().replace("/", "\\")
    if milestone == MILESTONES[-1]:
        suffix = "" if first_phase else " --auto-resume"
    else:
        suffix = f" --stop-after-step {milestone}" if first_phase else f" --auto-resume --stop-after-step {milestone}"
    return f".\\p scripts\\train.py --config {display}{suffix}"


def build_resolved_config(dataset_config: Path, template_config: Path, output_config: Path, repo_root: Path) -> dict[str, Any]:
    dataset = load_json(dataset_config)
    template = load_json(template_config)
    required_same = (
        "training_scope",
        "tokenizer_path",
        "train_data",
        "val_data",
        "dataset_manifest",
        "dataset_manifest_sha256",
        "tokenizer_sha256",
        "seed",
        "eval_seed",
        "vocab_size",
        "seq_len",
        "batch_size",
        "grad_accum_steps",
        "learning_rate",
        "min_learning_rate",
        "warmup_steps",
        "weight_decay",
        "grad_clip",
        "n_layers",
        "d_model",
        "n_heads",
        "n_kv_heads",
        "d_ff",
        "dropout",
        "rope_base",
        "train_loss_mask",
        "val_loss_mask",
    )
    for key in required_same:
        if key in template and dataset.get(key) != template.get(key):
            if key in {"tokenizer_path", "train_data", "val_data", "dataset_manifest"}:
                if Path(str(dataset[key])).resolve() == (repo_root / str(template[key])).resolve():
                    continue
            raise StagingError(f"extended config changed dataset-critical field {key}")
    if dataset.get("training_scope") != PROBE_SCOPE:
        raise StagingError("dataset config has the wrong training scope")
    resolved = dict(dataset)
    for key in ("run_name", "out_dir", "max_steps", "eval_interval", "eval_iters", "save_interval", "torch_threads"):
        resolved[key] = template[key]
    if int(resolved["max_steps"]) != 2000:
        raise StagingError("extended diagnostic must remain bounded to 2,000 steps")
    if int(resolved["max_steps"]) * int(resolved["batch_size"]) * int(resolved["seq_len"]) * int(resolved["grad_accum_steps"]) != 1_024_000:
        raise StagingError("extended diagnostic token budget drifted")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_config, resolved)
    return resolved


def baseline_summary(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise StagingError(f"Work Packet 16 baseline report is required: {report_path}")
    report = load_json(report_path)
    if report.get("status") != "passed" or int(report.get("completed_step", -1)) != 500:
        raise StagingError("Work Packet 16 baseline did not pass at 500 steps")
    return {
        "report_path": report_path.as_posix(),
        "report_sha256": sha256_file(report_path),
        "completed_step": int(report["completed_step"]),
        "training_tokens_seen": int(report["training_tokens_seen"]),
        "initial_train_loss": float(report["initial_train_loss"]),
        "final_train_loss": float(report["final_train_loss"]),
        "initial_val_loss": float(report["initial_val_loss"]),
        "final_val_loss": float(report["final_val_loss"]),
        "best_val_loss": float(report["best_val_loss"]),
        "best_val_loss_step": int(report["best_val_loss_step"]),
        "effective_tokens_per_second": float(report["effective_tokens_per_second"]),
        "peak_rss_bytes": int(report["peak_rss_bytes"]),
        "latest_checkpoint_bytes": int(report["latest_checkpoint_bytes"]),
    }


def repeated_token_diagnostics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    worst_run = 0
    lowest_unique_ratio = 1.0
    rows: list[dict[str, Any]] = []
    for sample in samples:
        tokens = str(sample.get("text", "")).split()
        if not tokens:
            ratio = 0.0
            run = 0
        else:
            ratio = len(set(tokens)) / len(tokens)
            run = 1
            current = 1
            for left, right in zip(tokens, tokens[1:]):
                current = current + 1 if left == right else 1
                run = max(run, current)
        worst_run = max(worst_run, run)
        lowest_unique_ratio = min(lowest_unique_ratio, ratio)
        rows.append({
            "id": sample.get("id"),
            "token_count": len(tokens),
            "unique_token_ratio": ratio,
            "max_repeated_token_run": run,
        })
    return {
        "lowest_unique_token_ratio": lowest_unique_ratio,
        "max_repeated_token_run": worst_run,
        "samples": rows,
    }


def point_by_step(rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for row in rows:
        if int(row["step"]) == step:
            return row
    raise StagingError(f"missing evaluation point for step {step}")


def run_probe(
    repo_root: Path,
    dataset_config: Path,
    template_config: Path,
    resolved_config: Path,
    approved_root: Path,
    probes_path: Path,
    baseline_report: Path,
    execute: bool,
    force: bool,
) -> dict[str, Any]:
    slice_result = verify_slice(approved_root)
    dataset_result = verify_dataset(dataset_config.parent)
    cfg = build_resolved_config(dataset_config, template_config, resolved_config, repo_root)
    run_root = (repo_root / str(cfg["out_dir"])).resolve()
    tokenizer_path = Path(str(cfg["tokenizer_path"])).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    probe_rows = load_probe_rows(probes_path)
    baseline = baseline_summary(baseline_report)

    phase_commands: list[dict[str, Any]] = []
    for index, milestone in enumerate(MILESTONES):
        argv = [sys.executable, "scripts/train.py", "--config", resolved_config.as_posix()]
        if index > 0:
            argv.append("--auto-resume")
        if milestone != MILESTONES[-1]:
            argv.extend(["--stop-after-step", str(milestone)])
        phase_commands.append({
            "milestone": milestone,
            "actual_argv": argv,
            "powershell": powershell_command(resolved_config, milestone, index == 0),
        })

    summary = {
        "schema_version": "1.0",
        "report_id": "stage_a_250k_extended_diagnostic_v0_1",
        "status": "dry_run_ready" if not execute else "running",
        "generated_utc": utc_now(),
        "repo_root": repo_root.as_posix(),
        "template_config_path": template_config.as_posix(),
        "template_config_sha256": sha256_file(template_config),
        "resolved_config_path": resolved_config.as_posix(),
        "resolved_config_sha256": sha256_file(resolved_config),
        "baseline_report": baseline,
        "dataset": dataset_result,
        "slice": slice_result,
        "run_root": run_root.as_posix(),
        "milestones": list(MILESTONES),
        "commands": phase_commands,
        "training_tokens_budget": int(cfg["max_steps"]) * int(cfg["batch_size"]) * int(cfg["seq_len"]) * int(cfg["grad_accum_steps"]),
        "capability_claim_authorized": False,
        "expansion_authorized": False,
    }
    if not execute:
        return summary

    if run_root.exists():
        if not force:
            raise StagingError(f"run output exists; use --force after review: {run_root}")
        safe_remove_run(run_root, repo_root)
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "extended_probe_report.json"

    try:
        pretrain = initial_generation(cfg, tokenizer, probe_rows)
        atomic_write_json(run_root / "generation_step_0000.json", {
            "schema_version": "1.0",
            "model_state": "deterministic_random_initialization",
            "temperature": 0.0,
            "samples": pretrain,
            "repetition_diagnostics": repeated_token_diagnostics(pretrain),
        })

        phase_results: list[dict[str, Any]] = []
        for index, command in enumerate(phase_commands, 1):
            milestone = int(command["milestone"])
            completed, seconds = run_command(command["actual_argv"], repo_root, run_root / f"phase{index}_to_{milestone}.log")
            latest_path = run_root / "latest.pt"
            checkpoint = load_checkpoint(latest_path)
            if int(checkpoint.get("step", -1)) != milestone:
                raise StagingError(f"phase {index} did not stop at milestone {milestone}")
            milestone_path = run_root / f"checkpoint_step_{milestone:04d}.pt"
            shutil.copyfile(latest_path, milestone_path)
            generated = checkpoint_generation(milestone_path, tokenizer, probe_rows)
            generation_path = run_root / f"generation_step_{milestone:04d}.json"
            atomic_write_json(generation_path, {
                "schema_version": "1.0",
                "checkpoint_step": milestone,
                "checkpoint_sha256": sha256_file(milestone_path),
                "temperature": 0.0,
                "samples": generated,
                "repetition_diagnostics": repeated_token_diagnostics(generated),
            })
            if index > 1 and f"completed step {MILESTONES[index - 2]:,}" not in completed.stdout:
                raise StagingError(f"phase {index} log does not prove auto-resume")
            phase_results.append({
                "phase": index,
                "milestone": milestone,
                "seconds": seconds,
                "log_path": (run_root / f"phase{index}_to_{milestone}.log").as_posix(),
                "checkpoint_path": milestone_path.as_posix(),
                "checkpoint_sha256": sha256_file(milestone_path),
                "generation_path": generation_path.as_posix(),
                "generation_sha256": sha256_file(generation_path),
            })

        rows = evaluation_rows(run_root / "metrics.jsonl")
        initial = point_by_step(rows, 0)
        final = point_by_step(rows, 2000)
        milestones = {str(step): point_by_step(rows, step) for step in MILESTONES}
        best_val_row = min(rows[1:], key=lambda row: float(row["val_loss"]))
        min_train_row = min(rows[1:], key=lambda row: float(row["train_loss"]))
        if float(min_train_row["train_loss"]) >= float(initial["train_loss"]):
            raise StagingError("fixed-batch training loss never decreased below step 0")
        if float(final["val_loss"]) >= float(initial["val_loss"]):
            raise StagingError("final validation loss did not improve from step 0")

        latest_path = run_root / "latest.pt"
        best_path = run_root / "best.pt"
        run_metadata = load_json(run_root / "run_metadata.json")
        token_budget = int(cfg["max_steps"]) * int(cfg["batch_size"]) * int(cfg["seq_len"]) * int(cfg["grad_accum_steps"])
        total_seconds = sum(float(item["seconds"]) for item in phase_results)
        rss_values = [row.get("peak_rss_bytes") for row in rows if isinstance(row.get("peak_rss_bytes"), int)]
        if not rss_values:
            raise StagingError("training did not record peak RSS")
        peak_rss = max(int(value) for value in rss_values)
        gap = float(final["val_loss"]) - float(final["train_loss"])
        overfit_warning = float(final["val_loss"]) > float(best_val_row["val_loss"]) + 0.15
        baseline_delta = {
            "step_500_train_loss_delta": float(milestones["500"]["train_loss"]) - baseline["final_train_loss"],
            "step_500_val_loss_delta": float(milestones["500"]["val_loss"]) - baseline["final_val_loss"],
            "final_train_loss_delta": float(final["train_loss"]) - baseline["final_train_loss"],
            "final_val_loss_delta": float(final["val_loss"]) - baseline["final_val_loss"],
            "tokens_seen_ratio": token_budget / baseline["training_tokens_seen"],
            "peak_rss_bytes_delta": peak_rss - baseline["peak_rss_bytes"],
        }
        special_ids = verify_special_tokens(tokenizer, special_tokens("code"))
        final_generation = load_json(run_root / "generation_step_2000.json")

        report = {
            **summary,
            "status": "passed",
            "completed_utc": utc_now(),
            "completed_step": 2000,
            "parameter_count": int(run_metadata["parameters"]),
            "train_tokens": int(run_metadata["train_tokens"]),
            "val_tokens": int(run_metadata["val_tokens"]),
            "tokens_per_step": int(run_metadata["tokens_per_step"]),
            "training_tokens_seen": token_budget,
            "phase_results": phase_results,
            "resume_verified": True,
            "loss_decrease_verified": True,
            "validation_loss_improved": True,
            "initial_train_loss": float(initial["train_loss"]),
            "final_train_loss": float(final["train_loss"]),
            "minimum_train_loss": float(min_train_row["train_loss"]),
            "minimum_train_loss_step": int(min_train_row["step"]),
            "initial_val_loss": float(initial["val_loss"]),
            "final_val_loss": float(final["val_loss"]),
            "best_val_loss": float(best_val_row["val_loss"]),
            "best_val_loss_step": int(best_val_row["step"]),
            "final_generalization_gap": gap,
            "overfit_warning": overfit_warning,
            "evaluation_points": rows,
            "milestone_evaluation_points": milestones,
            "total_command_seconds": total_seconds,
            "effective_tokens_per_second": token_budget / max(total_seconds, 1e-9),
            "peak_rss_bytes": peak_rss,
            "latest_checkpoint": latest_path.as_posix(),
            "latest_checkpoint_sha256": sha256_file(latest_path),
            "latest_checkpoint_bytes": latest_path.stat().st_size,
            "best_checkpoint": best_path.as_posix(),
            "best_checkpoint_sha256": sha256_file(best_path),
            "best_checkpoint_bytes": best_path.stat().st_size,
            "special_tokens_verified": True,
            "special_token_ids": special_ids,
            "generation_verified": True,
            "final_generation_repetition_diagnostics": final_generation["repetition_diagnostics"],
            "eval_leakage_count": 0,
            "stage_b_record_count": 0,
            "model_training_scope": PROBE_SCOPE,
            "work_packet_16_delta": baseline_delta,
            "capability_claim_authorized": False,
            "expansion_authorized": False,
            "decision_note": (
                "This diagnostic tests whether the same 250K corpus benefits from longer training. "
                "It does not admit new data and does not authorize the 500K probe automatically."
            ),
        }
        atomic_write_json(report_path, report)
        return report
    except BaseException as exc:
        failed = {
            **summary,
            "status": "failed",
            "completed_utc": utc_now(),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "traceback": traceback.format_exc(),
            "capability_claim_authorized": False,
            "expansion_authorized": False,
        }
        atomic_write_json(report_path, failed)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE_CONFIG)
    parser.add_argument("--resolved-config", type=Path, default=DEFAULT_RESOLVED_CONFIG)
    parser.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_probe(
        args.repo_root.resolve(),
        args.dataset_config.resolve(),
        args.template_config.resolve(),
        args.resolved_config.resolve(),
        args.approved_root.resolve(),
        args.probes.resolve(),
        args.baseline_report.resolve(),
        args.execute,
        args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
