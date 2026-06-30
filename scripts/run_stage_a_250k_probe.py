#!/usr/bin/env python3
"""Run the bounded Stage A 250K probe and emit a machine-checkable report."""

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

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyllm.model import TinyConfig, TinyLanguageModel

from stage_a_staging_common import StagingError, atomic_write_json, load_json, sha256_file, utc_now
from stage_a_probe_common import PROBE_SCOPE, load_jsonl, verify_special_tokens
from train_tokenizer import special_tokens
from verify_stage_a_250k_probe import verify_dataset, verify_slice

DEFAULT_CONFIG = Path("data/foundation/processed/stage_a_250k_probe_v0_1/train_config.json")
DEFAULT_APPROVED_ROOT = Path("data/foundation/approved/stage_a_250k_probe_v0_1")
DEFAULT_PROBES = Path("data/eval/stage_a_250k_probe_prompts_v0_1.jsonl")
DEFAULT_SMOKE_REPORT = Path("runs/stage_a_smoke_probe_8k/smoke_probe_report.json")


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def safe_remove_run(root: Path, repo_root: Path) -> None:
    resolved = root.resolve()
    runs_root = (repo_root / "runs").resolve()
    try:
        resolved.relative_to(runs_root)
    except ValueError as exc:
        raise StagingError(f"refusing to remove a run outside {runs_root}: {resolved}") from exc
    if resolved.exists():
        shutil.rmtree(resolved)


def load_probe_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        raise StagingError("generation probe file is empty")
    ids: set[str] = set()
    categories: set[str] = set()
    for row in rows:
        row_id = str(row.get("id", ""))
        prompt = row.get("prompt")
        if not row_id or row_id in ids:
            raise StagingError("generation probes require unique non-empty IDs")
        if not isinstance(prompt, str) or not prompt.strip():
            raise StagingError(f"{row_id}: missing prompt")
        if row.get("training_allowed") is not False:
            raise StagingError(f"{row_id}: probe must remain excluded from training")
        category = str(row.get("category", ""))
        if category not in {"english", "code"}:
            raise StagingError(f"{row_id}: category must be english or code")
        max_new = int(row.get("max_new_tokens", 0))
        if not 1 <= max_new <= 64:
            raise StagingError(f"{row_id}: max_new_tokens must be in 1..64")
        ids.add(row_id)
        categories.add(category)
    if categories != {"english", "code"}:
        raise StagingError("generation probes must include English and code")
    return rows


def generate_with_model(
    model: TinyLanguageModel,
    tokenizer: Tokenizer,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model.eval()
    outputs: list[dict[str, Any]] = []
    eos_id = tokenizer.token_to_id("<eos>")
    bos_id = tokenizer.token_to_id("<bos>")
    for row in rows:
        prompt = str(row["prompt"])
        prompt_ids = tokenizer.encode(prompt).ids
        if not prompt_ids:
            prompt_ids = [int(bos_id) if bos_id is not None else 0]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        generated = model.generate(
            input_ids,
            max_new_tokens=int(row["max_new_tokens"]),
            temperature=0.0,
            top_k=0,
            eos_id=eos_id,
        )[0].tolist()
        if len(generated) <= len(prompt_ids):
            raise StagingError(f"{row['id']}: generation did not append a token")
        outputs.append({
            "id": row["id"],
            "category": row["category"],
            "prompt": prompt,
            "prompt_token_count": len(prompt_ids),
            "output_token_count": len(generated),
            "generated_token_count": len(generated) - len(prompt_ids),
            "text": tokenizer.decode(generated, skip_special_tokens=False),
        })
    return outputs


def initial_generation(cfg: dict[str, Any], tokenizer: Tokenizer, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    torch.manual_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    model = TinyLanguageModel(TinyConfig.from_dict(cfg))
    return generate_with_model(model, tokenizer, rows)


def checkpoint_generation(
    checkpoint_path: Path,
    tokenizer: Tokenizer,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    model = TinyLanguageModel(TinyConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    return generate_with_model(model, tokenizer, rows)


def evaluation_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows = [row for row in load_jsonl(metrics_path) if row.get("event") == "evaluation"]
    rows.sort(key=lambda row: int(row["step"]))
    if len(rows) < 5:
        raise StagingError("250K run did not record enough evaluation points")
    return rows


def powershell_commands(config: Path, boundary: int) -> list[str]:
    display = config.as_posix().replace("/", "\\")
    return [
        f".\\p scripts\\train.py --config {display} --stop-after-step {boundary}",
        f".\\p scripts\\train.py --config {display} --auto-resume",
    ]


def summarize_smoke_baseline(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise StagingError(f"Work Packet 15 smoke report is required for comparison: {report_path}")
    report = load_json(report_path)
    if report.get("status") != "passed":
        raise StagingError("Work Packet 15 smoke baseline did not pass")
    rows = list(report.get("evaluation_points") or [])
    boundary = int(report.get("phase1_completed_step", 30))
    boundary_row = next((row for row in rows if int(row.get("step", -1)) == boundary), None)
    final_row = rows[-1] if rows else None
    elapsed = None
    if boundary_row is not None and final_row is not None:
        elapsed = float(boundary_row.get("elapsed_seconds", 0.0)) + float(final_row.get("elapsed_seconds", 0.0))
    tokens_seen = int(report.get("training_tokens_seen", 0))
    effective_tps = tokens_seen / elapsed if elapsed and elapsed > 0 else None
    latest = Path(str(report.get("latest_checkpoint", report_path.parent / "latest.pt")))
    if not latest.is_file():
        latest = report_path.parent / "latest.pt"
    return {
        "report_path": report_path.as_posix(),
        "report_sha256": sha256_file(report_path),
        "completed_step": int(report.get("completed_step", 0)),
        "training_tokens_seen": tokens_seen,
        "initial_train_loss": float(report["initial_train_loss"]),
        "final_train_loss": float(report["final_train_loss"]),
        "initial_val_loss": float(report["initial_val_loss"]),
        "final_val_loss": float(report["final_val_loss"]),
        "loss_reduction_train": float(report["initial_train_loss"]) - float(report["final_train_loss"]),
        "loss_reduction_val": float(report["initial_val_loss"]) - float(report["final_val_loss"]),
        "effective_tokens_per_second": effective_tps,
        "peak_rss_bytes": report.get("peak_rss_bytes"),
        "checkpoint_bytes": latest.stat().st_size if latest.is_file() else None,
        "generation_verified": bool(report.get("generation_verified")),
        "special_tokens_verified": bool(report.get("special_tokens_verified")),
        "note": "Peak RSS may be null because Work Packet 15 predated native memory instrumentation.",
    }


def comparison(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def ratio(value: float | int | None, base: float | int | None) -> float | None:
        if value is None or base in (None, 0):
            return None
        return float(value) / float(base)

    return {
        "baseline": baseline,
        "current": current,
        "ratios": {
            "training_tokens_seen": ratio(current.get("training_tokens_seen"), baseline.get("training_tokens_seen")),
            "effective_tokens_per_second": ratio(current.get("effective_tokens_per_second"), baseline.get("effective_tokens_per_second")),
            "peak_rss_bytes": ratio(current.get("peak_rss_bytes"), baseline.get("peak_rss_bytes")),
            "checkpoint_bytes": ratio(current.get("checkpoint_bytes"), baseline.get("checkpoint_bytes")),
        },
        "interpretation_boundary": (
            "This comparison measures pipeline behavior and resource use. Free-generation text is qualitative evidence only; "
            "it does not authorize a capability claim."
        ),
    }


def run_probe(
    repo_root: Path,
    config_path: Path,
    approved_root: Path,
    probes_path: Path,
    smoke_report_path: Path,
    resume_boundary: int,
    execute: bool,
    force: bool,
) -> dict[str, Any]:
    slice_result = verify_slice(approved_root)
    dataset_root = config_path.parent
    dataset_result = verify_dataset(dataset_root)
    cfg = load_json(config_path)
    if cfg.get("training_scope") != PROBE_SCOPE:
        raise StagingError("resolved 250K config has the wrong training scope")
    max_steps = int(cfg["max_steps"])
    if not 1 <= resume_boundary < max_steps:
        raise StagingError(f"resume boundary must be between 1 and {max_steps - 1}")
    run_root = (repo_root / str(cfg["out_dir"])).resolve()
    tokenizer_path = Path(str(cfg["tokenizer_path"])).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    probe_rows = load_probe_rows(probes_path)
    baseline = summarize_smoke_baseline(smoke_report_path)

    actual_commands = [
        [sys.executable, "scripts/train.py", "--config", config_path.as_posix(), "--stop-after-step", str(resume_boundary)],
        [sys.executable, "scripts/train.py", "--config", config_path.as_posix(), "--auto-resume"],
    ]
    summary = {
        "schema_version": "1.0",
        "report_id": "stage_a_250k_probe_v0_1",
        "status": "dry_run_ready" if not execute else "running",
        "generated_utc": utc_now(),
        "repo_root": repo_root.as_posix(),
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": dataset_result["manifest_sha256"],
        "approved_slice_manifest_sha256": slice_result["manifest_sha256"],
        "provisional_tokenizer_path": tokenizer_path.as_posix(),
        "provisional_tokenizer_sha256": sha256_file(tokenizer_path),
        "provisional_tokenizer_status": "provisional_for_250k_probe_not_final",
        "probe_prompts_path": probes_path.as_posix(),
        "probe_prompts_sha256": sha256_file(probes_path),
        "probe_prompt_count": len(probe_rows),
        "run_root": run_root.as_posix(),
        "resume_boundary_step": resume_boundary,
        "max_steps": max_steps,
        "commands": {
            "powershell": powershell_commands(config_path, resume_boundary),
            "actual_argv": actual_commands,
        },
        "dataset": dataset_result,
        "slice": slice_result,
        "work_packet_15_baseline": baseline,
        "failure_handling": {
            "keyboard_interrupt": "train.py atomically saves latest.pt at the last completed step",
            "command_failure": "runner records failed status and preserves logs and any valid checkpoint",
            "resume_mismatch": "train.py rejects changes to architecture, data, tokenizer, schedule, or optimizer-critical config",
            "loss_or_integrity_gate_failure": "do not expand to 500K/1M; inspect the report and preserve artifacts",
        },
    }
    if not execute:
        return summary

    if run_root.exists():
        if not force:
            raise StagingError(f"run output exists; use --force after review: {run_root}")
        safe_remove_run(run_root, repo_root)
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "probe_report.json"

    try:
        pretrain = initial_generation(cfg, tokenizer, probe_rows)
        atomic_write_json(run_root / "generation_pretrain.json", {
            "schema_version": "1.0",
            "model_state": "deterministic_random_initialization",
            "seed": int(cfg["seed"]),
            "temperature": 0.0,
            "samples": pretrain,
        })

        phase1, phase1_seconds = run_command(actual_commands[0], repo_root, run_root / "phase1.log")
        latest_path = run_root / "latest.pt"
        best_path = run_root / "best.pt"
        if not latest_path.is_file() or not best_path.is_file():
            raise StagingError("phase 1 did not save latest.pt and best.pt")
        phase1_checkpoint = load_checkpoint(latest_path)
        if int(phase1_checkpoint.get("step", -1)) != resume_boundary:
            raise StagingError("phase 1 checkpoint did not stop at the planned resume boundary")

        phase2, phase2_seconds = run_command(actual_commands[1], repo_root, run_root / "phase2.log")
        if f"completed step {resume_boundary:,}" not in phase2.stdout:
            raise StagingError("phase 2 log does not prove the checkpoint was resumed")
        final_checkpoint = load_checkpoint(latest_path)
        if int(final_checkpoint.get("step", -1)) != max_steps:
            raise StagingError("final checkpoint did not reach configured max_steps")

        rows = evaluation_rows(run_root / "metrics.jsonl")
        if int(rows[0]["step"]) != 0 or int(rows[-1]["step"]) != max_steps:
            raise StagingError("metrics do not cover step 0 through the final step")
        initial = rows[0]
        final = rows[-1]
        later = rows[1:]
        minimum_train_row = min(later, key=lambda row: float(row["train_loss"]))
        best_val_row = min(later, key=lambda row: float(row["val_loss"]))
        loss_decrease = float(minimum_train_row["train_loss"]) < float(initial["train_loss"])
        validation_improved = float(best_val_row["val_loss"]) < float(initial["val_loss"])
        final_validation_improved = float(final["val_loss"]) < float(initial["val_loss"])
        if not loss_decrease:
            raise StagingError("fixed-batch training loss never decreased below step 0")
        if not validation_improved or not final_validation_improved:
            raise StagingError("validation loss did not finish below step 0; do not expand the corpus")
        if float(final["train_loss"]) > float(initial["train_loss"]) + 0.25:
            raise StagingError("final fixed-batch training loss shows material divergence")

        special_ids = verify_special_tokens(tokenizer, special_tokens("code"))
        posttrain = checkpoint_generation(latest_path, tokenizer, probe_rows)
        atomic_write_json(run_root / "generation_posttrain.json", {
            "schema_version": "1.0",
            "checkpoint_sha256": sha256_file(latest_path),
            "temperature": 0.0,
            "samples": posttrain,
        })

        run_metadata = load_json(run_root / "run_metadata.json")
        tokens_per_step = int(run_metadata["tokens_per_step"])
        training_tokens_seen = max_steps * tokens_per_step
        total_seconds = phase1_seconds + phase2_seconds
        effective_tps = training_tokens_seen / max(total_seconds, 1e-9)
        rss_values = [row.get("peak_rss_bytes") for row in rows if isinstance(row.get("peak_rss_bytes"), int)]
        if not rss_values:
            raise StagingError("training did not record peak RSS")
        peak_rss = max(int(value) for value in rss_values)
        current = {
            "completed_step": max_steps,
            "training_tokens_seen": training_tokens_seen,
            "initial_train_loss": float(initial["train_loss"]),
            "final_train_loss": float(final["train_loss"]),
            "initial_val_loss": float(initial["val_loss"]),
            "final_val_loss": float(final["val_loss"]),
            "loss_reduction_train": float(initial["train_loss"]) - float(final["train_loss"]),
            "loss_reduction_val": float(initial["val_loss"]) - float(final["val_loss"]),
            "phase1_seconds": phase1_seconds,
            "phase2_seconds": phase2_seconds,
            "total_command_seconds": total_seconds,
            "effective_tokens_per_second": effective_tps,
            "peak_rss_bytes": peak_rss,
            "checkpoint_bytes": latest_path.stat().st_size,
            "best_checkpoint_bytes": best_path.stat().st_size,
            "generation_verified": True,
            "special_tokens_verified": True,
        }
        comparison_result = comparison(current, baseline)

        report = {
            **summary,
            "status": "passed",
            "completed_utc": utc_now(),
            "parameter_count": int(run_metadata["parameters"]),
            "train_tokens": int(run_metadata["train_tokens"]),
            "val_tokens": int(run_metadata["val_tokens"]),
            "tokens_per_step": tokens_per_step,
            "training_tokens_seen": training_tokens_seen,
            "completed_step": max_steps,
            "phase1_completed_step": resume_boundary,
            "resume_verified": True,
            "loss_decrease_verified": True,
            "validation_loss_improved": True,
            "final_validation_loss_improved": True,
            "initial_train_loss": float(initial["train_loss"]),
            "final_train_loss": float(final["train_loss"]),
            "minimum_train_loss": float(minimum_train_row["train_loss"]),
            "minimum_train_loss_step": int(minimum_train_row["step"]),
            "initial_val_loss": float(initial["val_loss"]),
            "final_val_loss": float(final["val_loss"]),
            "best_val_loss": float(best_val_row["val_loss"]),
            "best_val_loss_step": int(best_val_row["step"]),
            "evaluation_points": rows,
            "phase1_seconds": phase1_seconds,
            "phase2_seconds": phase2_seconds,
            "total_command_seconds": total_seconds,
            "effective_tokens_per_second": effective_tps,
            "peak_rss_bytes": peak_rss,
            "latest_checkpoint": latest_path.as_posix(),
            "latest_checkpoint_sha256": sha256_file(latest_path),
            "latest_checkpoint_bytes": latest_path.stat().st_size,
            "best_checkpoint": best_path.as_posix(),
            "best_checkpoint_sha256": sha256_file(best_path),
            "best_checkpoint_bytes": best_path.stat().st_size,
            "generation_verified": True,
            "generation_pretrain_path": (run_root / "generation_pretrain.json").as_posix(),
            "generation_pretrain_sha256": sha256_file(run_root / "generation_pretrain.json"),
            "generation_posttrain_path": (run_root / "generation_posttrain.json").as_posix(),
            "generation_posttrain_sha256": sha256_file(run_root / "generation_posttrain.json"),
            "generation_probe_categories": sorted({str(row["category"]) for row in probe_rows}),
            "special_tokens_verified": True,
            "special_token_ids": special_ids,
            "protocol_special_tokens_atomic": True,
            "eval_leakage_count": 0,
            "stage_b_record_count": 0,
            "model_training_scope": PROBE_SCOPE,
            "work_packet_15_comparison": comparison_result,
            "capability_claim_authorized": False,
            "capability_note": (
                "Loss movement and free-generation samples may show emerging structure, but this bounded probe "
                "does not by itself establish reliable English understanding or coding capability."
            ),
            "expansion_decision": (
                "The probe passed its mechanical and loss gates. Leo must review loss, resources, and probe outputs "
                "before authorizing any 500K/1M expansion; this report does not authorize expansion automatically."
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
            "latest_checkpoint_exists": (run_root / "latest.pt").is_file(),
            "phase1_log_exists": (run_root / "phase1.log").is_file(),
            "phase2_log_exists": (run_root / "phase2.log").is_file(),
            "expansion_authorized": False,
        }
        atomic_write_json(report_path, failed)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--smoke-report", type=Path, default=DEFAULT_SMOKE_REPORT)
    parser.add_argument("--resume-boundary-step", type=int, default=150)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_probe(
        args.repo_root.resolve(),
        args.config.resolve(),
        args.approved_root.resolve(),
        args.probes.resolve(),
        args.smoke_report.resolve(),
        args.resume_boundary_step,
        args.execute,
        args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
