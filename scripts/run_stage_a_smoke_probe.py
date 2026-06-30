#!/usr/bin/env python3
"""Run the two-phase Stage A smoke probe and emit a machine-checkable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyllm.model import TinyConfig, TinyLanguageModel

from stage_a_staging_common import StagingError, atomic_write_json, load_json, sha256_file, utc_now
from stage_a_smoke_common import SMOKE_SCOPE, load_jsonl, verify_special_tokens
from train_tokenizer import special_tokens
from verify_stage_a_smoke_probe import verify_dataset, verify_slice

DEFAULT_CONFIG = Path("data/foundation/processed/stage_a_smoke_probe_v0_1/train_config.json")
DEFAULT_APPROVED_ROOT = Path("data/foundation/approved/stage_a_smoke_probe_v0_1")

GENERATION_PROMPTS = (
    "A variable stores a value. ",
    "To check whether a file exists, ",
    "<instruction> Explain the next step. </instruction>\n<answer>",
)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_command(command: list[str], cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise StagingError(
            f"command failed with exit code {completed.returncode}; see {log_path}: {' '.join(command)}"
        )
    return completed


def safe_remove_run(root: Path, repo_root: Path) -> None:
    resolved = root.resolve()
    runs_root = (repo_root / "runs").resolve()
    try:
        resolved.relative_to(runs_root)
    except ValueError as exc:
        raise StagingError(f"refusing to remove a run outside {runs_root}: {resolved}") from exc
    if resolved.exists():
        shutil.rmtree(resolved)


def generate_samples(
    checkpoint_path: Path,
    tokenizer_path: Path,
    prompts: tuple[str, ...],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint["config"]
    model_cfg = TinyConfig.from_dict(cfg)
    model = TinyLanguageModel(model_cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")
    outputs: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt).ids
        if not prompt_ids:
            bos_id = tokenizer.token_to_id("<bos>")
            prompt_ids = [int(bos_id) if bos_id is not None else 0]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            eos_id=eos_id,
        )[0].tolist()
        if len(generated) <= len(prompt_ids):
            raise StagingError("generation did not append any token")
        text = tokenizer.decode(generated, skip_special_tokens=False)
        outputs.append({
            "prompt": prompt,
            "prompt_token_count": len(prompt_ids),
            "output_token_count": len(generated),
            "generated_token_count": len(generated) - len(prompt_ids),
            "text": text,
        })
    return outputs


def evaluation_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows = [row for row in load_jsonl(metrics_path) if row.get("event") == "evaluation"]
    rows.sort(key=lambda row: int(row["step"]))
    if len(rows) < 3:
        raise StagingError("smoke run did not record enough evaluation points")
    return rows


def power_shell_commands(config: Path, boundary: int) -> list[str]:
    return [
        f".\\p scripts\\train.py --config {config.as_posix().replace('/', '\\')} --stop-after-step {boundary}",
        f".\\p scripts\\train.py --config {config.as_posix().replace('/', '\\')} --auto-resume",
    ]


def run_probe(
    repo_root: Path,
    config_path: Path,
    approved_root: Path,
    resume_boundary: int,
    execute: bool,
    force: bool,
) -> dict[str, Any]:
    slice_result = verify_slice(approved_root)
    dataset_root = config_path.parent
    dataset_result = verify_dataset(dataset_root)
    cfg = load_json(config_path)
    if cfg.get("training_scope") != SMOKE_SCOPE:
        raise StagingError("resolved smoke config has the wrong training scope")
    max_steps = int(cfg["max_steps"])
    if not 1 <= resume_boundary < max_steps:
        raise StagingError(f"resume boundary must be between 1 and {max_steps - 1}")
    run_root = (repo_root / str(cfg["out_dir"])).resolve()
    tokenizer_path = Path(str(cfg["tokenizer_path"])).resolve()

    actual_commands = [
        [sys.executable, "scripts/train.py", "--config", config_path.as_posix(), "--stop-after-step", str(resume_boundary)],
        [sys.executable, "scripts/train.py", "--config", config_path.as_posix(), "--auto-resume"],
    ]
    summary = {
        "schema_version": "1.0",
        "report_id": "stage_a_smoke_probe_v0_1",
        "status": "dry_run_ready" if not execute else "running",
        "generated_utc": utc_now(),
        "repo_root": repo_root.as_posix(),
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": dataset_result["manifest_sha256"],
        "approved_slice_manifest_sha256": slice_result["manifest_sha256"],
        "provisional_tokenizer_path": tokenizer_path.as_posix(),
        "provisional_tokenizer_sha256": sha256_file(tokenizer_path),
        "provisional_tokenizer_status": "provisional_for_smoke_probe_not_final",
        "run_root": run_root.as_posix(),
        "resume_boundary_step": resume_boundary,
        "max_steps": max_steps,
        "commands": {
            "powershell": power_shell_commands(config_path, resume_boundary),
            "actual_argv": actual_commands,
        },
        "dataset": dataset_result,
        "slice": slice_result,
        "failure_handling": {
            "keyboard_interrupt": "train.py atomically saves latest.pt at the last completed step",
            "command_failure": "runner records failed status and preserves phase logs",
            "resume_mismatch": "train.py rejects changes to architecture, data, tokenizer, schedule, or optimizer-critical config",
        },
    }
    if not execute:
        return summary

    if run_root.exists():
        if not force:
            raise StagingError(f"run output exists; use --force after review: {run_root}")
        safe_remove_run(run_root, repo_root)
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "smoke_probe_report.json"

    try:
        phase1 = run_command(actual_commands[0], repo_root, run_root / "phase1.log")
        latest_path = run_root / "latest.pt"
        best_path = run_root / "best.pt"
        if not latest_path.is_file() or not best_path.is_file():
            raise StagingError("phase 1 did not save latest.pt and best.pt")
        phase1_checkpoint = load_checkpoint(latest_path)
        if int(phase1_checkpoint.get("step", -1)) != resume_boundary:
            raise StagingError("phase 1 checkpoint did not stop at the planned resume boundary")

        phase2 = run_command(actual_commands[1], repo_root, run_root / "phase2.log")
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
        best_val = min(float(row["val_loss"]) for row in rows)
        later_rows = rows[1:]
        minimum_train_row = min(later_rows, key=lambda row: float(row["train_loss"]))
        minimum_train_loss = float(minimum_train_row["train_loss"])
        loss_decrease = minimum_train_loss < float(initial["train_loss"])
        validation_improved = best_val < float(initial["val_loss"])
        if not loss_decrease:
            raise StagingError(
                "fixed-batch training loss never decreased below step 0; "
                "do not promote the tokenizer or expand the corpus"
            )
        if float(final["train_loss"]) > float(initial["train_loss"]) + 0.25:
            raise StagingError("final fixed-batch training loss shows material divergence")

        special_ids = verify_special_tokens(
            Tokenizer.from_file(str(tokenizer_path)), special_tokens("code")
        )
        generations = generate_samples(latest_path, tokenizer_path, GENERATION_PROMPTS, max_new_tokens=24)
        atomic_write_json(run_root / "generation_samples.json", {
            "schema_version": "1.0",
            "checkpoint_sha256": sha256_file(latest_path),
            "temperature": 0.0,
            "max_new_tokens": 24,
            "samples": generations,
        })

        run_metadata = load_json(run_root / "run_metadata.json")
        tokens_per_step = int(run_metadata["tokens_per_step"])
        report = {
            **summary,
            "status": "passed",
            "completed_utc": utc_now(),
            "parameter_count": int(run_metadata["parameters"]),
            "train_tokens": int(run_metadata["train_tokens"]),
            "val_tokens": int(run_metadata["val_tokens"]),
            "tokens_per_step": tokens_per_step,
            "training_tokens_seen": max_steps * tokens_per_step,
            "completed_step": max_steps,
            "phase1_completed_step": resume_boundary,
            "resume_verified": True,
            "loss_decrease_verified": True,
            "initial_train_loss": float(initial["train_loss"]),
            "final_train_loss": float(final["train_loss"]),
            "minimum_train_loss": minimum_train_loss,
            "minimum_train_loss_step": int(minimum_train_row["step"]),
            "initial_val_loss": float(initial["val_loss"]),
            "final_val_loss": float(final["val_loss"]),
            "best_val_loss": best_val,
            "validation_loss_improved": validation_improved,
            "evaluation_points": rows,
            "latest_checkpoint": latest_path.as_posix(),
            "latest_checkpoint_sha256": sha256_file(latest_path),
            "best_checkpoint": best_path.as_posix(),
            "best_checkpoint_sha256": sha256_file(best_path),
            "generation_verified": True,
            "generation_samples_path": (run_root / "generation_samples.json").as_posix(),
            "generation_samples_sha256": sha256_file(run_root / "generation_samples.json"),
            "special_tokens_verified": True,
            "special_token_ids": special_ids,
            "protocol_special_tokens_atomic": True,
            "eval_leakage_count": 0,
            "model_training_scope": SMOKE_SCOPE,
            "promotion_decision": (
                "Smoke pipeline passed. Results may inform the next bounded probe, but the 8K tokenizer "
                "remains provisional until capability/efficiency comparison is reviewed."
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
        }
        atomic_write_json(report_path, failed)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)
    parser.add_argument("--resume-boundary-step", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_probe(
        args.repo_root.resolve(), args.config.resolve(), args.approved_root.resolve(),
        args.resume_boundary_step, args.execute, args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
