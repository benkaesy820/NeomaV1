from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyllm.model import TinyConfig, TinyLanguageModel, count_parameters


RESUME_CONFIG_KEYS = (
    "run_name",
    "tokenizer_path",
    "tokenizer_sha256",
    "train_data",
    "val_data",
    "dataset_manifest",
    "dataset_manifest_sha256",
    "seed",
    "vocab_size",
    "seq_len",
    "batch_size",
    "grad_accum_steps",
    "max_steps",
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


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())




def peak_rss_bytes() -> int | None:
    """Return peak resident memory for the current process when supported."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            process = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
            return int(counters.PeakWorkingSetSize) if ok else None

        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def validate_bound_artifacts(cfg: dict[str, Any]) -> None:
    tokenizer_path = Path(str(cfg["tokenizer_path"]))
    if not tokenizer_path.is_file():
        raise SystemExit(f"Missing tokenizer: {tokenizer_path}")
    expected_tokenizer_hash = cfg.get("tokenizer_sha256")
    if expected_tokenizer_hash and sha256_file(tokenizer_path) != expected_tokenizer_hash:
        raise SystemExit("Tokenizer SHA-256 does not match the training config")

    manifest_value = cfg.get("dataset_manifest")
    if manifest_value:
        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_file():
            raise SystemExit(f"Missing dataset manifest: {manifest_path}")
        expected_manifest_hash = cfg.get("dataset_manifest_sha256")
        if expected_manifest_hash and sha256_file(manifest_path) != expected_manifest_hash:
            raise SystemExit("Dataset manifest SHA-256 does not match the training config")
        manifest = load_config(manifest_path)
        if manifest.get("model_training_allowed") is not True:
            raise SystemExit("Dataset manifest does not grant model-training permission")
        configured_scope = cfg.get("training_scope")
        if configured_scope and manifest.get("training_scope") != configured_scope:
            raise SystemExit("Dataset manifest training scope does not match the config")
        train_path = Path(str(cfg["train_data"]))
        val_path = Path(str(cfg["val_data"]))
        if manifest.get("train_bin_sha256") and sha256_file(train_path) != manifest["train_bin_sha256"]:
            raise SystemExit("Training token file does not match the dataset manifest")
        if manifest.get("val_bin_sha256") and sha256_file(val_path) != manifest["val_bin_sha256"]:
            raise SystemExit("Validation token file does not match the dataset manifest")


def load_array(path: str | Path, dtype: np.dtype) -> np.memmap:
    array_path = Path(path)
    if not array_path.exists():
        raise SystemExit(f"Missing data file: {array_path}")
    return np.memmap(array_path, dtype=dtype, mode="r")


def optional_mask(cfg: dict[str, Any], key: str, expected_length: int) -> np.memmap | None:
    value = cfg.get(key)
    if not value:
        return None
    mask = load_array(value, np.uint8)
    if len(mask) != expected_length:
        raise SystemExit(
            f"Mask length mismatch for {value}: expected {expected_length:,}, got {len(mask):,}"
        )
    return mask


def get_batch(
    data: np.memmap,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    loss_mask: np.memmap | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(data) <= seq_len + 1:
        raise SystemExit("Token dataset is too small for the configured sequence length")

    upper = len(data) - seq_len - 1
    for _ in range(16):
        starts = torch.randint(0, upper, (batch_size,), generator=generator)
        x = np.stack([data[int(start) : int(start) + seq_len] for start in starts])
        y = np.stack([data[int(start) + 1 : int(start) + seq_len + 1] for start in starts])

        mask_tensor = None
        if loss_mask is not None:
            mask = np.stack(
                [loss_mask[int(start) + 1 : int(start) + seq_len + 1] for start in starts]
            )
            mask_tensor = torch.from_numpy(mask.astype(np.float32, copy=False)).to(device)
            if mask_tensor.sum().item() == 0:
                continue

        return (
            torch.from_numpy(x.astype(np.int64, copy=False)).to(device),
            torch.from_numpy(y.astype(np.int64, copy=False)).to(device),
            mask_tensor,
        )

    raise RuntimeError(
        "Could not sample a batch containing supervised tokens. "
        "Check the loss-mask files or reduce sequence length."
    )


def learning_rate(step: int, cfg: dict[str, Any]) -> float:
    """Return the learning rate for a one-based optimizer step."""
    max_lr = float(cfg["learning_rate"])
    min_lr = float(cfg["min_learning_rate"])
    warmup_steps = int(cfg["warmup_steps"])
    max_steps = int(cfg["max_steps"])

    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return min_lr + cosine * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: TinyLanguageModel,
    train_data: np.memmap,
    val_data: np.memmap,
    train_mask: np.memmap | None,
    val_mask: np.memmap | None,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate fixed batches so loss values are comparable across checkpoints."""
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    eval_seed = int(cfg.get("eval_seed", int(cfg["seed"]) + 100000))
    try:
        for split_index, (split, data, mask) in enumerate((
            ("train", train_data, train_mask),
            ("val", val_data, val_mask),
        )):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(eval_seed + split_index)
            losses = []
            for _ in range(int(cfg["eval_iters"])):
                x, y, batch_mask = get_batch(
                    data,
                    int(cfg["batch_size"]),
                    int(cfg["seq_len"]),
                    device,
                    mask,
                    generator=generator,
                )
                _, loss = model(x, y, batch_mask)
                assert loss is not None
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
    finally:
        model.train(was_training)
    return out


def resume_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: cfg.get(key) for key in RESUME_CONFIG_KEYS}


def save_checkpoint(
    path: Path,
    model: TinyLanguageModel,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 3,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "resume_signature": resume_signature(cfg),
            "step": step,
            "best_val_loss": best_val_loss,
            "torch_rng_state": torch.get_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing checkpoint: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_resume_signature(checkpoint: dict[str, Any], cfg: dict[str, Any]) -> None:
    stored = checkpoint.get("resume_signature")
    if stored is None:
        stored_cfg = checkpoint.get("config")
        if not isinstance(stored_cfg, dict):
            raise SystemExit("Checkpoint does not contain a valid training config")
        stored = resume_signature(stored_cfg)
    current = resume_signature(cfg)
    mismatches = [key for key in RESUME_CONFIG_KEYS if stored.get(key) != current.get(key)]
    if mismatches:
        details = ", ".join(mismatches)
        raise SystemExit(f"Resume config differs from checkpoint for: {details}")


def restore_checkpoint(
    path: Path,
    model: TinyLanguageModel,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> tuple[int, float]:
    checkpoint = load_checkpoint(path)
    validate_resume_signature(checkpoint, cfg)
    try:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not resume from {path}: {exc}") from exc

    rng_state = checkpoint.get("torch_rng_state")
    if isinstance(rng_state, torch.Tensor):
        torch.set_rng_state(rng_state)

    completed_step = int(checkpoint.get("step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return completed_step, best_val_loss


def record_evaluation(
    metrics_path: Path,
    cfg: dict[str, Any],
    step: int,
    losses: dict[str, float],
    lr: float,
    elapsed_seconds: float,
    tokens_per_step: int,
) -> None:
    append_jsonl(metrics_path, {
        "event": "evaluation",
        "run_name": cfg["run_name"],
        "step": step,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "learning_rate": lr,
        "elapsed_seconds": elapsed_seconds,
        "tokens_seen": step * tokens_per_step,
        "peak_rss_bytes": peak_rss_bytes(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from OUT_DIR/latest.pt when it exists.",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Stop cleanly after this completed step while preserving the config's scheduler horizon.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_bound_artifacts(cfg)
    tokenizer = Tokenizer.from_file(str(cfg["tokenizer_path"]))
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if int(cfg["vocab_size"]) != tokenizer_vocab_size:
        print(
            f"Using tokenizer vocab size {tokenizer_vocab_size} "
            f"instead of configured {cfg['vocab_size']}"
        )
        cfg["vocab_size"] = tokenizer_vocab_size

    torch.manual_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg["torch_threads"]))
    device = torch.device("cpu")

    train_data = load_array(cfg["train_data"], np.uint16)
    val_data = load_array(cfg["val_data"], np.uint16)
    train_mask = optional_mask(cfg, "train_loss_mask", len(train_data))
    val_mask = optional_mask(cfg, "val_loss_mask", len(val_data))

    model_cfg = TinyConfig.from_dict(cfg)
    model = TinyLanguageModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["weight_decay"]),
    )

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    if args.resume is not None and args.auto_resume:
        raise SystemExit("Use either --resume or --auto-resume, not both")
    resume_path = args.resume
    if args.auto_resume:
        candidate = out_dir / "latest.pt"
        if candidate.exists():
            resume_path = candidate

    if resume_path is None and (out_dir / "latest.pt").exists():
        raise SystemExit(
            f"A checkpoint already exists in {out_dir}. Use --resume/--auto-resume or move the run."
        )
    if resume_path is None and metrics_path.exists():
        raise SystemExit(f"Metrics already exist in {out_dir}; refusing to mix runs")

    completed_step = 0
    best_val_loss = float("inf")
    if resume_path is not None:
        completed_step, best_val_loss = restore_checkpoint(resume_path, model, optimizer, cfg)
        print(f"Resumed from {resume_path} at completed step {completed_step:,}")

    max_steps = int(cfg["max_steps"])
    if args.stop_after_step is not None:
        if args.stop_after_step < 0 or args.stop_after_step > max_steps:
            raise SystemExit(f"--stop-after-step must be in 0..{max_steps}")
        if args.stop_after_step < completed_step:
            raise SystemExit("--stop-after-step is before the resumed checkpoint step")
    effective_end = min(max_steps, args.stop_after_step) if args.stop_after_step is not None else max_steps
    start_step = completed_step + 1

    tokens_per_step = int(cfg["batch_size"]) * int(cfg["seq_len"]) * int(cfg["grad_accum_steps"])
    print(f"Run: {cfg['run_name']}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    print(f"Loss masks: {'enabled' if train_mask is not None else 'disabled'}")
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"Tokens per optimizer step: {tokens_per_step:,}")

    atomic_write_json(out_dir / "run_metadata.json", {
        "run_name": cfg["run_name"],
        "config_path": args.config.as_posix(),
        "config_sha256": sha256_file(args.config),
        "tokenizer_sha256": sha256_file(cfg["tokenizer_path"]),
        "dataset_manifest_sha256": sha256_file(cfg["dataset_manifest"]) if cfg.get("dataset_manifest") else None,
        "parameters": count_parameters(model),
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "tokens_per_step": tokens_per_step,
        "max_steps": max_steps,
        "peak_rss_bytes_at_start": peak_rss_bytes(),
    })

    if completed_step == 0:
        initial_losses = estimate_loss(model, train_data, val_data, train_mask, val_mask, cfg, device)
        best_val_loss = initial_losses["val"]
        record_evaluation(metrics_path, cfg, 0, initial_losses, 0.0, 0.0, tokens_per_step)
        print(f"step 0: train {initial_losses['train']:.4f}, val {initial_losses['val']:.4f}")
        save_checkpoint(out_dir / "best.pt", model, optimizer, cfg, 0, best_val_loss)

    if start_step > effective_end:
        print(f"Checkpoint already completed requested end step {effective_end:,}.")
        return

    start_time = time.time()
    interval_start = start_time
    interval_start_step = completed_step
    last_completed_step = completed_step
    progress = tqdm(
        range(start_step, effective_end + 1),
        desc="training",
        initial=completed_step,
        total=max_steps,
    )

    try:
        for step in progress:
            lr = learning_rate(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(int(cfg["grad_accum_steps"])):
                x, y, batch_mask = get_batch(
                    train_data,
                    int(cfg["batch_size"]),
                    int(cfg["seq_len"]),
                    device,
                    train_mask,
                )
                _, loss = model(x, y, batch_mask)
                assert loss is not None
                scaled_loss = loss / int(cfg["grad_accum_steps"])
                scaled_loss.backward()
                total_loss += scaled_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            last_completed_step = step

            elapsed_interval = max(1e-9, time.time() - interval_start)
            completed_interval = step - interval_start_step
            tokens_per_second = completed_interval * tokens_per_step / elapsed_interval
            progress.set_postfix(
                loss=f"{total_loss:.4f}",
                lr=f"{lr:.2e}",
                tok_s=f"{tokens_per_second:.0f}",
            )

            if step % int(cfg["eval_interval"]) == 0 or step == effective_end:
                losses = estimate_loss(
                    model, train_data, val_data, train_mask, val_mask, cfg, device,
                )
                elapsed = time.time() - start_time
                print(
                    f"step {step}: train {losses['train']:.4f}, "
                    f"val {losses['val']:.4f}, elapsed {elapsed / 60:.1f}m"
                )
                record_evaluation(metrics_path, cfg, step, losses, lr, elapsed, tokens_per_step)
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(out_dir / "best.pt", model, optimizer, cfg, step, best_val_loss)
                interval_start = time.time()
                interval_start_step = step

            if step % int(cfg["save_interval"]) == 0 or step == effective_end:
                save_checkpoint(out_dir / "latest.pt", model, optimizer, cfg, step, best_val_loss)

    except KeyboardInterrupt:
        save_checkpoint(
            out_dir / "latest.pt",
            model,
            optimizer,
            cfg,
            last_completed_step,
            best_val_loss,
        )
        print(f"\nInterrupted. Saved resumable checkpoint at step {last_completed_step:,}.")
        return

    save_checkpoint(out_dir / "latest.pt", model, optimizer, cfg, effective_end, best_val_loss)
    if effective_end < max_steps:
        print(
            f"Stopped as requested at step {effective_end:,}. "
            f"Resume with --auto-resume. Checkpoint: {out_dir / 'latest.pt'}"
        )
    else:
        print(f"Done. Latest checkpoint: {out_dir / 'latest.pt'}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
