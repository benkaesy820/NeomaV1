from __future__ import annotations

import argparse
import json
import math
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


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(data) <= seq_len + 1:
        raise SystemExit("Token dataset is too small for the configured sequence length")

    upper = len(data) - seq_len - 1
    # With answer-only masks, avoid spending an optimizer step on a batch with
    # no supervised target tokens. A small retry limit keeps sampling bounded.
    for _ in range(16):
        starts = torch.randint(0, upper, (batch_size,))
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
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    try:
        for split, data, mask in (
            ("train", train_data, train_mask),
            ("val", val_data, val_mask),
        ):
            losses = []
            for _ in range(int(cfg["eval_iters"])):
                x, y, batch_mask = get_batch(
                    data,
                    int(cfg["batch_size"]),
                    int(cfg["seq_len"]),
                    device,
                    mask,
                )
                _, loss = model(x, y, batch_mask)
                assert loss is not None
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
    finally:
        model.train(was_training)
    return out


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
            "format_version": 2,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
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
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location="cpu")


def restore_checkpoint(
    path: Path,
    model: TinyLanguageModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    checkpoint = load_checkpoint(path)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from OUT_DIR/latest.pt when it exists.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = Tokenizer.from_file(cfg["tokenizer_path"])
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

    if args.resume is not None and args.auto_resume:
        raise SystemExit("Use either --resume or --auto-resume, not both")
    resume_path = args.resume
    if args.auto_resume:
        candidate = out_dir / "latest.pt"
        if candidate.exists():
            resume_path = candidate

    completed_step = 0
    best_val_loss = float("inf")
    if resume_path is not None:
        completed_step, best_val_loss = restore_checkpoint(resume_path, model, optimizer)
        print(f"Resumed from {resume_path} at completed step {completed_step:,}")

    max_steps = int(cfg["max_steps"])
    start_step = completed_step + 1

    print(f"Run: {cfg['run_name']}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    print(f"Loss masks: {'enabled' if train_mask is not None else 'disabled'}")
    print(f"Torch threads: {torch.get_num_threads()}")

    if start_step > max_steps:
        print(f"Checkpoint already completed configured max_steps={max_steps:,}.")
        return

    start_time = time.time()
    interval_start = start_time
    interval_start_step = completed_step
    last_completed_step = completed_step
    tokens_per_step = (
        int(cfg["batch_size"]) * int(cfg["seq_len"]) * int(cfg["grad_accum_steps"])
    )
    progress = tqdm(range(start_step, max_steps + 1), desc="training", initial=completed_step, total=max_steps)

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

            if step == start_step or step % int(cfg["eval_interval"]) == 0:
                losses = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    train_mask,
                    val_mask,
                    cfg,
                    device,
                )
                elapsed = time.time() - start_time
                print(
                    f"step {step}: train {losses['train']:.4f}, "
                    f"val {losses['val']:.4f}, elapsed {elapsed / 60:.1f}m"
                )
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(out_dir / "best.pt", model, optimizer, cfg, step, best_val_loss)
                interval_start = time.time()
                interval_start_step = step

            if step == start_step or step % int(cfg["save_interval"]) == 0:
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

    save_checkpoint(out_dir / "latest.pt", model, optimizer, cfg, max_steps, best_val_loss)
    print(f"Done. Latest checkpoint: {out_dir / 'latest.pt'}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
