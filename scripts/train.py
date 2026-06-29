from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyllm.model import TinyConfig, TinyLanguageModel, count_parameters


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tokens(path: str) -> np.memmap:
    token_path = Path(path)
    if not token_path.exists():
        raise SystemExit(f"Missing token file: {token_path}")
    return np.memmap(token_path, dtype=np.uint16, mode="r")


def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= seq_len + 1:
        raise SystemExit("Token dataset is too small for the configured sequence length")

    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = np.stack([data[int(start) : int(start) + seq_len] for start in starts])
    y = np.stack([data[int(start) + 1 : int(start) + seq_len + 1] for start in starts])
    return (
        torch.from_numpy(x.astype(np.int64)).to(device),
        torch.from_numpy(y.astype(np.int64)).to(device),
    )


def learning_rate(step: int, cfg: dict) -> float:
    max_lr = float(cfg["learning_rate"])
    min_lr = float(cfg["min_learning_rate"])
    warmup_steps = int(cfg["warmup_steps"])
    max_steps = int(cfg["max_steps"])

    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + cosine * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: TinyLanguageModel,
    train_data: np.memmap,
    val_data: np.memmap,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, data in {"train": train_data, "val": val_data}.items():
        losses = []
        for _ in range(int(cfg["eval_iters"])):
            x, y = get_batch(data, int(cfg["batch_size"]), int(cfg["seq_len"]), device)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def save_checkpoint(
    path: Path,
    model: TinyLanguageModel,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
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

    train_data = load_tokens(cfg["train_data"])
    val_data = load_tokens(cfg["val_data"])

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

    print(f"Run: {cfg['run_name']}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    print(f"Torch threads: {torch.get_num_threads()}")

    best_val_loss = float("inf")
    start_time = time.time()
    progress = tqdm(range(1, int(cfg["max_steps"]) + 1), desc="training")

    for step in progress:
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(int(cfg["grad_accum_steps"])):
            x, y = get_batch(train_data, int(cfg["batch_size"]), int(cfg["seq_len"]), device)
            _, loss = model(x, y)
            assert loss is not None
            loss = loss / int(cfg["grad_accum_steps"])
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        optimizer.step()

        progress.set_postfix(loss=f"{total_loss:.4f}", lr=f"{lr:.2e}")

        if step == 1 or step % int(cfg["eval_interval"]) == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            elapsed = time.time() - start_time
            print(
                f"step {step}: train {losses['train']:.4f}, "
                f"val {losses['val']:.4f}, elapsed {elapsed / 60:.1f}m"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, cfg, step, best_val_loss)

        if step == 1 or step % int(cfg["save_interval"]) == 0:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, cfg, step, best_val_loss)

    save_checkpoint(out_dir / "latest.pt", model, optimizer, cfg, int(cfg["max_steps"]), best_val_loss)
    print(f"Done. Latest checkpoint: {out_dir / 'latest.pt'}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
