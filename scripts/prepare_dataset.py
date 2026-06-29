from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def find_text_files(path: Path) -> list[Path]:
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in {".txt", ".md"}
    )
    if not files:
        raise SystemExit(f"No .txt or .md files found in {path}")
    return files


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def encode_text(tokenizer: Tokenizer, text: str, eos_id: int) -> np.ndarray:
    ids = tokenizer.encode(text).ids
    ids.append(eos_id)
    return np.asarray(ids, dtype=np.uint16)


def write_split(
    files: list[Path],
    out_path: Path,
    tokenizer: Tokenizer,
    eos_id: int,
) -> int:
    total = 0
    with out_path.open("wb") as handle:
        for file in files:
            tokens = encode_text(tokenizer, read_text(file), eos_id)
            tokens.tofile(handle)
            total += int(tokens.size)
    return total


def write_token_level_split(
    files: list[Path],
    out_dir: Path,
    tokenizer: Tokenizer,
    eos_id: int,
    val_fraction: float,
) -> tuple[int, int]:
    arrays = [encode_text(tokenizer, read_text(file), eos_id) for file in files]
    tokens = np.concatenate(arrays)
    split_at = max(1, int(tokens.size * (1.0 - val_fraction)))
    split_at = min(split_at, tokens.size - 1)
    train_tokens = tokens[:split_at]
    val_tokens = tokens[split_at:]
    train_tokens.tofile(out_dir / "train.bin")
    val_tokens.tofile(out_dir / "val.bin")
    return int(train_tokens.size), int(val_tokens.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise SystemExit("Tokenizer is missing the <eos> token")
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise SystemExit("This script stores tokens as uint16; use a smaller vocabulary")

    files = find_text_files(args.input)
    args.out.mkdir(parents=True, exist_ok=True)

    if len(files) < 20:
        train_count, val_count = write_token_level_split(
            files,
            args.out,
            tokenizer,
            eos_id,
            args.val_fraction,
        )
    else:
        rng = random.Random(args.seed)
        rng.shuffle(files)
        val_count_files = max(1, int(round(len(files) * args.val_fraction)))
        val_files = files[:val_count_files]
        train_files = files[val_count_files:]
        if not train_files:
            train_files, val_files = val_files[:-1], val_files[-1:]
        train_count = write_split(train_files, args.out / "train.bin", tokenizer, eos_id)
        val_count = write_split(val_files, args.out / "val.bin", tokenizer, eos_id)

    print(f"Saved train tokens: {train_count:,}")
    print(f"Saved val tokens: {val_count:,}")
    print(f"Output folder: {args.out}")


if __name__ == "__main__":
    main()
