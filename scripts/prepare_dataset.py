from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


@dataclass(frozen=True)
class Record:
    source: str
    index: int
    text: str
    is_instruction: bool

    @property
    def record_id(self) -> str:
        return f"{self.source}#{self.index}"


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


def records_from_file(path: Path, root: Path) -> list[Record]:
    text = read_text(path)
    source = path.relative_to(root).as_posix()
    starts = [match.start() for match in re.finditer(r"<instruction(?:\s[^>]*)?>", text)]
    if not starts:
        stripped = text.strip()
        return [Record(source, 0, stripped, False)] if stripped else []

    records: list[Record] = []
    prefix = text[: starts[0]].strip()
    if prefix:
        records.append(Record(source, 0, prefix, False))

    offset = len(records)
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            records.append(Record(source, index + offset, chunk, True))
    return records


def collect_records(path: Path) -> list[Record]:
    records: list[Record] = []
    for file in find_text_files(path):
        records.extend(records_from_file(file, path))
    if not records:
        raise SystemExit(f"No non-empty records found in {path}")
    return records


def find_token(ids: list[int], token_id: int, start: int = 0) -> int | None:
    try:
        return ids.index(token_id, start)
    except ValueError:
        return None


def encode_record(
    tokenizer: Tokenizer,
    record: Record,
    eos_id: int,
    instruction_loss_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ids = tokenizer.encode(record.text).ids
    ids.append(eos_id)
    mask = np.ones(len(ids), dtype=np.uint8)

    if instruction_loss_mask and record.is_instruction:
        answer_open_id = tokenizer.token_to_id("<answer>")
        answer_close_id = tokenizer.token_to_id("</answer>")
        if answer_open_id is None or answer_close_id is None:
            raise SystemExit(
                "Instruction loss masking requires <answer> and </answer> special tokens. "
                "Train the tokenizer with --preset code."
            )

        answer_open = find_token(ids, answer_open_id)
        answer_close = (
            find_token(ids, answer_close_id, answer_open + 1) if answer_open is not None else None
        )
        if answer_open is None or answer_close is None or answer_close <= answer_open:
            raise SystemExit(f"Malformed instruction record without a complete answer: {record.record_id}")

        mask.fill(0)
        # The target tokens containing the answer body, the closing answer tag,
        # and the document EOS contribute to the supervised objective.
        mask[answer_open + 1 : answer_close + 1] = 1
        mask[-1] = 1

    return np.asarray(ids, dtype=np.uint16), mask


def split_records(
    records: list[Record],
    val_fraction: float,
    seed: int,
) -> tuple[list[Record], list[Record]]:
    if len(records) < 2:
        return records, []
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    return train_records, val_records


def write_records(
    records: list[Record],
    token_path: Path,
    mask_path: Path | None,
    tokenizer: Tokenizer,
    eos_id: int,
    instruction_loss_mask: bool,
) -> tuple[int, int]:
    total_tokens = 0
    supervised_tokens = 0
    with token_path.open("wb") as token_handle:
        mask_handle = mask_path.open("wb") if mask_path is not None else None
        try:
            for record in records:
                tokens, mask = encode_record(
                    tokenizer,
                    record,
                    eos_id,
                    instruction_loss_mask,
                )
                tokens.tofile(token_handle)
                total_tokens += int(tokens.size)
                supervised_tokens += int(mask.sum())
                if mask_handle is not None:
                    mask.tofile(mask_handle)
        finally:
            if mask_handle is not None:
                mask_handle.close()
    return total_tokens, supervised_tokens


def write_single_record_token_split(
    record: Record,
    out_dir: Path,
    tokenizer: Tokenizer,
    eos_id: int,
    val_fraction: float,
    instruction_loss_mask: bool,
) -> tuple[int, int, int, int]:
    tokens, mask = encode_record(tokenizer, record, eos_id, instruction_loss_mask)
    if tokens.size < 2:
        raise SystemExit("Dataset must contain at least two tokens")
    split_at = max(1, int(tokens.size * (1.0 - val_fraction)))
    split_at = min(split_at, tokens.size - 1)

    train_tokens = tokens[:split_at]
    val_tokens = tokens[split_at:]
    train_mask = mask[:split_at]
    val_mask = mask[split_at:]
    train_tokens.tofile(out_dir / "train.bin")
    val_tokens.tofile(out_dir / "val.bin")
    if instruction_loss_mask:
        train_mask.tofile(out_dir / "train_mask.bin")
        val_mask.tofile(out_dir / "val_mask.bin")
    return (
        int(train_tokens.size),
        int(val_tokens.size),
        int(train_mask.sum()),
        int(val_mask.sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--instruction-loss-mask",
        action="store_true",
        help="Train only on answer targets inside instruction records; plain-text records remain unmasked.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise SystemExit("Tokenizer is missing the <eos> token")
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise SystemExit("This script stores tokens as uint16; use a smaller vocabulary")

    records = collect_records(args.input)
    args.out.mkdir(parents=True, exist_ok=True)

    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    if val_records:
        train_count, train_supervised = write_records(
            train_records,
            args.out / "train.bin",
            args.out / "train_mask.bin" if args.instruction_loss_mask else None,
            tokenizer,
            eos_id,
            args.instruction_loss_mask,
        )
        val_count, val_supervised = write_records(
            val_records,
            args.out / "val.bin",
            args.out / "val_mask.bin" if args.instruction_loss_mask else None,
            tokenizer,
            eos_id,
            args.instruction_loss_mask,
        )
        split_mode = "record"
    else:
        print("WARNING: only one record was found; using a token-level validation split")
        train_count, val_count, train_supervised, val_supervised = write_single_record_token_split(
            records[0],
            args.out,
            tokenizer,
            eos_id,
            args.val_fraction,
            args.instruction_loss_mask,
        )
        split_mode = "token"

    if args.instruction_loss_mask and (train_supervised == 0 or val_supervised == 0):
        raise SystemExit(
            "The generated loss mask has no supervised tokens in one split. "
            "Add more complete instruction records or disable --instruction-loss-mask."
        )

    manifest = {
        "input": args.input.as_posix(),
        "tokenizer": args.tokenizer.as_posix(),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "split_mode": split_mode,
        "instruction_loss_mask": args.instruction_loss_mask,
        "total_records": len(records),
        "instruction_records": sum(record.is_instruction for record in records),
        "train_records": [record.record_id for record in train_records],
        "val_records": [record.record_id for record in val_records],
        "train_tokens": train_count,
        "val_tokens": val_count,
        "train_supervised_tokens": train_supervised,
        "val_supervised_tokens": val_supervised,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    print(f"Records: {len(records):,}")
    print(f"Instruction records: {manifest['instruction_records']:,}")
    print(f"Split mode: {split_mode}")
    print(f"Saved train tokens: {train_count:,}")
    print(f"Saved val tokens: {val_count:,}")
    if args.instruction_loss_mask:
        print(f"Train supervised tokens: {train_supervised:,}")
        print(f"Val supervised tokens: {val_supervised:,}")
    print(f"Output folder: {args.out}")


if __name__ == "__main__":
    main()
