from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BASE_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

CODE_SPECIAL_TOKENS = [
    "<instruction>",
    "</instruction>",
    "<constraints>",
    "</constraints>",
    "<answer>",
    "</answer>",
    "<bad_code>",
    "</bad_code>",
    "<reasoning>",
    "</reasoning>",
    "<file>",
    "</file>",
]


def text_files(path: Path) -> list[str]:
    files = sorted(
        str(file)
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in {".txt", ".md"}
    )
    if not files:
        raise SystemExit(f"No .txt or .md files found in {path}")
    return files


def special_tokens(preset: str) -> list[str]:
    if preset == "base":
        return BASE_SPECIAL_TOKENS
    if preset == "code":
        return BASE_SPECIAL_TOKENS + CODE_SPECIAL_TOKENS
    raise ValueError(f"unknown tokenizer preset: {preset}")


def build_tokenizer(
    vocab_size: int,
    min_frequency: int,
    max_token_length: int | None,
    preset: str,
) -> tuple[Tokenizer, trainers.BpeTrainer]:
    try:
        model = models.BPE(unk_token="<unk>", byte_fallback=True)
    except TypeError:
        model = models.BPE(unk_token="<unk>")

    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens(preset),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=max_token_length,
    )
    return tokenizer, trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-token-length", type=int, default=32)
    parser.add_argument("--preset", choices=["base", "code"], default="code")
    args = parser.parse_args()

    files = text_files(args.input)
    tokenizer, trainer = build_tokenizer(
        args.vocab_size,
        args.min_frequency,
        args.max_token_length,
        args.preset,
    )
    tokenizer.train(files=files, trainer=trainer)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))
    print(f"Saved tokenizer to {args.out}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Preset: {args.preset}")
    print(f"Min frequency: {args.min_frequency}")
    print(f"Max token length: {args.max_token_length}")
    print(f"Training files: {len(files)}")


if __name__ == "__main__":
    main()
