from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
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


def raw_texts(path: Path) -> list[tuple[str, str]]:
    rows = []
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix.lower() in {".txt", ".md"}:
            rows.append((file.as_posix(), file.read_text(encoding="utf-8", errors="replace")))
    if not rows:
        raise SystemExit(f"No .txt or .md files found in {path}")
    return rows


def eval_texts(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = (
                f"<instruction>\n{item['prompt']}\n</instruction>\n"
                f"<answer>\n"
            )
            rows.append((f"{path.as_posix()}:{line_number}:{item['id']}", prompt))
    return rows


def used_token_stats(tokenizer: Tokenizer, texts: list[tuple[str, str]]) -> tuple[int, Counter[int]]:
    counts: Counter[int] = Counter()
    total = 0
    for _, text in texts:
        ids = tokenizer.encode(text).ids
        total += len(ids)
        counts.update(ids)
    return total, counts


def roundtrip_failures(tokenizer: Tokenizer, texts: list[tuple[str, str]]) -> list[str]:
    failures = []
    for name, text in texts:
        ids = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            failures.append(name)
    return failures


def unknown_count(tokenizer: Tokenizer, texts: list[tuple[str, str]]) -> int:
    unk_id = tokenizer.token_to_id("<unk>")
    if unk_id is None:
        return 0
    total = 0
    for _, text in texts:
        total += tokenizer.encode(text).ids.count(unk_id)
    return total


def special_token_report(tokenizer: Tokenizer) -> dict[str, bool]:
    report = {}
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            report[token] = False
            continue
        report[token] = tokenizer.encode(token).ids == [token_id]
    return report


def long_token_count(tokenizer: Tokenizer, max_length: int) -> int:
    vocab = tokenizer.get_vocab()
    return sum(1 for token in vocab if len(token) > max_length and token not in SPECIAL_TOKENS)


def benchmark_tokenizer(
    tokenizer_path: Path,
    train_texts: list[tuple[str, str]],
    prompt_texts: list[tuple[str, str]],
    d_model: int,
    max_token_length: int,
) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    all_texts = train_texts + prompt_texts
    train_bytes = sum(len(text.encode("utf-8")) for _, text in train_texts)
    prompt_bytes = sum(len(text.encode("utf-8")) for _, text in prompt_texts)
    train_tokens, train_counts = used_token_stats(tokenizer, train_texts)
    prompt_tokens, prompt_counts = used_token_stats(tokenizer, prompt_texts)
    used_counts = train_counts + prompt_counts
    vocab_size = tokenizer.get_vocab_size()

    return {
        "tokenizer": tokenizer_path.as_posix(),
        "vocab_size": vocab_size,
        "embedding_parameters_at_d_model": vocab_size * d_model,
        "train_bytes_per_token": round(train_bytes / max(1, train_tokens), 3),
        "prompt_bytes_per_token": round(prompt_bytes / max(1, prompt_tokens), 3),
        "train_tokens": train_tokens,
        "prompt_tokens": prompt_tokens,
        "avg_prompt_tokens": round(prompt_tokens / max(1, len(prompt_texts)), 1),
        "roundtrip_failures": roundtrip_failures(tokenizer, all_texts),
        "unknown_tokens": unknown_count(tokenizer, all_texts),
        "special_tokens_present": special_token_report(tokenizer),
        "used_vocab": len(used_counts),
        "vocab_utilization": round(len(used_counts) / max(1, vocab_size), 3),
        "tokens_seen_once": sum(1 for count in used_counts.values() if count == 1),
        "long_tokens": long_token_count(tokenizer, max_token_length),
    }


def print_table(results: list[dict]) -> None:
    headers = [
        "name",
        "vocab",
        "embedM",
        "code_bpt",
        "prompt_bpt",
        "avg_prompt",
        "used",
        "once",
        "long",
        "unk",
        "rt_fail",
        "tags",
    ]
    print("\t".join(headers))
    for row in results:
        missing_tags = [
            token for token, present in row["special_tokens_present"].items() if not present
        ]
        name = Path(row["tokenizer"]).stem
        values = [
            name,
            str(row["vocab_size"]),
            f"{row['embedding_parameters_at_d_model'] / 1_000_000:.2f}",
            str(row["train_bytes_per_token"]),
            str(row["prompt_bytes_per_token"]),
            str(row["avg_prompt_tokens"]),
            str(row["used_vocab"]),
            str(row["tokens_seen_once"]),
            str(row["long_tokens"]),
            str(row["unknown_tokens"]),
            str(len(row["roundtrip_failures"])),
            "ok" if not missing_tags else f"missing:{len(missing_tags)}",
        ]
        print("\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, action="append", required=True)
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval/code_prompts.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/tokenizer_benchmark.json"))
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--max-token-length", type=int, default=32)
    args = parser.parse_args()

    train_texts = raw_texts(args.input)
    prompt_texts = eval_texts(args.eval)
    results = [
        benchmark_tokenizer(
            path,
            train_texts,
            prompt_texts,
            args.d_model,
            args.max_token_length,
        )
        for path in args.tokenizer
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_table(results)
    print(f"Saved benchmark to {args.out}")


if __name__ == "__main__":
    main()
