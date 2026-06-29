from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PAIR_TAGS = [
    "instruction",
    "constraints",
    "answer",
    "bad_code",
    "reasoning",
    "file",
]

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"][^'\"]{16,}['\"]"),
    re.compile(r"\bghp_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def raw_files(path: Path) -> list[Path]:
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in {".txt", ".md"}
    )
    if not files:
        raise SystemExit(f"No .txt or .md files found in {path}")
    return files


def tag_blocks(text: str, tag: str) -> list[re.Match[str]]:
    return list(re.finditer(rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", text, re.DOTALL))


def check_balanced_tags(path: Path, text: str, errors: list[str]) -> None:
    for tag in PAIR_TAGS:
        opens = len(re.findall(rf"<{tag}(?:\s[^>]*)?>", text))
        closes = text.count(f"</{tag}>")
        if opens != closes:
            errors.append(f"{path}: unbalanced <{tag}> tags: {opens} open, {closes} close")


def check_protocol_examples(path: Path, text: str, errors: list[str], warnings: list[str]) -> list[str]:
    instructions: list[str] = []
    starts = [match.start() for match in re.finditer(r"<instruction>", text)]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunk = text[start:end]
        instruction_match = re.search(r"<instruction>(.*?)</instruction>", chunk, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", chunk, re.DOTALL)
        if instruction_match is None:
            errors.append(f"{path}: malformed instruction block near byte {start}")
            continue
        instruction = instruction_match.group(1).strip()
        instructions.append(instruction)
        if not instruction:
            errors.append(f"{path}: empty instruction near byte {start}")
        if answer_match is None:
            errors.append(f"{path}: instruction has no answer: {instruction[:80]}")
            continue
        answer = answer_match.group(1).strip()
        if not answer:
            errors.append(f"{path}: empty answer for instruction: {instruction[:80]}")
        if len(answer) > 5000:
            warnings.append(f"{path}: long answer over 5000 characters: {instruction[:80]}")
        if chunk.find("<answer>") < chunk.find("</instruction>"):
            errors.append(f"{path}: answer appears before instruction closes: {instruction[:80]}")
        has_bad_code = "<bad_code>" in chunk
        has_reasoning = "<reasoning>" in chunk
        if has_bad_code and not has_reasoning:
            warnings.append(f"{path}: bad_code example has no reasoning: {instruction[:80]}")
    return instructions


def check_file_blocks(path: Path, text: str, errors: list[str], warnings: list[str]) -> None:
    for match in tag_blocks(text, "file"):
        opening = re.match(r"<file\s+path=\"([^\"]+)\">", match.group(0))
        if opening is None:
            errors.append(f"{path}: <file> block is missing path attribute near byte {match.start()}")
        body = match.group(1).strip()
        if not body:
            errors.append(f"{path}: empty <file> block near byte {match.start()}")
        if len(body) > 8000:
            warnings.append(f"{path}: long <file> block over 8000 characters near byte {match.start()}")


def check_secrets(path: Path, text: str, errors: list[str]) -> None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            errors.append(f"{path}: possible secret matched pattern {pattern.pattern}")


def load_eval(path: Path, errors: list[str]) -> list[dict]:
    prompts = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                continue
            for key in ["id", "language", "category", "prompt"]:
                if key not in row or not isinstance(row[key], str) or not row[key].strip():
                    errors.append(f"{path}:{line_number}: missing non-empty string field {key}")
            row_id = row.get("id")
            if isinstance(row_id, str):
                if row_id in seen_ids:
                    errors.append(f"{path}:{line_number}: duplicate eval id {row_id}")
                seen_ids.add(row_id)
            prompts.append(row)
    return prompts


def check_eval_leakage(raw_text: str, eval_rows: list[dict], errors: list[str]) -> None:
    normalized_raw = normalize(raw_text)
    for row in eval_rows:
        prompt = row.get("prompt")
        if isinstance(prompt, str) and normalize(prompt) in normalized_raw:
            errors.append(f"eval prompt leaked into training data exactly: {row.get('id')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval/code_prompts.jsonl"))
    parser.add_argument("--max-duplicate-instruction", type=int, default=1)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    all_instructions: list[str] = []
    combined_raw = []

    for path in raw_files(args.raw):
        text = path.read_text(encoding="utf-8", errors="replace")
        combined_raw.append(text)
        check_balanced_tags(path, text, errors)
        check_file_blocks(path, text, errors, warnings)
        all_instructions.extend(check_protocol_examples(path, text, errors, warnings))
        check_secrets(path, text, errors)

    eval_rows = load_eval(args.eval, errors)
    check_eval_leakage("\n".join(combined_raw), eval_rows, errors)

    normalized_counts = Counter(normalize(instruction) for instruction in all_instructions)
    for instruction, count in normalized_counts.items():
        if instruction and count > args.max_duplicate_instruction:
            errors.append(f"duplicate instruction appears {count} times: {instruction[:100]}")

    print(f"Raw files checked: {len(raw_files(args.raw))}")
    print(f"Instruction examples: {len(all_instructions)}")
    print(f"Eval prompts checked: {len(eval_rows)}")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("Training data quality check passed.")


if __name__ == "__main__":
    main()
