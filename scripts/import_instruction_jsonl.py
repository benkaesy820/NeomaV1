from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ["id", "language", "category", "difficulty", "instruction", "answer"]
VALID_LANGUAGES = {"python", "typescript", "javascript", "powershell", "sql", "text"}
VALID_DIFFICULTIES = {"basic", "intermediate"}


def clean_id(value: str) -> str:
    item_id = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower())
    item_id = re.sub(r"_+", "_", item_id).strip("_")
    if not item_id:
        raise ValueError("id must contain at least one letter or number")
    return item_id


def require_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def optional_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when present")
    return value.strip()


def string_list(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    result = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key} must contain strings only")
        cleaned = item.strip()
        if cleaned:
            result.append(cleaned)
    return result


def validate_row(row: dict[str, Any], line_number: int, seen_ids: set[str]) -> dict[str, Any]:
    for field in REQUIRED_FIELDS:
        require_string(row, field)

    item_id = clean_id(require_string(row, "id"))
    if item_id in seen_ids:
        raise ValueError(f"duplicate id {item_id}")
    seen_ids.add(item_id)

    language = require_string(row, "language").lower()
    if language not in VALID_LANGUAGES:
        raise ValueError(f"invalid language {language}")

    difficulty = require_string(row, "difficulty").lower()
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(f"invalid difficulty {difficulty}")

    instruction = require_string(row, "instruction")
    answer = require_string(row, "answer")
    if len(instruction) > 1000:
        raise ValueError("instruction is too long")
    if len(answer) > 6000:
        raise ValueError("answer is too long")

    return {
        "id": item_id,
        "language": language,
        "category": require_string(row, "category").lower(),
        "difficulty": difficulty,
        "instruction": instruction,
        "constraints": string_list(row, "constraints"),
        "answer": answer,
        "bad_code": optional_string(row, "bad_code"),
        "reasoning": optional_string(row, "reasoning"),
        "edge_cases": string_list(row, "edge_cases"),
        "quality_notes": string_list(row, "quality_notes"),
        "line_number": line_number,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise SystemExit(f"{path}:{line_number}: row must be a JSON object")
            try:
                rows.append(validate_row(raw, line_number, seen_ids))
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise SystemExit(f"No examples found in {path}")
    return rows


def bullet_block(items: list[str]) -> str:
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def render_row(row: dict[str, Any]) -> str:
    parts = [
        f"<instruction>\n{row['instruction']}\n</instruction>",
        f"<constraints>\n{bullet_block(row['constraints'])}\n</constraints>",
    ]

    if row["bad_code"]:
        parts.append(f"<bad_code>\n{row['bad_code']}\n</bad_code>")
    if row["reasoning"]:
        parts.append(f"<reasoning>\n{row['reasoning']}\n</reasoning>")

    parts.append(f"<answer>\n{row['answer']}\n</answer>")

    if row["edge_cases"] or row["quality_notes"]:
        notes = []
        if row["edge_cases"]:
            notes.append("Edge cases:")
            notes.extend(f"- {item}" for item in row["edge_cases"])
        if row["quality_notes"]:
            notes.append("Quality notes:")
            notes.extend(f"- {item}" for item in row["quality_notes"])
        parts.append(f"<reasoning>\n{chr(10).join(notes)}\n</reasoning>")

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(render_row(row) for row in rows)
    args.out.write_text(body + "\n", encoding="utf-8", newline="\n")

    print(f"Imported examples: {len(rows)}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
