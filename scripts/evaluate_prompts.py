from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyllm.model import TinyConfig, TinyLanguageModel


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def format_prompt(row: dict) -> str:
    prompt = row["prompt"]
    return f"<instruction>\n{prompt}\n</instruction>\n<constraints>\n</constraints>\n<answer>\n"


def extract_answer(text: str) -> str:
    marker = "<answer>"
    if marker in text:
        text = text.split(marker, 1)[1]
    if "</answer>" in text:
        text = text.split("</answer>", 1)[0]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("data/eval/code_prompts.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/eval_outputs.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    model = TinyLanguageModel(TinyConfig.from_dict(cfg))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = load_jsonl(args.prompts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            prompt = format_prompt(row)
            input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_id=tokenizer.token_to_id("<eos>"),
            )[0].tolist()
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            answer = extract_answer(output)
            record = {
                "id": row["id"],
                "language": row.get("language"),
                "category": row.get("category"),
                "prompt": row["prompt"],
                "output": output,
                "answer": answer,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{row['id']}]")
            print(answer if answer else output)
            print()

    print(f"Saved eval outputs to {args.out}")


if __name__ == "__main__":
    main()
