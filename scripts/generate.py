from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyllm.model import TinyConfig, TinyLanguageModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    model_cfg = TinyConfig.from_dict(cfg)
    model = TinyLanguageModel(model_cfg)
    model.load_state_dict(checkpoint["model_state"])

    if not args.tokenizer.is_file():
        raise SystemExit(f"Missing tokenizer: {args.tokenizer}")
    expected_hash = cfg.get("tokenizer_sha256")
    if expected_hash and sha256_file(args.tokenizer) != expected_hash:
        raise SystemExit("Tokenizer SHA-256 does not match the checkpoint config")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() != model_cfg.vocab_size:
        raise SystemExit(
            f"Tokenizer vocabulary {tokenizer.get_vocab_size()} does not match model vocabulary {model_cfg.vocab_size}"
        )

    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        bos_id = tokenizer.token_to_id("<bos>")
        prompt_ids = [bos_id if bos_id is not None else 0]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    eos_id = tokenizer.token_to_id("<eos>")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=eos_id,
    )[0].tolist()
    print(tokenizer.decode(output_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
