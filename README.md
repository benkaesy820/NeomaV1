# Tiny From-Scratch Language Model

This project trains a small GPT-style language model from random weights on a CPU.
It is designed for a normal Windows laptop: small codebase, small dependencies,
and a model size that can actually finish experiments.

## What You Need To Do Now

1. Install Python 3.12 or newer.
2. Put plain text training files in `data/raw/`.
3. Run the setup and training commands below.

Good first datasets are focused and clean:

- Your own notes or chats exported as `.txt`
- A narrow FAQ or knowledge base
- Public-domain books
- A folder of clean articles around one subject

Avoid starting with huge internet dumps. A small clean corpus is better for this
machine than a large noisy one.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

After setup, you can use the short project Python command:

```powershell
.\p --version
```

That is the same as running:

```powershell
.\.venv\Scripts\python.exe
```

## Train The Tokenizer

```powershell
.\p scripts/train_tokenizer.py --input data/raw --out data/tokenizer.json --vocab-size 8000
```

## Prepare Token Data

```powershell
.\p scripts/prepare_dataset.py --input data/raw --tokenizer data/tokenizer.json --out data/processed
```

## Quick Smoke Train

Use this first to verify that everything works.

```powershell
.\p scripts/train.py --config configs/smoke_cpu.json
```

## Train The First Real Model

This is the CPU-friendly ~7M parameter model.

```powershell
.\p scripts/train.py --config configs/tiny_cpu.json
```

## Generate Text

```powershell
.\p scripts/generate.py --checkpoint runs/tiny_cpu/latest.pt --tokenizer data/tokenizer.json --prompt "Once upon a time"
```

## Project Layout

- `src/tinyllm/model.py` contains the Transformer model.
- `scripts/train_tokenizer.py` trains an 8K byte-level BPE tokenizer.
- `scripts/prepare_dataset.py` converts text to token IDs.
- `scripts/train.py` trains from random weights.
- `scripts/generate.py` runs inference from a checkpoint.
- `configs/smoke_cpu.json` is for fast testing.
- `configs/tiny_cpu.json` is the first real training target.

The model is not fine-tuned from an existing checkpoint. Training starts from
random weights.

## Code Model Path

For a code-specialized model, collect clean source files into one corpus first.
Point this at folders containing your own projects or public code you are
allowed to use:

```powershell
.\p scripts/collect_code_dataset.py C:\path\to\your\repo --out data/raw/code_corpus.txt
```

You can pass more than one folder:

```powershell
.\p scripts/collect_code_dataset.py C:\repo1 C:\repo2 C:\repo3 --out data/raw/code_corpus.txt
```

Then train a code tokenizer and prepare code tokens:

```powershell
.\p scripts/train_tokenizer.py --input data/raw --out data/code_tokenizer.json --vocab-size 4000 --min-frequency 2 --max-token-length 16 --preset code
.\p scripts/prepare_dataset.py --input data/raw --tokenizer data/code_tokenizer.json --out data/code_processed
```

Benchmark tokenizer candidates before serious training:

```powershell
.\p scripts/benchmark_tokenizers.py --tokenizer data/code_tokenizer.json --input data/raw --eval data/eval/code_prompts.jsonl --out runs/tokenizer_benchmark.json
```

Check data quality:

```powershell
.\p scripts/check_training_data.py --raw data/raw --eval data/eval/code_prompts.jsonl
```

Synthetic instruction data from other models should go through `data/incoming`
and the importer:

```powershell
.\p scripts/import_instruction_jsonl.py data/incoming/examples.jsonl --out data/raw/imported_examples.txt
.\p scripts/check_training_data.py --raw data/raw --eval data/eval/code_prompts.jsonl
```

Smoke test:

```powershell
.\p scripts/train.py --config configs/code_smoke_cpu.json
```

Phase 3 validation train:

```powershell
.\p scripts/train.py --config configs/code_phase3_cpu.json
```

First real code model:

```powershell
.\p scripts/train.py --config configs/code_tiny_cpu.json
```

Generate code:

```powershell
.\p scripts/generate.py --checkpoint runs/code_tiny_cpu/latest.pt --tokenizer data/code_tokenizer.json --prompt "def "
```

Run the fixed coding eval prompts:

```powershell
.\p scripts/evaluate_prompts.py --checkpoint runs/code_tiny_cpu/latest.pt --tokenizer data/code_tokenizer.json --out runs/code_tiny_cpu/eval_outputs.jsonl
```
