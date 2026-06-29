# NeomaV1 — Tiny From-Scratch Language Model

NeomaV1 trains a small GPT-style language model from random weights on a CPU.
It is designed for a normal Windows laptop: a small codebase, few dependencies,
and model sizes that can complete real experiments.

The project does **not** load a pretrained model. The tokenizer, model weights,
training data, checkpoints, and inference path are controlled by this project.

For the complete file-by-file engineering handoff, phase-label clarification,
and remaining decision points, read [`UPDATES.md`](UPDATES.md).

Phase 3.5B capability planning is tracked in
[`PHASE3_5B_WORK_PACKET_01.md`](PHASE3_5B_WORK_PACKET_01.md). Its held-out
evaluation suite under `data/eval/phase3_5b_heldout_v1.jsonl` is never training
data.

## Hardware target

The default configurations are aimed at a 6-core CPU and 8–16 GB RAM. The main
`tiny_cpu` model uses 6 layers, width 256, grouped-query attention, RoPE,
RMSNorm, SwiGLU, and tied embeddings.

The bundled sample corpus is only for checking the pipeline. Do not run the
20,000-step configuration until you have collected substantially more clean
training data.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use the short project Python command after setup:

```powershell
.\p --version
```

That is equivalent to:

```powershell
.\.venv\Scripts\python.exe
```

## Run the tests

```powershell
.\p -m unittest discover -s tests -v
```

## General-text model

Put clean `.txt` or `.md` files in `data/raw/`, then run:

```powershell
.\p scripts/train_tokenizer.py --input data/raw --out data/tokenizer.json --vocab-size 8000 --preset base
.\p scripts/prepare_dataset.py --input data/raw --tokenizer data/tokenizer.json --out data/processed
.\p scripts/train.py --config configs/smoke_cpu.json
```

After the smoke run succeeds:

```powershell
.\p scripts/train.py --config configs/tiny_cpu.json
```

Generate text:

```powershell
.\p scripts/generate.py --checkpoint runs/tiny_cpu/latest.pt --tokenizer data/tokenizer.json --prompt "Once upon a time"
```

## Code and instruction model

Collect code you own or are allowed to use:

```powershell
.\p scripts/collect_code_dataset.py C:\path\to\repo --out data/raw/code_corpus.txt
```

Multiple folders are supported:

```powershell
.\p scripts/collect_code_dataset.py C:\repo1 C:\repo2 C:\repo3 --out data/raw/code_corpus.txt
```

Train the code tokenizer and build a record-level split. The loss-mask option
keeps plain code/text records fully supervised while instruction records train
mainly on their `<answer>` targets.

```powershell
.\p scripts/train_tokenizer.py --input data/raw --out data/code_tokenizer.json --vocab-size 4000 --min-frequency 2 --max-token-length 16 --preset code
.\p scripts/prepare_dataset.py --input data/raw --tokenizer data/code_tokenizer.json --out data/code_processed --instruction-loss-mask
```

The dataset builder writes a `manifest.json` showing which complete records went
into training and validation. It does not split an instruction example in half.

Check the data and tokenizer:

```powershell
.\p scripts/check_training_data.py --raw data/raw --eval data/eval/code_prompts.jsonl
.\p scripts/benchmark_tokenizers.py --tokenizer data/code_tokenizer.json --input data/raw --eval data/eval/code_prompts.jsonl --out runs/tokenizer_benchmark.json
```

Synthetic instruction data from other models should go through `data/incoming`
and the importer. Do not paste outside-model output directly into `data/raw`.

```powershell
.\p scripts/import_instruction_jsonl.py data/incoming/examples.jsonl --out data/raw/imported_examples.txt
.\p scripts/check_training_data.py --raw data/raw --eval data/eval/code_prompts.jsonl
```

Run progressively larger experiments:

```powershell
.\p scripts/train.py --config configs/code_smoke_cpu.json
.\p scripts/train.py --config configs/code_probe_cpu.json
.\p scripts/train.py --config configs/code_phase3_cpu.json
```

Only after the dataset is much larger and the smaller evaluations improve:

```powershell
.\p scripts/train.py --config configs/code_tiny_cpu.json
```

## Resume an interrupted run

Training saves a resumable `latest.pt` checkpoint. Pressing `Ctrl+C` saves the
latest completed optimizer step before exiting.

Resume automatically from the run folder:

```powershell
.\p scripts/train.py --config configs/code_phase3_cpu.json --auto-resume
```

Or resume from an explicit checkpoint:

```powershell
.\p scripts/train.py --config configs/code_phase3_cpu.json --resume runs/code_phase3_cpu/latest.pt
```

## Evaluate code prompts

```powershell
.\p scripts/evaluate_prompts.py --checkpoint runs/code_phase3_cpu/best.pt --tokenizer data/code_tokenizer.json --out runs/code_phase3_cpu/eval_outputs.jsonl
```

## Project layout

- `src/tinyllm/model.py` — decoder-only Transformer and generation
- `scripts/train_tokenizer.py` — byte-level BPE tokenizer training
- `scripts/prepare_dataset.py` — record-safe splitting, token files, optional loss masks
- `scripts/train.py` — CPU training, validation, checkpoints, and resume
- `scripts/generate.py` — text generation from a checkpoint
- `scripts/evaluate_prompts.py` — fixed-prompt code evaluation
- `tests/` — causal masking, masked loss, generation, and dataset tests
- `configs/` — smoke, probe, Phase 3, and larger CPU configurations

## Important limitation

A 2M–7M parameter model trained on a laptop can learn syntax, style, local
patterns, and narrow tasks. It will not become a general ChatGPT replacement.
Capability will depend more on clean, diverse data and repeated evaluation than
on increasing the step count over a tiny corpus.
