# Phase 3 Status

## Status

Phase 3 v1 is complete.

The model now has a clean instruction-protocol dataset, a data quality gate, a
measured tokenizer, rebuilt token data, and a stronger validation checkpoint.

## Current Corpus

Raw training files:

```text
data/raw/code_basics.txt
data/raw/code_quality_seed.txt
data/raw/code_instruction_seed.txt
data/raw/code_instruction_phase3.txt
data/raw/code_instruction_direct.txt
```

Quality gate result:

```text
Raw files checked: 5
Instruction examples: 53
Eval prompts checked: 12
Warnings: 0
Training data quality check passed.
```

## Current Tokenizer

Current provisional tokenizer:

```text
byte-level BPE
target vocab: 4k
actual vocab: 1,823
min_frequency: 2
max_token_length: 16
special preset: code
```

Benchmark result:

```text
unknown tokens: 0
round-trip failures: 0
protocol tags: ok
long tokens: 0
embedding parameters at d_model=256: about 0.47M
```

The tokenizer is still provisional. Re-test it after adding much larger or more
diverse code data.

## Current Training Data

Prepared tokens:

```text
train tokens: 12,720
validation tokens: 670
```

## Phase 3 Validation Model

Config:

```text
configs/code_phase3_cpu.json
```

Checkpoint:

```text
runs/code_phase3_cpu/best.pt
```

Training summary:

```text
parameters: 2,121,216
best validation loss: 2.5843
best step: 500
```

## Eval Result

Eval output:

```text
runs/code_phase3_cpu/eval_outputs.jsonl
```

What improved:

- The model no longer produces only blanks.
- It learned some concrete task shapes.
- It can reproduce a correct Node health handler pattern.
- It can fix the simple Python indentation loop.
- It can generate partial PowerShell and JavaScript structures.

Remaining weaknesses:

- It mixes languages on many prompts.
- It copies nearby training patterns instead of fully solving new requests.
- It repeats fields or fragments in TypeScript outputs.
- It still needs more examples for Python file handling, tests, SQL, and explanations.
- It needs more direct instruction examples with empty constraints.

## Next Recommended Step

Move to Phase 4 only after adding a larger high-quality instruction set.

Recommended next work:

1. Add 100 to 300 more instruction examples.
2. Keep the data quality gate mandatory.
3. Add category-balanced eval scoring.
4. Train a larger CPU-friendly model only after eval quality improves.
