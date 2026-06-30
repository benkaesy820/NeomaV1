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

## Stage A Work Packet 14 — Tokenizer Sample Admission

Baseline: `35e6d17`.

The Wikimedia English filtering pass produced 33,110 clean candidates, 2,625 review rows, about 12M proxy tokens, and 17,495 families. Packet 14 adds deterministic selection and review tooling for an approximately 500K-token representative tokenizer corpus.

The corpus was approved for tokenizer comparison only. Generic and model-training permission remain false. The completed comparison covered 2K versus 4K versus 8K byte-level BPE, with real context-length measurement for all 331 frozen instruction records.

The 8K tokenizer is the provisional next-probe candidate because it gave the best Stage B context fit. No processed model dataset or model training run has been completed by this packet.

## Stage A Work Packet 15 — Smoke Training Probe

Status: passed locally.

- Baseline: `32d3ef4`.
- Derived and approved a separate 42,520 exact-token Stage A slice from non-Stage-B approved sample records.
- Rechecked protected-evaluation leakage and preserved source/family provenance.
- Prepared a family-disjoint full-loss dataset with 35,720 train tokens and 6,800 validation tokens.
- Ran a 100-step, 3,307,200-parameter CPU smoke probe with a stop/resume boundary at step 30.
- Verified fixed-batch loss decrease: train 9.0108 -> 6.9403 and validation 8.9711 -> 7.2472.
- Verified checkpoint hashes, resume integrity, deterministic generation, and special-token survival.
- Do not promote the tokenizer or expand the corpus based on pipeline success alone.

## Stage A Work Packet 16 — 250K Probe

- Baseline: `321f0f2`.
- Status: passed locally.
- Approved a separately reviewed 248,250 exact-token Stage A slice.
- Kept frozen Stage B instructions and protected evaluations excluded.
- Used the provisional 8K tokenizer and the 3,307,200-parameter model.
- Ran 500 steps / 256,000 tokens seen with a verified step-150 resume boundary.
- Fixed Windows peak-RSS recording and measured peak RSS at 439,386,112 bytes.
- Measured effective throughput at about 2,675 tokens/second.
- Reduced fixed-batch train loss from 8.9803 to 6.0542 and validation loss from 9.0019 to 6.0914.
- Deterministic pre/post generation ran, but output remains repetitive and does not authorize a capability claim.
- No 500K/1M expansion is authorized by this packet.

## Stage A Work Packet 17 - 250K Extended Diagnostic

- Baseline: `22ad99c`.
- Status: passed locally.
- Kept the same approved 248,250-token Stage A slice and the same provisional 8K tokenizer.
- Ran a separate 2,000-step diagnostic in `runs/stage_a_250k_extended_8k`.
- Saved milestone checkpoints and deterministic generation samples at steps 500, 1000, 1500, and 2000.
- Processed 1,024,000 tokens with peak RSS at 439,472,128 bytes and throughput around 2,886 tokens/second.
- Reduced train loss from 8.9803 to 4.8433 and validation loss from 9.0019 to 5.1575.
- Best validation loss was 5.1336 at step 1900; the simple overfit gate did not trigger.
- Free generation is still repetitive and malformed in places, especially English continuation and mixed-language code.
- No capability claim or automatic 500K/1M expansion is authorized by this packet.
