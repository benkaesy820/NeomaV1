# Tokenizer Foundation

## Goal

Build a tokenizer that lets the tiny code model learn efficiently without
wasting model capacity.

For this project, a strong tokenizer must:

- Represent any code or English instruction without unknown tokens.
- Preserve case, indentation, punctuation, symbols, and newlines.
- Keep protocol tags atomic, such as `<instruction>` and `<answer>`.
- Avoid giant one-off tokens from repeated strings or repository-specific names.
- Compress code enough to make training efficient.
- Stay small enough that embeddings do not consume the whole tiny model.

## Current Direction

The recommended first tokenizer family is:

```text
byte-level BPE trained on code + English coding instructions
```

This gives us a practical balance:

- Byte-level fallback protects against out-of-vocabulary text.
- BPE keeps sequences much shorter than raw bytes or characters.
- Code-domain training improves compression on source code.
- A modest vocabulary keeps the tiny model's embedding table affordable.

## Special Protocol Tokens

The tokenizer should always preserve these as single tokens:

```text
<pad>
<bos>
<eos>
<unk>
<instruction>
</instruction>
<constraints>
</constraints>
<answer>
</answer>
<bad_code>
</bad_code>
<reasoning>
</reasoning>
<file>
</file>
```

These tags are part of the training language. They help the model learn where a
task starts, where constraints live, and where the answer begins.

## Vocabulary Size Targets

We will test:

```text
4k, 8k, 12k, 16k
```

For a model with `d_model = 256`, embedding cost is:

```text
4k vocab  ~= 1.0M parameters
8k vocab  ~= 2.0M parameters
12k vocab ~= 3.1M parameters
16k vocab ~= 4.1M parameters
```

The likely first real choice is 8k or 12k. Bigger is not automatically better
because this model is intentionally tiny.

## Current V1 Choice

After improving the seed data and re-running the benchmark, the current
provisional tokenizer is:

```text
byte-level BPE
target vocabulary: 4k
actual current vocabulary: about 1.5k
min_frequency: 2
max_token_length: 16
special token preset: code
```

After Phase 3 v1, the actual vocabulary is about 1.8k on the current five-file
seed corpus. The same decision still holds: keep `max_token_length = 16` and
`min_frequency = 2` until the corpus is much larger.

The stricter token-length cap prevents the tokenizer from memorizing long
project-specific names as single tokens. That is better for this tiny model
because it encourages compositional learning while keeping compression nearly
the same.

## Benchmark Criteria

Before training a serious model, compare candidate tokenizers on:

- Round-trip decode correctness.
- Unknown-token count.
- Bytes per token on raw code.
- Bytes per token on English instruction prompts.
- Average eval prompt length.
- Vocabulary utilization.
- Long-token count.
- Atomic protocol tag handling.
- Embedding parameter cost.

## Decision Rule

Pick the smallest tokenizer that:

- Has safe round-tripping.
- Has zero unknown tokens.
- Keeps all protocol tags atomic.
- Compresses code and prompts well enough.
- Avoids obvious junk tokens.
- Leaves enough model capacity for actual learning.

If ordinary byte-level BPE fails these checks, then we can consider a custom
source-aware BPE later.

## 2026 Stage A Production Comparison

This section supersedes earlier provisional 4K/8K/12K/16K guidance written for the tiny seed corpus.

The production comparison is:

```text
2K byte-level BPE
4K byte-level BPE
8K byte-level BPE
```

All candidates use the complete code-protocol special-token set, `min_frequency=2`, and `max_token_length=32` over a reviewed approximately 500K-token representative corpus. Evaluation text is excluded from tokenizer training.

For the `code_phase3_cpu` architecture, approximate total parameter counts are:

```text
2K vocab: 2,155,200
4K vocab: 2,539,200
8K vocab: 3,307,200
```

Hard gates are zero round-trip failures, zero unknown tokens, and atomic protocol tags. The decision must also consider code and English compression, vocabulary utilization, Stage B fit at 256 tokens, and the parameter cost. A provisional static comparison is not enough to finalize the tokenizer; the selected candidate must also survive the planned small Stage A probe.

Leo's local comparison passed the hard gates for all three candidates and selected 8K as the provisional next-probe tokenizer because it gave the strongest frozen Stage B context fit. The choice remains provisional until the small Stage A probe confirms the capability and parameter trade-off.

## Provisional 8K Stage A Smoke Gate

The Work Packet 14 8K tokenizer is used in Work Packet 15 only as a provisional probe candidate. Its exact SHA-256 is bound into the smoke slice, prepared dataset, resolved config, checkpoint, and run report. Passing the smoke run proves artifact compatibility and trainability; it does not finalize the vocabulary. Final promotion requires review of loss behavior, CPU cost, memory, context fit, and later bounded capability probes.

## Provisional 8K use in Work Packet 16

The 8K byte-level BPE remains provisional during the 250K Stage A probe. A successful run may support continued use, but tokenizer finalization requires review of compression, parameter cost, speed, memory, real instruction lengths, loss behavior, and probe results.

