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
