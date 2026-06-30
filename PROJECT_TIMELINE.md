# Tiny Code Model Project Timeline

## Project Aim

Build a small, efficient, from-scratch coding model that can help with lower-level
coding tasks while Codex remains the main assistant.

The goal is not to beat large frontier models. The goal is to create a personal,
local, code-specialized model that learns from high-quality examples, understands
basic English coding instructions, follows a consistent task format, and becomes
useful for small repeatable development work.

This project values quality over speed. We will keep the model small enough to
train on the current laptop, but we will not fill it with low-quality data just
to make it look bigger.

## Why This Project Exists

Large models are powerful because they have huge training data, huge compute, and
massive engineering teams. This project is different. It is about ownership,
learning, and creating a focused assistant that can improve through careful data
and testing.

The reasons for building it are:

- Learn how language models work from tokenizer to training loop to inference.
- Create a model trained from random weights, not a fine-tune of another model.
- Build a code-focused assistant for small tasks beneath Codex-level work.
- Keep the system simple, local, inspectable, and practical on CPU.
- Make quality data, evaluation, and iteration the center of the project.
- Later, teach the model to prefer efficient code, not just code that works.

## Current Hardware Assumption

The current training target is a normal Windows laptop using CPU training.

Practical implications:

- Small models are realistic.
- Clean data matters more than huge data.
- Training runs must be short enough to inspect and repeat.
- We should avoid GPU-first tools unless the hardware changes.
- We should not start with giant model sizes, long context, MoE, RL, CUDA, ROCm,
  DeepSpeed, or complex distributed training.

## Core Design Decisions

### 1. From Scratch Means From Random Weights

The model starts with random weights. We train our own tokenizer, prepare our own
dataset, train the model, save checkpoints, and run generation.

We may later compare against open models, but the core project is a true
from-scratch training path.

### 2. Code First, Not General Chat

The model should become a tiny coding assistant.

Primary target abilities:

- Write small functions.
- Complete simple code blocks.
- Generate tests.
- Fix simple bugs.
- Explain simple errors.
- Produce Python, JavaScript, TypeScript, PowerShell, and SQL snippets.
- Follow clean coding patterns.
- Learn the user's project style over time.

### 3. English Plus Structured Task Protocol

The model does not naturally understand English. It learns English instructions
only from training examples.

We will not invent a full new language for the model. A new language would waste
capacity because the model would still need to learn English and code.

Instead, we will use English plus a compact structure:

```text
<instruction>
Write a Python function that counts words in text.
</instruction>
<constraints>
- lowercase words
- ignore simple punctuation
- return dict[str, int]
</constraints>
<answer>
def count_words(text: str) -> dict[str, int]:
    ...
</answer>
```

This gives the model clear boundaries while keeping the format readable.

### 4. Quality Before Size

We will prefer a small curated dataset over a large noisy one.

Good data:

- Correct code.
- Clear naming.
- Small functions.
- Useful tests.
- Realistic errors and fixes.
- Short explanations.
- Security-conscious backend patterns.
- Efficient solutions where appropriate.

Bad data:

- Broken random code.
- Build output.
- `node_modules`.
- Lockfiles.
- Generated files.
- Secrets or private tokens.
- Giant unrelated dumps.
- Repeated duplicates.

### 5. Evaluation Before Bigger Training

We will not judge the model only by vibes.

The project has a fixed evaluation set in:

```text
data/eval/code_prompts.jsonl
```

After training runs, we generate outputs and compare them. This lets us see if
the model is actually improving.

## What Exists Now

The current project already has:

- Python 3.12 virtual environment.
- Short Python command: `.\p`.
- A tiny Transformer model.
- Tokenizer training script.
- Dataset preparation script.
- Training script.
- Generation script.
- Code dataset collector.
- Fixed evaluation runner.
- Small curated starter code corpus.
- Quality seed corpus.
- Code smoke training config.
- Code tiny training config.

Important files:

```text
src/tinyllm/model.py
scripts/train_tokenizer.py
scripts/prepare_dataset.py
scripts/train.py
scripts/generate.py
scripts/evaluate_prompts.py
scripts/collect_code_dataset.py
data/raw/code_basics.txt
data/raw/code_quality_seed.txt
data/eval/code_prompts.jsonl
configs/code_smoke_cpu.json
configs/code_tiny_cpu.json
```

## Timeline

### Phase 0: Project Bootstrap

Status: complete.

Purpose:

Prove that the local machine can train and run a tiny Transformer from random
weights.

Completed work:

- Created project folder.
- Installed Python 3.12.
- Installed CPU-only PyTorch and supporting libraries.
- Built a small GPT-style decoder-only Transformer.
- Added tests for forward pass, loss, and generation.
- Added `.\p` shortcut so commands are short.

Success condition:

The model code imports, tests pass, and a checkpoint can be trained and loaded.

### Phase 1: Tiny Smoke Model

Status: complete.

Purpose:

Verify the full loop:

```text
raw text -> tokenizer -> token data -> training -> checkpoint -> generation
```

Completed work:

- Created `code_basics.txt`.
- Trained a tokenizer.
- Prepared token files.
- Trained `code_smoke_cpu`.
- Generated text from a checkpoint.

Result:

The model learned enough for loss to go down, but output quality was poor. This
is expected because the dataset and training run were tiny.

Lesson:

The pipeline works. The next bottleneck is data quality and instruction format,
not architecture.

### Phase 2: Quality Seed And Evaluation

Status: complete, with tokenizer foundation strengthened afterward.

Purpose:

Move from raw code continuation toward measurable coding ability.

Completed work:

- Added `code_quality_seed.txt`.
- Added fixed eval prompts in `data/eval/code_prompts.jsonl`.
- Added `scripts/evaluate_prompts.py`.
- Improved token splitting for tiny corpora.
- Ran smoke train and eval baseline.

Current result:

The model still outputs weak answers, often blank or malformed. This is normal.
It has not yet seen enough instruction-answer examples.

Lesson:

The next major need is instruction-style training data.

### Phase 2.5: Tokenizer Foundation

Status: complete for v1.

Purpose:

Pause before larger training and make sure the model has a strong, efficient way
to read code and English instructions.

Completed work:

- Added tokenizer foundation notes in `TOKENIZER_FOUNDATION.md`.
- Added protocol special tokens to tokenizer training.
- Added `max_token_length` to prevent long junk tokens.
- Added tokenizer benchmark tooling.
- Added `code_instruction_seed.txt` so tokenizer training sees the task format.
- Tested conservative and loose tokenizer variants.

Selected v1 tokenizer:

```text
byte-level BPE
target vocabulary: 4k
actual current vocabulary: about 1.4k
min_frequency: 2
max_token_length: 16
special token preset: code
```

Reason:

This tokenizer had zero unknown tokens, zero round-trip failures, atomic protocol
tags, no long junk tokens, and lower vocabulary waste than the `min_frequency=1`
variant. After improving the seed data, the stricter token-length cap also
avoided memorizing long project-specific function names as single tokens while
keeping compression nearly unchanged.

Success condition:

The tokenizer is safe, efficient enough, and does not waste much embedding
capacity. Future data growth can justify retesting 8k, 12k, or 16k vocabularies.

### Phase 3: Instruction Protocol Dataset

Status: complete for v1.

Purpose:

Teach the model that English coding requests should produce code answers.

Work to do:

- Create `data/raw/code_instruction_seed.txt`.
- Create `data/raw/code_instruction_phase3.txt`.
- Create `data/raw/code_instruction_direct.txt`.
- Use the agreed format:

```text
<instruction>
...
</instruction>
<constraints>
...
</constraints>
<answer>
...
</answer>
```

Include examples for:

- Python functions.
- JavaScript functions.
- TypeScript types and utilities.
- PowerShell filesystem scripts.
- SQL schema and query generation.
- Bug fixing.
- Unit tests.
- Short code explanations.

Success condition:

After smoke training, eval outputs should begin to resemble the requested
language and task shape, even if the code is still incomplete.

Phase 3 v1 result:

```text
instruction examples: 53
tokenizer actual vocab: about 1.8k
train tokens: 12,720
validation tokens: 670
phase3 model parameters: about 2.1M
best validation loss: 2.5843
```

The model has started learning task shapes, but it is not yet a useful coding
assistant. It still mixes languages and copies nearby patterns. This is a data
coverage issue, not a reason to skip ahead to a much larger model.

### Phase 4: First Meaningful Tiny Code Model

Status: planned.

Purpose:

Train a small but real code model on curated code and instruction data.

Likely command:

```powershell
.\p scripts/train.py --config configs/code_tiny_cpu.json
```

Model target:

- CPU-friendly.
- About 6 layers.
- 256 hidden dimension.
- 256-token context to start.
- Code-trained tokenizer.
- Random starting weights.

Data target:

- At least hundreds of high-quality instruction examples.
- Clean project code.
- Tests and explanations.
- No low-quality dumps.

Success condition:

The model can produce recognizable, structured answers for some eval prompts.
It does not need to be fully correct yet.

### Phase 5: Evaluation Loop

Status: planned.

Purpose:

Improve the model by changing data deliberately, not guessing.

After each training run:

1. Run fixed eval prompts.
2. Save outputs.
3. Inspect failures.
4. Add targeted examples for weak categories.
5. Retrain or continue training.
6. Compare outputs again.

Failure categories to track:

- Blank output.
- Wrong language.
- Syntax errors.
- Ignores constraints.
- Cannot finish function.
- Cannot write tests.
- Cannot explain bug.
- Repeats text.
- Produces insecure code.
- Produces inefficient code.

Success condition:

The model improves on the same prompts over time.

### Phase 6: User Project Style Training

Status: planned.

Purpose:

Make the model more personal and useful for the user's work.

Work to do:

- Collect selected user project folders.
- Exclude generated files, dependencies, secrets, and build output.
- Prefer the best project files over all project files.
- Add project-specific examples:
  - common component patterns
  - API handlers
  - data validation
  - CLI scripts
  - tests
  - config patterns
  - error handling style

Success condition:

The model begins matching familiar local coding patterns.

### Phase 7: Efficiency Curriculum

Status: planned reminder.

Purpose:

Teach the model from scratch to prefer efficient code.

This is important, but it should come after the model understands basic
instruction-following and code structure. If added too early, it may learn
phrases about efficiency without actually writing better code.

Efficiency topics to teach:

- Avoid unnecessary loops.
- Avoid repeated expensive computation.
- Choose appropriate data structures.
- Avoid unnecessary allocations.
- Keep I/O bounded.
- Batch work only when it helps.
- Use indexes for database queries.
- Avoid N+1 query patterns.
- Keep request handlers fast.
- Move slow external work off hot paths when correctness allows.
- Use timeouts for network calls.
- Make retryable work idempotent.
- Avoid overengineering for small problems.

Training example style:

```text
<instruction>
Improve this Python function so membership checks are efficient.
</instruction>
<bad_code>
def contains_any(values, allowed):
    for value in values:
        if value in allowed:
            return True
    return False
</bad_code>
<reasoning>
If allowed is a list, each membership check can scan the list. Converting it to
a set makes repeated membership checks faster.
</reasoning>
<answer>
def contains_any(values: list[str], allowed: list[str]) -> bool:
    allowed_set = set(allowed)
    return any(value in allowed_set for value in values)
</answer>
```

Success condition:

The model can produce simple efficient fixes and explain the practical reason.

### Phase 8: Better Local Interface

Status: planned.

Purpose:

Make the model easy to test and use.

Possible tools:

- CLI prompt runner.
- Simple local chat loop.
- Eval dashboard text report.
- Side-by-side output comparison between checkpoints.
- Small script that asks the model common coding tasks.

Success condition:

The model can be tested without manually typing long commands.

### Phase 9: Scaling Up Carefully

Status: future.

Purpose:

Increase capability only after data and evaluation are working.

Possible upgrades:

- More curated data.
- More instruction examples.
- More training steps.
- Slightly larger model.
- Longer context.
- Better tokenizer.
- More eval prompts.
- Separate datasets for train, validation, and held-out benchmark.

Avoid upgrading model size before the small model learns the basic format.

Success condition:

Model size increases only when the smaller model has reached a real limitation.

## Suggested Near-Term Order

1. Create `code_instruction_seed.txt`.
2. Rebuild tokenizer and token data.
3. Run `code_smoke_cpu`.
4. Run fixed eval prompts.
5. Inspect outputs.
6. Add targeted examples for the weakest categories.
7. Train `code_tiny_cpu`.
8. Start collecting selected user project code.
9. Add efficiency curriculum after instruction-following is visible.

## Definition Of A Useful First Model

The first useful version does not need to be brilliant.

It should be able to:

- Understand simple English coding tasks.
- Produce code in the requested language.
- Follow the task protocol.
- Write small functions that are sometimes correct.
- Generate basic tests.
- Fix very simple bugs.
- Avoid obvious unsafe patterns.
- Produce outputs that are easier to edit than starting from nothing.

## Long-Term Vision

The long-term version should become a small personal coding engine:

- Local-first.
- Code-specialized.
- Trained from scratch.
- Efficient in architecture and behavior.
- Familiar with the user's coding style.
- Useful for repetitive tasks beneath Codex.
- Measured by evals, not hype.
- Improved through carefully chosen data.

Codex remains the main engineer. This model becomes the small local apprentice:
fast to run, easy to inspect, and trained to handle simpler coding work with the
user's preferred style.

### Work Packet 14 — Representative Tokenizer Corpus

Status: tokenizer-only sample approved; tokenizer comparison completed.

- deterministically select about 500K representative proxy tokens;
- include 327 non-leaking frozen Stage B records for protocol and developer vocabulary, while measuring all 331;
- include balanced repository code/docs/tests and Wikimedia English;
- admit only stable self-knowledge facts;
- exclude all review queues, rejected rows, leakage, and cross-source duplicates;
- approve the exact sample for tokenizer use only;
- compare 2K, 4K, and 8K BPE candidates;
- remeasure Stage B fit at 128/192/256/384/512 tokens;
- keep 8K as the provisional next-probe tokenizer, pending the small Stage A training probe.

The following packet should admit the Stage A model-training corpus and run the small training ladder only after a tokenizer is selected.

### Work Packet 15 — First Stage A Model Smoke

Status: passed locally.

1. Built a deterministic 42,520-token candidate slice.
2. Reviewed and approved only the exact manifest hash.
3. Split by document family and prepared uint16 train/validation data.
4. Trained from random weights to step 30, saved, and stopped cleanly.
5. Auto-resumed the same optimizer/scheduler run to step 100.
6. Recorded fixed-batch losses, checkpoint hashes, generation samples, and failure state.
7. Review efficiency and correctness before any 250K expansion.
