# NeomaV1 Engineering Handoff and Detailed Update Record

**Snapshot date:** 2026-06-29  
**Recommended phase label:** Phase 3.5 — Engineering Foundation  
**Audience:** The model or developer continuing NeomaV1  
**Comparison baseline:** The uploaded `NeomaV1-main.zip`

## 1. Executive status

This snapshot is **not a completed Phase 4 capability release**. It is best
understood as a **Phase 3.5 engineering patch** placed between the completed
Phase 3 protocol experiment and the planned Phase 4 meaningful-model run.

The patch improves correctness, training efficiency, interruption safety,
dataset splitting, supervision targeting, evaluation reliability, and test
coverage. It deliberately does **not** add a new training corpus, synthetic
examples, pretrained weights, or a large checkpoint.

The current project still trains from random model weights. The existing raw
training files and fixed evaluation prompts are byte-for-byte unchanged from
the uploaded baseline.

### Current capability status

The bundled data remains a small pipeline-validation corpus:

- 5 raw training files
- 55 complete records after record extraction
- 53 instruction records
- 12 fixed evaluation prompts
- tokenizer target vocabulary: 4,000
- tokenizer actual vocabulary on the bundled corpus: 1,823
- prepared training tokens with the new record split: 12,728
- prepared validation tokens with the new record split: 607
- answer-supervised training tokens: 10,133
- answer-supervised validation tokens: 497

These counts are sufficient to validate the machinery. They are not sufficient
to make a generally smart coding assistant or to justify the 20,000-step
`code_tiny_cpu` run.

## 2. Why no new examples or training data were produced

Phase 3 was defined in `PROJECT_TIMELINE.md` as an **instruction protocol
validation phase**. Its success condition was that outputs begin to resemble
the requested language and task shape, even if the code remained incomplete.
The existing Phase 3 result met that limited purpose: the model stopped
producing only blank output and learned some recognizable coding structures.

Producing hundreds of examples during this engineering patch would have mixed
two different experiments:

1. whether the training and data pipeline is correct and efficient; and
2. whether a larger, better-balanced corpus improves capability.

Keeping these separate makes later comparisons trustworthy. If model behaviour
changes after this patch, the cause can be attributed to the engineering
changes rather than an undocumented data replacement.

No examples were silently generated for the following reasons:

- **Quality control:** generated examples must be reviewed for correctness,
  security, efficiency, duplication, and language consistency.
- **Balance:** the next corpus needs planned coverage across Python,
  JavaScript, TypeScript, PowerShell, SQL, tests, debugging, explanations, and
  optimization. A quick bulk generation could make one category dominate.
- **Evaluation integrity:** training examples must not duplicate or closely
  paraphrase held-out evaluation prompts.
- **Licensing and provenance:** imported code must be owned, permitted, or
  appropriately licensed, with source/provenance tracked.
- **Generalization:** repeatedly rewording the same small set of solutions would
  increase memorization without adding useful behavioural coverage.
- **Human/model decision point:** the other model should first inspect the new
  objective, record splitter, tokenizer choice, evaluation plan, and target
  model sizes before committing CPU time to a corpus design.

Therefore, the patch prepares the project to use better data efficiently, but
does not pretend that engineering changes alone make the model smart.

## 3. High-level change summary

### Added

- resumable training via `--resume` and `--auto-resume`
- interruption-safe saving when `Ctrl+C` is pressed
- atomic checkpoint replacement through a temporary file
- PyTorch RNG state in checkpoints
- optional answer-target loss masks
- complete-record dataset extraction and splitting
- deterministic dataset manifests
- shared precomputed RoPE buffers
- native grouped-query attention use when supported by PyTorch
- compatibility fallback for grouped-query attention
- GPT-style residual projection scaling at initialization
- EOS-aware generation
- exact greedy generation when `temperature=0`
- token-per-second training telemetry
- stronger configuration and tensor-shape validation
- dedicated dataset tests
- an expanded README and the original short patch summary

### Changed or improved

- instruction examples are no longer arbitrarily cut at a token boundary when
  multiple records are available
- instruction training can optimize answer targets instead of spending most of
  the objective on reproducing prompt text and protocol boilerplate
- RoPE trigonometric tables are no longer rebuilt independently in every layer
  and every forward pass
- GQA no longer always expands KV heads when native PyTorch GQA is available
- learning-rate warm-up now follows the intended one-based optimizer-step
  calculation
- generation no longer treats zero temperature as an extremely small random
  temperature
- generation and evaluation stop early when all rows produce EOS
- generation restores the model's previous train/eval mode
- tokenizer training now defaults safely to the general `base` preset; code
  workflows request the `code` preset explicitly
- checkpoint loading supports PyTorch versions with and without the
  `weights_only` argument
- model validation rejects several invalid configurations earlier

### Removed or avoided

No project files, raw examples, evaluation prompts, dependencies, model layers,
or configurations were deleted.

The patch removes or avoids these behaviours:

- repeated per-layer/per-forward RoPE cache construction
- silent use of code protocol special tokens in a general tokenizer command
- arbitrary token-level validation splitting for normal multi-record datasets
- non-resumable long CPU runs
- corrupted destination checkpoints from a partially written direct save
- unavoidable stochastic sampling when the user requests `temperature=0`
- training batches with an answer-only mask containing zero supervised targets

## 4. File-by-file implementation details

## `src/tinyllm/model.py`

### Configuration validation

`TinyConfig.validate()` now checks that vocabulary size, context length, layer
count, model width, attention-head counts, and feed-forward width are positive.
It also validates dropout and RoPE base ranges before checking divisibility
between model width and attention heads.

**Reason:** fail early with a useful message instead of reaching a tensor reshape
or attention failure deep into training.

**Still worth inspecting:** the code does not currently assert that the derived
attention head dimension is even, although the current RoPE implementation
splits it into even/odd pairs. All supplied configurations use an even head
dimension, so the current presets are safe.

### RMSNorm numerical stability

The RMS variance calculation is now performed in float32 and cast back to the
input dtype afterward.

**Reason:** this is safer if lower-precision execution is added later, while
remaining correct for the current CPU float32 path.

### Shared RoPE cache

The model constructs cosine and sine tables once for the configured maximum
sequence length. They are registered as non-persistent buffers and shared by
all Transformer blocks.

Previously, every attention layer recreated positions, inverse frequencies,
cosines, and sines during every forward call.

**Benefits:**

- less repeated CPU computation
- fewer temporary allocations
- buffers move with the model if another device is used later
- RoPE tables are excluded from checkpoints because they are deterministic and
  can be regenerated

### Grouped-query attention path

Attention now first tries PyTorch SDPA with `enable_gqa=True`. If the installed
PyTorch build rejects or cannot execute native GQA, the result is cached and the
model falls back to the previous explicit KV-head expansion.

**Benefits:** avoids unnecessary KV duplication on supported installations while
remaining compatible with older CPU PyTorch builds.

**Decision for the next model:** benchmark native GQA, fallback GQA, and ordinary
multi-head attention on the actual laptop. At this small scale, the theoretically
more efficient option is not guaranteed to have the best wall-clock speed.

### Residual projection scaling

After normal random initialization, each attention output projection and each
SwiGLU down projection is multiplied by `(2 * number_of_layers)^-0.5`.

**Reason:** control residual-stream variance as depth grows without changing the
architecture or introducing a pretrained initialization.

This still qualifies as training from scratch: all weights begin as randomly
initialized project weights.

### Optional masked language-model loss

`TinyLanguageModel.forward()` now accepts an optional `loss_mask` aligned with
`targets`. Cross-entropy is calculated per target token, multiplied by the mask,
and normalized by the number of supervised target positions.

Without a mask, behaviour remains standard full next-token prediction.

With an instruction mask:

- instruction, constraint, and answer-opening tokens are context but do not
  directly contribute to the scalar loss
- answer-body targets contribute
- the closing `</answer>` target contributes
- the final EOS target contributes
- plain code/text records remain fully supervised

Gradients still flow through the prompt context because the predicted answer is
conditioned on that context.

### Forward input checks

The model now verifies:

- `input_ids` has `[batch, sequence]` shape
- sequence length does not exceed the configured limit
- targets match the input shape
- masks are only accepted with targets
- masks match target shape

### Generation changes

Generation now uses `torch.inference_mode()`, validates generation arguments,
and supports an optional EOS token.

- `temperature=0` performs exact greedy argmax decoding
- positive temperatures retain top-k sampling
- finished batch rows stay finished
- generation exits once every row has emitted EOS
- the previous training/evaluation mode is restored afterward

This makes fixed-prompt comparisons reproducible when greedy decoding is used.
It does not add a KV cache; generation still recomputes the active context on
each token. A KV cache is a possible later inference optimization but is not
needed for correctness of the current training phase.

## `scripts/prepare_dataset.py`

### Record representation

A `Record` dataclass was added with source file, source-local index, text,
instruction flag, and a deterministic record ID such as
`code_instruction_seed.txt#15`.

### Instruction extraction

Files containing `<instruction>` tags are split into complete instruction
records at instruction boundaries. Text before the first instruction is kept as
a plain record when non-empty. A file without instruction tags is treated as
one plain-text/code record.

**Reason:** preserve example boundaries and make train/validation membership
auditable.

### Record-level splitting

All records are deterministically shuffled using the configured seed and then
split into disjoint train and validation record sets. This replaces the old
behaviour that concatenated and sliced at an arbitrary token position when the
project had fewer than 20 files.

For a dataset containing only one record, token-level splitting remains as a
last-resort fallback and emits a warning because a disjoint record split is
impossible.

### Answer-only mask generation

The new `--instruction-loss-mask` option creates `train_mask.bin` and
`val_mask.bin` as uint8 arrays aligned one-to-one with the token files.

For complete instruction records, only the answer body, closing answer tag, and
EOS are marked as supervised. Plain records receive a mask of ones and therefore
continue to use full next-token training.

The script stops with an error if required answer tags are absent, an
instruction record is malformed, or either split ends up with zero supervised
tokens.

### Dataset manifest

`manifest.json` records:

- input and tokenizer paths
- random seed and validation fraction
- split mode
- whether instruction masking was enabled
- total and instruction record counts
- exact record IDs assigned to each split
- total token counts
- supervised token counts

**Reason:** make data experiments reproducible and expose accidental leakage or
split changes.

### Important current limitations for inspection

- record splitting is deterministic but not yet stratified by language, task,
  difficulty, or source
- validation fraction is based on record count rather than token count
- near-duplicate detection is not part of this script
- a plain file is one large record, so finer code-document boundaries require a
  future explicit record format
- structural protocol validation still relies partly on the separate data
  quality checker

These are sensible candidates for the capability-focused data phase.

## `scripts/train.py`

### Generic data and mask loading

Token arrays and optional uint8 mask arrays are memory-mapped. Mask length must
exactly match its token file or training stops.

### Target-aligned mask batching

Batch masks use the same one-token shift as targets. When an answer-only mask is
enabled, sampling retries a bounded number of times to avoid performing an
optimizer step with no supervised target tokens.

**Reason:** a very sparse answer objective can otherwise waste CPU time on zero
loss batches.

### Learning-rate correction

Warm-up is calculated for one-based optimizer steps:

- first optimizer step receives `max_lr / warmup_steps`
- the final warm-up step reaches `max_lr`
- cosine progress is clamped to the valid range

The scientific-notation edits in the JSON files, such as `0.00005` to `5e-05`,
do not change those numeric values.

### Evaluation state handling

Evaluation remembers whether the model was training and restores the prior mode
in a `finally` block. It also passes masks to both train-loss and validation-loss
estimates.

### Checkpoint format and atomic saving

Checkpoints now include:

- `format_version: 2`
- model state
- optimizer state
- full configuration
- completed step
- best validation loss
- PyTorch CPU RNG state

A checkpoint is first written to a temporary file in the same directory and
then atomically replaces the target path.

**Reason:** reduce the chance of losing a long CPU run to interruption during a
checkpoint write.

### Resume support

- `--resume PATH` restores an explicit checkpoint
- `--auto-resume` restores `OUT_DIR/latest.pt` when it exists
- using both flags is rejected
- training continues from the next optimizer step
- a checkpoint already at or beyond `max_steps` exits cleanly
- model, optimizer, best loss, and PyTorch RNG are restored

Only PyTorch RNG is currently persisted. That is sufficient for the present
training sampler because batch starts use `torch.randint`. If future training
uses Python or NumPy randomness after startup, their RNG states should also be
saved.

### Interruption handling

On `KeyboardInterrupt`, the last fully completed optimizer step is saved to
`latest.pt`, and the program exits normally with a resume message.

### Throughput reporting

The progress display now reports approximate training tokens per second based
on batch size, sequence length, accumulation steps, completed optimizer steps,
and elapsed interval time.

This enables evidence-based comparison of model widths, sequence lengths,
thread counts, GQA paths, and future kernel changes on the actual laptop.

### Compatibility and experiment cautions

- checkpoint configuration compatibility is not yet deeply compared against the
  newly supplied configuration; shape mismatches are caught by state loading
- old checkpoints may load because parameter names/shapes are unchanged, but
  continuing an old run with newly enabled answer masks changes the objective
  and should be labelled as a different experiment
- evaluation currently consumes the same global PyTorch RNG stream as training;
  a separate evaluation generator could improve strict experiment isolation
- all parameters still use the same AdamW weight decay policy; norm/embedding
  parameter grouping has not been introduced

## `scripts/train_tokenizer.py`

The default tokenizer preset changed from `code` to `base`.

**Reason:** a general tokenizer command should not silently inject code protocol
special tokens. Code commands in the README explicitly pass `--preset code`, so
the code workflow remains intentional and reproducible.

No tokenizer algorithm, dependency, or existing tokenizer file was replaced.
The bundled corpus still produces an actual vocabulary of 1,823 tokens under
the current 4K target settings.

## `scripts/generate.py`

- checkpoint loading supports both newer and older PyTorch call signatures
- the tokenizer's EOS ID is passed into generation
- generation can therefore stop naturally before `max_new_tokens`

Checkpoint files should be treated as trusted inputs because standard PyTorch
checkpoint loading can deserialize Python pickle content.

## `scripts/evaluate_prompts.py`

- uses the same PyTorch checkpoint-loading compatibility path
- passes EOS into model generation
- preserves the existing fixed-prompt evaluation format and answer extraction

No evaluation prompt was added, removed, or modified.

## `configs/code_smoke_cpu.json`
## `configs/code_probe_cpu.json`
## `configs/code_phase3_cpu.json`
## `configs/code_tiny_cpu.json`

Each code configuration now points to:

- `data/code_processed/train_mask.bin`
- `data/code_processed/val_mask.bin`

This activates the answer-target objective after the documented dataset command
is run with `--instruction-loss-mask`.

The `min_learning_rate` values were only rewritten into equivalent scientific
notation. Model dimensions, context sizes, batch sizes, accumulation factors,
step counts, optimizer settings, and output directories were not increased or
otherwise changed.

The general `smoke_cpu.json` and `tiny_cpu.json` configurations remain unmasked
for ordinary full next-token pretraining.

## `tests/test_model.py`

The original forward/loss/generation test was retained and expanded. Added
checks cover:

- causal attention: changing future tokens cannot alter earlier logits
- exact masked-loss agreement with a manually selected cross-entropy slice
- RoPE caches are absent from the serialized model state
- generation restores the prior training mode
- deterministic greedy generation path
- invalid configuration rejection

## `tests/test_prepare_dataset.py` — new file

Added checks cover:

- complete instruction records remain intact
- answer-only masks select the intended target region and EOS
- record splits are deterministic and disjoint for a fixed seed

## `README.md`

The README was reorganized to distinguish:

- general-text training
- code/instruction training
- data preparation with masks
- smoke, probe, Phase 3, and large-run progression
- explicit resume commands
- fixed-prompt evaluation
- the warning that the bundled corpus is not enough for a 20,000-step run

## `PHASE3_5_ENGINEERING_PATCH.md` — short summary

This is the concise local patch note. The more accurate project-stage
interpretation is Phase 3.5 engineering, as documented here. `UPDATES.md` is
the authoritative detailed handoff.

## 5. What was intentionally left unchanged

The following remain exactly as supplied unless noted elsewhere:

- all five files under `data/raw/`
- `data/eval/code_prompts.jsonl`
- `DATA_COLLECTION_PROMPTS.md`
- `PHASE3_STATUS.md`
- `PROJECT_TIMELINE.md`
- `TOKENIZER_FOUNDATION.md`
- `requirements.txt`
- core architecture choice: decoder-only Transformer
- RMSNorm, RoPE, SwiGLU, GQA, causal SDPA, and tied embeddings
- AdamW optimizer and gradient clipping
- CPU-only training target
- uint16 token storage
- random-weight training requirement
- model-size presets and their intended progression

No pretrained model, LoRA adapter, distillation teacher, generated checkpoint,
or external dataset was added.

## 6. Verification performed on this snapshot

The following checks were rerun against the packaged code on 2026-06-29:

### Automated tests

```text
8 tests passed
```

Covered model forward/generation, causality, masked loss, RoPE serialization,
configuration validation, complete-record extraction, answer masking, and
deterministic disjoint splitting.

### Bundled-data preparation

```text
Vocabulary size: 1,823
Records: 55
Instruction records: 53
Split mode: record
Train tokens: 12,728
Validation tokens: 607
Train supervised tokens: 10,133
Validation supervised tokens: 497
```

### CPU smoke training

A temporary 30-step smoke run completed with masks enabled. A subsequent
`--auto-resume` invocation restored step 30 and correctly reported that the
configured run was already complete.

The temporary tokenizer, processed arrays, run checkpoints, and validation
artifacts were kept outside the packaged project and are not included in the
ZIP.

## 7. Decisions the continuing model should make before capability training

The next model should inspect and decide the following rather than assuming the
patch is final:

1. **Phase naming:** adopt `Phase 3.5 — Engineering Foundation`; reserve Phase 4
   for the first capability-focused dataset and meaningful model run.
2. **Objective mix:** confirm answer-only instruction loss plus full-loss raw
   code, and consider a configurable weighting ratio rather than only binary
   masks.
3. **Split strategy:** add source/category/language stratification and
   near-duplicate checks before expanding the corpus.
4. **Evaluation:** score syntax, executable tests, exact outputs, language
   selection, repetition, security, efficiency, and instruction compliance by
   category rather than relying only on validation loss.
5. **Tokenizer size:** retrain and benchmark 2K/4K/8K candidates only after the
   corpus expands; select by compression, embedding cost, throughput, and eval
   behaviour.
6. **Model scale:** benchmark several sizes and keep a larger model only when it
   yields enough behavioural improvement per CPU hour and byte of memory.
7. **Attention choice:** benchmark native GQA, fallback GQA, and MHA on the real
   laptop.
8. **Data target:** create a reviewed, provenance-tracked, balanced corpus with
   at least hundreds of genuinely distinct instruction examples plus clean raw
   code, tests, and explanations.
9. **Experiment separation:** start a fresh checkpoint for the new masked
   objective rather than presenting continuation from the old Phase 3 loss as a
   directly comparable run.
10. **Long-run gate:** do not launch `code_tiny_cpu` until smaller probe runs show
    measurable held-out behavioural gains.

## 8. Efficiency principle for future decisions

NeomaV1 should not optimize for the smallest parameter count or shortest code at
all costs. The target is the highest amount of correct, robust, generalizable
behaviour per unit of compute and memory.

A useful decision rule is:

```text
efficiency = verified behavioural quality
             / (parameters + RAM pressure + training tokens + CPU time)
```

More parameters, more data, or a more advanced implementation are justified
when controlled evaluation shows a proportionate improvement. A smaller model
that is fast but consistently wrong is not truly efficient. A somewhat larger
model that is much more reliable may have the better efficiency-to-capability
ratio.

## 9. Recommended next action

Treat this ZIP as the engineering baseline. Review the open decisions above,
then design a separate capability-data phase with a documented schema,
provenance, balance targets, deduplication, held-out evaluation, and acceptance
checks. Run the smoke and probe configurations first; promote a model/data
combination only when fixed evaluation demonstrates real generalization rather
than memorization.

## Work Packet 14

Added deterministic Stage A tokenizer-sample selection, hash-bound human approval, integrity verification, and 2K/4K/8K tokenizer comparison tooling. The representative sample remains blocked from model dataset preparation. Stable self-knowledge is explicitly separated from transient project-state statements.

Tokenizer-sample leakage handling excludes four frozen Stage B records from tokenizer training because they partially overlap protected evaluation wording. All 331 records remain in the post-training context-length benchmark.

Leo's local run adjusted the representative sample quotas to account for the intentionally small filtered TypeScript Website pool and the small stable self-knowledge allowlist. The approved tokenizer-only sample has 1,818 records and 500,104 proxy tokens. The tokenizer comparison passed hard gates for 2K, 4K, and 8K candidates, with 8K recorded as the provisional next-probe candidate.

## Work Packet 15

Added the first bounded Stage A model-training probe. The packet creates a separately reviewed 30K–48K-token slice, family-disjoint dataset preparation, fixed-batch loss logging, artifact hash binding, clean stop/resume testing, deterministic generation checks, and a machine-verifiable smoke report. The provisional 8K tokenizer remains non-final and no broad corpus expansion is authorized.

Leo's local smoke run passed. The approved slice contained 42,520 exact 8K-token IDs. The dataset had 35,720 train tokens and 6,800 validation tokens with no family overlap. Training ran from random weights to step 30, resumed to step 100, and reduced fixed-batch train loss from 9.0108 to 6.9403 and validation loss from 8.9711 to 7.2472. Generation and special-token checks passed, but this is still only a pipeline smoke pass.

## Work Packet 16

Added the bounded Stage A 250K probe: separate hash-bound admission, strict evaluation and Stage B exclusion, family-disjoint preparation, a 500-step CPU configuration, step-150 resume verification, deterministic pre/post English/code probes, native peak-RSS instrumentation, checkpoint-size and throughput reporting, and a structured comparison with Work Packet 15. The provisional 8K tokenizer remains non-final and no 500K/1M expansion is automatic.

Leo's local 250K probe passed after fixing Windows peak-RSS reporting in `scripts/train.py`. The approved slice contained 248,250 exact 8K-token IDs. The dataset had 219,731 train tokens and 28,519 validation tokens with no family overlap. Training ran from random weights to step 150, resumed to step 500, and reduced train loss from 8.9803 to 6.0542 and validation loss from 9.0019 to 6.0914. Peak RSS was 439,386,112 bytes and effective throughput was about 2,675 tokens/second. Generation probes ran but remain repetitive, so no capability claim is authorized.

## Work Packet 17

Added and executed the 250K extended diagnostic locally. The run kept the same approved 248,250-token Stage A slice, same provisional 8K tokenizer, and same 3,307,200-parameter model, then trained a separate run to 2,000 steps.

The diagnostic passed: train loss 8.9803 -> 4.8433, validation loss 9.0019 -> 5.1575, best validation loss 5.1336 at step 1900, peak RSS 439,472,128 bytes, throughput about 2,886 tokens/second, zero Stage B records, zero protected-evaluation leakage, and milestone checkpoints/generation at 500, 1000, 1500, and 2000.

Generation remains shallow and repetitive, so the run supports better Stage A scaling but not a capability claim. The next data move should be a reviewed 500K Stage A corpus with stronger English/developer-language balance and the same diagnostic gates.
