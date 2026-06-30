# Phase 3.5B Work Packet 15 — Stage A Smoke Training Probe

Baseline: `32d3ef4`

## Purpose

This packet starts the first real Neoma model-training test, but keeps it deliberately small. It admits one separately reviewed Stage A slice of **30K–48K exact provisional-8K tokens**, prepares a family-disjoint train/validation dataset, and runs a **100-step CPU smoke probe** from random weights.

The purpose is pipeline proof, not capability claims and not final tokenizer promotion.

## Frozen boundaries

- No new broad instruction batch.
- No 250K/500K model-training expansion.
- The 8K tokenizer remains **provisional**.
- Frozen Stage B instruction records are excluded from Stage A model training.
- Evaluation text is excluded and fingerprint-bound.
- Stage A uses ordinary next-token loss; answer-only masks remain disabled.
- The derived slice is approved only for `stage_a_smoke_probe_v0_1_only`.

## Why the tokenizer-only sample is not used directly

Work Packet 14 approved its 500K sample only for tokenizer comparison. Its manifest explicitly keeps model training forbidden. Packet 15 therefore derives a separate small slice from the non-Stage-B portions, rechecks leakage and hashes, writes a new candidate, and requires a new Leo review bound to the exact candidate-manifest SHA-256.

The original tokenizer sample remains unchanged and model-training-forbidden.

## Admitted-slice target

The exact count is measured with the provisional 8K tokenizer and includes one `<eos>` token per record.

| Component | Target | Allowed range |
|---|---:|---:|
| Repository code/docs/tests | ~22K | 16K–28K |
| Wikimedia English | ~16K | 12K–21K |
| Stable Neoma self-knowledge | ~2K | 0.3K–4K |
| **Total** | **~40K** | **30K–48K** |

Selection is deterministic and applies source quotas, family caps, exact hashing, special-token checks, and a fresh comparison against every protected evaluation field. `frozen_stage_b` is an explicit forbidden group.

## Model configuration

Tracked template: `configs/stage_a_smoke_probe_8k_cpu.json`

| Field | Value |
|---|---:|
| Vocabulary | provisional 8K BPE |
| Parameters | 3,307,200 |
| Layers | 4 |
| Model width | 192 |
| Query/KV heads | 4 / 2 |
| FFN width | 576 |
| Context | 128 |
| Batch / accumulation | 2 / 2 |
| Tokens per optimizer step | 512 |
| Maximum steps | 100 |
| Maximum tokens seen | 51,200 |
| Planned resume boundary | step 30 |
| Objective | full next-token loss |

The model starts from random weights. It does not load a pretrained checkpoint.

## Added files

```text
configs/stage_a_smoke_probe_8k_cpu.json

data/foundation/manifests/stage_a_smoke_probe_v0_1_plan.json
data/reviews/stage_a_smoke_slice_review_decision_template.json
data/reviews/stage_a_work_packet_15_validation.json

scripts/stage_a_smoke_common.py
scripts/build_stage_a_smoke_slice.py
scripts/prepare_stage_a_smoke_dataset.py
scripts/run_stage_a_smoke_probe.py
scripts/verify_stage_a_smoke_probe.py

tests/test_stage_a_smoke_probe.py
```

The packet also strengthens `scripts/train.py` and `scripts/generate.py`.

## Training-engineering changes

### Comparable loss measurements

Evaluation now uses fixed seeded batches. The step-0, intermediate, and final losses therefore measure the same token windows instead of unrelated random samples.

### Structured metrics

Every evaluation is appended to:

```text
runs/stage_a_smoke_probe_8k/metrics.jsonl
```

Each row records step, train loss, validation loss, learning rate, elapsed time, and tokens seen.

### True resume test

`--stop-after-step 30` stops cleanly without changing the configured 100-step scheduler horizon. The second invocation uses `--auto-resume`, restoring model, optimizer, RNG, step, and best validation loss.

Resume now rejects changes to architecture, tokenizer, dataset, optimizer-critical settings, or schedule.

### Artifact binding

The resolved local config is bound to:

- exact provisional tokenizer SHA-256;
- exact dataset-manifest SHA-256;
- exact train/validation binary hashes;
- approved smoke-only training scope.

### Safer generation

Generation verifies that the supplied tokenizer hash and vocabulary match the checkpoint before decoding.

## Execution procedure

### 1. Build the separate candidate slice

Dry run:

```powershell
.\p scripts\build_stage_a_smoke_slice.py build
```

Write local candidate files:

```powershell
.\p scripts\build_stage_a_smoke_slice.py build --execute --force
```

Review:

```text
data/foundation/approved/stage_a_smoke_probe_v0_1_candidate/manifest.json
data/foundation/approved/stage_a_smoke_probe_v0_1_candidate/review_sample.csv
```

### 2. Bind Leo's review

Copy the template to a reviewed decision file, enter the exact candidate manifest SHA-256, inspect the stratified records, and set approval fields only after review.

Suggested destination:

```text
data/reviews/stage_a_smoke_slice_v0_1_review_decision.json
```

Approve the exact slice:

```powershell
.\p scripts\build_stage_a_smoke_slice.py approve `
  --review-decision data\reviews\stage_a_smoke_slice_v0_1_review_decision.json `
  --force
```

### 3. Prepare the model dataset

```powershell
.\p scripts\prepare_stage_a_smoke_dataset.py --force
```

This writes local ignored artifacts:

```text
data/foundation/processed/stage_a_smoke_probe_v0_1/
  train.bin
  val.bin
  records.jsonl
  manifest.json
  train_config.json
  binding.json
```

The split is deterministic and family-disjoint. No loss-mask files are created.

### 4. Verify before training

```powershell
.\p scripts\verify_stage_a_smoke_probe.py --require-slice --require-dataset
```

### 5. Preview the exact run

```powershell
.\p scripts\run_stage_a_smoke_probe.py
```

### 6. Execute the two-phase smoke run

```powershell
.\p scripts\run_stage_a_smoke_probe.py --execute --force
```

The runner performs:

```powershell
.\p scripts\train.py `
  --config data\foundation\processed\stage_a_smoke_probe_v0_1\train_config.json `
  --stop-after-step 30

.\p scripts\train.py `
  --config data\foundation\processed\stage_a_smoke_probe_v0_1\train_config.json `
  --auto-resume
```

Then it loads the final checkpoint, performs deterministic generation on three mechanical smoke prompts, verifies every required special token, and writes a structured report.

### 7. Verify the completed run

```powershell
.\p scripts\verify_stage_a_smoke_probe.py `
  --require-slice --require-dataset --require-run
```

## Required pass conditions

- Exact admitted slice is between 30K and 48K provisional-8K tokens.
- No protected evaluation overlap is admitted.
- No Stage B instruction record enters Stage A model data.
- Train and validation families are disjoint.
- Step-0 and final fixed-batch losses are recorded.
- At least one later fixed-batch training loss is below step-0 training loss.
- Final training loss does not materially diverge; validation improvement is recorded but is not required from this tiny smoke run.
- `latest.pt` and `best.pt` exist and match the report hashes.
- Phase 1 stops at step 30.
- Phase 2 explicitly resumes step 30 and reaches step 100.
- Deterministic sample generation appends tokens without error.
- All base and code protocol special tokens remain atomic and round-trip correctly.
- The report records exact commands, config hash, token counts, losses, checkpoint paths, and failure state.

## Local output

```text
runs/stage_a_smoke_probe_8k/
  latest.pt
  best.pt
  metrics.jsonl
  run_metadata.json
  phase1.log
  phase2.log
  generation_samples.json
  smoke_probe_report.json
```

## Failure handling

- Ctrl+C saves `latest.pt` at the last completed optimizer step.
- A failed runner writes `smoke_probe_report.json` with `status=failed` and preserves logs.
- Hash, permission, leakage, family-split, or resume-signature failures stop before promotion.
- Failure does not justify overriding gates or expanding the token budget.

## Interpretation

Passing this packet means the Stage A data-to-checkpoint pipeline works. It does **not** mean Neoma understands English yet, that generation is useful, or that 8K is permanently selected.

After a clean report, Leo should compare the observed loss, speed, RAM, checkpoint size, and generation mechanics before authorizing the next bounded 250K probe.
