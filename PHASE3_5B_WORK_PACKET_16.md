# Phase 3.5B Work Packet 16 — Stage A 250K Probe

Baseline: `321f0f2`

## Purpose

Work Packet 15 proved that the Stage A training pipeline works. Work Packet 16 moves to a separately reviewed **approximately 250K exact-token** corpus and asks a narrower question:

> Does Neoma begin learning useful English and code structure while remaining stable, reproducible, and small enough for CPU experimentation?

This is still a bounded probe. It does not authorize the 500K/1M stages and does not make the provisional 8K tokenizer final.

## Frozen boundaries

- No new broad instruction batch.
- Frozen Stage B instruction records remain excluded.
- All protected evaluation fields remain excluded and fingerprint-bound.
- The source 500K tokenizer sample remains model-training-forbidden.
- The derived 250K slice requires a new Leo review bound to its exact manifest SHA-256.
- Stage A uses normal next-token loss with no answer-only mask.
- The provisional 8K tokenizer is used without promotion to final status.
- No 500K/1M expansion is automatic, even after a passing run.

## Slice target

Exact counts use the provisional 8K tokenizer and include one `<eos>` token per record.

| Component | Target | Allowed range |
|---|---:|---:|
| Repository code, tests, and documentation | 150K | 140K–160K |
| Wikimedia English | 95K | 85K–105K |
| Stable Neoma self-knowledge | 3K | 0.3K–4K |
| **Total** | **248K** | **235K–265K** |

Selection is deterministic, source-balanced, family-capped, hash-bound, deduplicated, and rechecked against every protected evaluation field. `frozen_stage_b` is a forbidden group.

## Model and run configuration

Tracked template: `configs/stage_a_250k_probe_8k_cpu.json`

| Field | Value |
|---|---:|
| Vocabulary | provisional 8K BPE |
| Parameters | 3,307,200 |
| Layers / width / FFN | 4 / 192 / 576 |
| Query / KV heads | 4 / 2 |
| Context | 128 |
| Batch / accumulation | 2 / 2 |
| Tokens per optimizer step | 512 |
| Maximum steps | 500 |
| Maximum tokens seen | 256,000 |
| Planned resume boundary | step 150 |
| Evaluation interval | 25 steps |
| Save interval | 50 steps |
| Objective | full next-token loss |

The run starts from random weights. It does not initialize from the Work Packet 15 checkpoint because the goal is a clean corpus-scale comparison under the same architecture and provisional tokenizer.

## Added files

```text
configs/stage_a_250k_probe_8k_cpu.json

data/eval/stage_a_250k_probe_prompts_v0_1.jsonl
data/foundation/manifests/stage_a_250k_probe_v0_1_plan.json
data/reviews/stage_a_250k_slice_review_decision_template.json
data/reviews/stage_a_work_packet_16_validation.json

scripts/stage_a_probe_common.py
scripts/build_stage_a_250k_slice.py
scripts/prepare_stage_a_250k_dataset.py
scripts/run_stage_a_250k_probe.py
scripts/verify_stage_a_250k_probe.py

tests/test_stage_a_250k_probe.py
```

The packet also extends `scripts/train.py` with cross-platform peak resident-memory reporting. On Windows this uses `GetProcessMemoryInfo`; on Unix-like systems it uses `getrusage`. Memory is recorded in evaluation rows rather than estimated.

## Probe outputs

Eight fixed, training-excluded prompts cover:

- missing versus empty;
- preserving input;
- operation order;
- conditional meaning;
- short Python continuation;
- loop/function structure;
- TypeScript continuation;
- SQL continuation.

The runner saves deterministic outputs from both the random initialization and the final checkpoint. These are qualitative mechanics probes only. The report always keeps:

```text
capability_claim_authorized=false
```

Loss movement plus cleaner-looking samples may justify the next experiment, but they do not prove reliable English understanding or coding ability.

## Work Packet 15 comparison

The runner requires the passed local report:

```text
runs/stage_a_smoke_probe_8k/smoke_probe_report.json
```

It compares:

- initial and final fixed-batch train loss;
- initial and final validation loss;
- effective tokens per second including evaluation/checkpoint overhead;
- peak resident memory;
- latest/best checkpoint size;
- generation and special-token mechanics.

Work Packet 15 did not record native peak RSS. That baseline value is reported as `null`, not guessed. Its speed is derived from its recorded phase elapsed times when available.

## Pass gates

The real run passes only when:

- the admitted slice contains 235K–265K exact tokens;
- no Stage B record enters Stage A;
- no critical/review evaluation leakage exists;
- train and validation families are disjoint;
- the run stops at step 150 and resumes the same optimizer/scheduler state;
- a later fixed-batch training loss is below step zero;
- best and final validation loss are below step zero;
- latest and best checkpoints exist and hash correctly;
- deterministic pre/post generation runs;
- required protocol/special tokens remain atomic and lossless;
- effective speed, peak RSS, and checkpoint byte sizes are recorded;
- no automatic 500K/1M authorization is emitted.

## Local execution

### 1. Build the candidate slice

```powershell
.\p scripts\build_stage_a_250k_slice.py build
.\p scripts\build_stage_a_250k_slice.py build --execute --force
```

Review:

```text
data/foundation/approved/stage_a_250k_probe_v0_1_candidate/manifest.json
data/foundation/approved/stage_a_250k_probe_v0_1_candidate/review_sample.csv
```

### 2. Approve the exact reviewed candidate

Copy the template to:

```text
data/reviews/stage_a_250k_slice_v0_1_review_decision.json
```

Fill the exact candidate-manifest SHA-256, exclusions, reviewer, time, and approval fields. Then run:

```powershell
.\p scripts\build_stage_a_250k_slice.py approve `
  --review-decision data\reviews\stage_a_250k_slice_v0_1_review_decision.json `
  --force
```

### 3. Prepare and verify the family-disjoint dataset

```powershell
.\p scripts\prepare_stage_a_250k_dataset.py --force

.\p scripts\verify_stage_a_250k_probe.py `
  --require-slice `
  --require-dataset
```

### 4. Preview and run

```powershell
.\p scripts\run_stage_a_250k_probe.py
.\p scripts\run_stage_a_250k_probe.py --execute --force
```

The runner performs:

```powershell
.\p scripts\train.py --config data\foundation\processed\stage_a_250k_probe_v0_1\train_config.json --stop-after-step 150
.\p scripts\train.py --config data\foundation\processed\stage_a_250k_probe_v0_1\train_config.json --auto-resume
```

### 5. Verify the completed run

```powershell
.\p scripts\verify_stage_a_250k_probe.py `
  --require-slice `
  --require-dataset `
  --require-run
```

## Run evidence

The real local run writes:

```text
runs/stage_a_250k_probe_8k/
  phase1.log
  phase2.log
  metrics.jsonl
  run_metadata.json
  latest.pt
  best.pt
  generation_pretrain.json
  generation_posttrain.json
  probe_report.json
```

`probe_report.json` binds the dataset, tokenizer, config, probe prompts, checkpoints, losses, commands, timing, memory, checkpoint sizes, Work Packet 15 comparison, and failure state.

## Failure handling

- Ctrl+C atomically saves `latest.pt` at the latest completed step.
- A failed command preserves logs and any valid checkpoint.
- Resume rejects changes to data, tokenizer, architecture, schedule, or optimizer-critical configuration.
- A failed loss, leakage, memory, checkpoint, generation, or integrity gate writes `status=failed` and leaves expansion unauthorized.
- `--force` removes only the expected run directory below `runs/`.

## Validation performed by Ted

- Python compilation passed for all new and modified scripts.
- Six new Work Packet 16 tests passed.
- Full project suite: **93 tests passed**.
- Parameter count remains **3,307,200**.
- Config is bounded to **500 steps / 256,000 tokens seen**.
- Evaluation probe rows remain `training_allowed=false`.
- ZIP integrity passed.

The real 250K corpus and run were not created in Ted's sandbox because the approved tokenizer sample, tokenizer, filtered source pools, and Work Packet 15 report correctly remain local on Leo's machine.
