# NeomaV1 Phase 3.5B — Capability Foundation

**Work packet:** 01
**Prepared for:** Leo and the NeomaV1 continuation model/developer
**Baseline:** `phase-3.5-engineering-reviewed`
**Status:** planning and held-out evaluation design; no capability training run authorized

## 1. Mission

Build the highest reliable coding behavior obtainable from a small CPU-friendly model trained from random weights. Optimize behavior per parameter, training token, RAM, storage, and CPU hour. Do not confuse a smaller model with a more efficient model when the smaller model is materially less correct.

## 2. Baseline rule

Treat `phase-3.5-engineering-reviewed` as immutable. Phase 3.5B work belongs on a new branch. No new corpus, tokenizer, or model checkpoint should be presented as part of the frozen engineering baseline.

## 3. Deliverables in this packet

- `data/eval/phase3_5b_heldout_v1.jsonl`: 80 locked prompts
- `data/eval/PHASE3_5B_EVAL_README.md`: suite-protection and scoring policy
- `data/plans/TRAINING_BATCH_PLAN.md`: 400-example reviewed instruction plan
- `data/plans/FOUNDATION_CORPUS_PLAN.md`: Stage A corpus design
- `data/plans/DATA_QUALITY_GATES.md`: admission and leakage controls
- this master decision record

No new training examples are included in Work Packet 01. This is intentional: evaluation and quality gates must exist before data generation begins.

## 4. Held-out evaluation suite

The suite has 80 prompts: Python 24, TypeScript 14, JavaScript 12, PowerShell 10, SQL 12, and text/explanation 8.

Category counts are recorded in the generated validation report. The suite includes implementation, debugging, tests, files, validation, security, efficiency, data, database, API, types, and explanations.

### Evaluation leakage policy

Reject a candidate training item when any of the following is true:

- its normalized instruction exactly matches an eval prompt;
- its solution directly answers an eval prompt with only names or constants changed;
- token 3-gram Jaccard similarity with an eval prompt is at least 0.80;
- character 5-gram cosine similarity is at least 0.88;
- a reviewer judges it to teach the same exact task/edge-case bundle as an eval item.

Similarity thresholds are review gates, not proof. A lower-scoring semantic copy can still leak; a higher-scoring generic phrase can be harmless. Store the top five nearest eval IDs for every incoming example so reviewers can decide with evidence.

## 5. Training stages

### Stage A — foundation pretraining

- clean source code, tests, concise technical text, SQL, and scripts;
- normal next-token loss;
- random model initialization;
- initial probe at 250k–500k tokens, then a reviewed target around 1M tokens;
- no held-out eval text and no generated answer keys copied from eval tasks.

### Stage B — instruction training

- start from the accepted Stage A model weights through a future `--init-from-model` path;
- create a fresh optimizer and fresh stage schedule;
- use 300–600 inspected instruction examples;
- retain answer-only binary masks for the first controlled comparison.

`--resume` continues the same interrupted run, including optimizer and scheduler state. `--init-from-model` begins a new stage from model weights only. These operations must never share ambiguous CLI behavior.

## 6. Binary versus weighted supervision

Keep answer-only masking **binary** for the first Phase 3.5B experiments. It is easy to audit and preserves causal attribution. Weighted prompt/reasoning loss should be considered only after a binary baseline exists and a specific failure suggests that the model is not learning protocol transitions or concise explanations. Any weighted experiment must be a separate configuration with its own result record.

## 7. Tokenizer retest timing

Do not retrain the tokenizer after every small batch. Retest after both conditions are met:

1. at least 300 reviewed instruction examples exist; and
2. at least 250k clean Stage A tokens exist.

Train tokenizer candidates only on training material, never on the held-out eval suite. Compare at least 2k, 4k, and 8k vocabularies using compression, embedding cost, round-trip correctness, protocol-tag atomicity, CPU throughput, and downstream small-probe behavior.

## 8. Small-model gate before 7M

Train the ~2.1M probe first. The 7M run is justified only after at least two data/evaluation cycles show one of these patterns:

- the 2.1M model improves clearly but plateaus while training/validation remain healthy;
- errors are broad and capacity-like rather than concentrated in missing data categories;
- the same failure persists after targeted examples and adequate token exposure;
- a controlled 7M short probe produces enough behavior gain per CPU hour to justify full training.

Do not scale because loss is merely nonzero or outputs look unimpressive after one run. First distinguish data coverage, undertraining, decoding, and capacity.

## 9. Required result record for every run

Record:

- Git commit and data-manifest hash;
- tokenizer hash and vocabulary size;
- config, seed, parameter count, train/validation token counts;
- elapsed time, tokens/second, peak RAM, checkpoint size;
- loss curves and best checkpoint;
- all 80 eval outputs;
- syntax/test/compliance/repetition/security metrics;
- per-language and per-category scores;
- human-review notes and next targeted corrections.

## 10. Authorized next action

Review and lock the 80-prompt suite, then implement or approve the data-quality gates. Only after both are accepted should Work Packet 02 contain the first 30–40 training examples.
