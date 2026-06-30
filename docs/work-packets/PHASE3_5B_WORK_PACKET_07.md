# NeomaV1 Phase 3.5B Work Packet 07

## Corpus Audit and Curriculum Gap Analysis

**Prepared by:** Ted
**Expected input baseline:** `b771d01` — before Batch 05 admission
**Relationship to Work Packet 06:** safe to review in the same session; this packet adds audit tooling and reports only.

## Purpose

Work Packet 07 changes Phase 3.5B from batch-driven growth to measurement-driven curriculum design. It audits the admitted instruction corpus without adding examples, changing model code, rebuilding a tokenizer, preparing datasets, or training.

It answers:

- What has Neoma actually been taught?
- Which languages and capabilities are weak, balanced, or overrepresented?
- Which records are duplicates or near-duplicates?
- Is held-out evaluation material leaking through prompts, tests, behavior notes, or distinctive terms?
- Which records may not fit the first 256-token model?
- How much of each record receives answer-only supervision?
- Should the next action be cleanup, one targeted gap batch, or an Instruction Corpus v0.1 freeze?

## Important sequencing for Leo

Leo may inspect Work Packets 06 and 07 together, but the safest operational order is:

1. Apply Work Packet 07 tooling to baseline `b771d01`.
2. Reproduce the bundled pre-Batch-05 audit.
3. Review and resolve or explicitly document every blocking exact-duplicate finding.
4. Audit all Work Packet 06 candidates separately; candidates remain non-training data.
5. Admit only approved Batch 05 rows, up to 40, after structural blockers are handled.
6. Rerun the audit against the post-admission repository.
7. Review the generated human-review queue and write a final audit manifest.
8. Select at most one final targeted instruction batch from measured gaps.

The bundled report is a reproducible preview from the 213-record pre-Batch-05 corpus. It is not the final post-Batch-05 curriculum decision.

## Added files

```text
PHASE3_5B_WORK_PACKET_07.md
APPLY_OVERLAY.md
scripts/audit_instruction_corpus.py
tests/test_audit_instruction_corpus.py
data/reports/README.md
data/reports/pre_batch05_baseline/
  phase3_5b_corpus_audit.json
  phase3_5b_corpus_audit.md
  phase3_5b_curriculum_gaps.md
  phase3_5b_near_duplicates.csv
  phase3_5b_eval_leakage_flags.csv
  phase3_5b_length_outliers.csv
  phase3_5b_record_inventory.csv
  phase3_5b_human_review_queue.csv
  phase3_5b_audit_validation.json
```

## Canonical corpus scope

The audit deliberately avoids double counting:

- Admitted Phase 3.5B records come from `data/incoming/phase3_5b_batch*_accepted.jsonl`.
- Candidate JSONL files are excluded.
- Rendered accepted copies under `data/raw/phase3_5b_batch*_accepted.txt` are integrity-checked but not counted again.
- Legacy protocol examples come from older raw files.
- Raw files without `<instruction>` records are inventoried as possible Stage A foundation seeds and excluded from instruction counts.

## Baseline findings

The preview found:

- 213 total instruction records
- 160 admitted Phase 3.5B records
- 53 legacy protocol records
- 2 non-instruction files inventoried as possible Stage A seeds
- 20 duplicate/near-duplicate flags
- 2 exact answer duplicates in legacy material
- 6 normalized code-structure matches
- 0 critical evaluation-leakage flags
- 0 review-level evaluation-leakage flags
- 88 informational distinctive-term overlaps
- 47 prioritized human-review rows: 20 duplicate findings and 27 severe proxy-length/supervision findings

The two exact legacy answer pairs are:

```text
legacy:code_instruction_direct:008  <->  legacy:code_instruction_seed:009
legacy:code_instruction_direct:011  <->  legacy:code_instruction_seed:004
```

They appear to teach the same Node health-handler and Python indentation-fix solutions, respectively. Leo must inspect the full records and either remove one copy from each pair or document a concrete reason for retaining both.

## Distribution findings

At the pre-Batch-05 baseline:

- PowerShell and text/explanation are materially under target.
- Python is above target.
- TypeScript, JavaScript, and SQL are near target under the audit tolerance.
- Functions/data, validation, debugging, tests, and files/API/automation already have strong overlapping coverage.
- Efficiency is the clearest capability gap.
- Security is low before Work Packet 06 and must be measured again after Batch 05 admission.
- Explanation and SQL/database coverage remain below the 400-example plan targets.

No final Batch 06 topic is authorized by this preview. The post-Batch-05 audit must choose it.

## Length and supervision analysis

For every record, the script measures:

- instruction tokens
- constraints tokens
- bad-code tokens
- reasoning tokens
- answer tokens
- full rendered-record tokens
- supervised answer tokens
- supervised percentage
- character and line counts
- fit at 128, 192, the requested context, and 512 tokens

When no tokenizer exists, the report is explicitly labeled `proxy_lexical_tokens`. Proxy counts are useful for ranking but are not final context decisions. The human-review queue includes only severe proxy outliers; the complete outlier CSV retains every threshold hit.

When an existing tokenizer is supplied, the script uses it without training or modifying it. A missing or unloadable explicitly requested tokenizer is treated as an error rather than silently falling back.

## Duplicate analysis

The audit compares:

- instructions
- answers
- complete rendered records
- normalized code structure

It flags exact normalized matches, high trigram similarity, and identical normalized code structures. Python structure uses AST normalization. Other code uses comment/literal removal and token normalization. Text explanations are excluded from code-structure matching.

Flags require human review. The script never deletes or rewrites data.

## Evaluation leakage analysis

The audit checks admitted training records against both evaluation suites using:

- canonicalized exact prompt containment
- instruction/prompt trigram similarity
- exact complete `test_code` containment
- meaningful test-fixture lines
- expected-behavior-note similarity
- distinctive required or forbidden terms

Generic words are suppressed from the distinctive-term report to reduce noise. Informational term overlap is not automatically leakage.

## Capability coverage

The 400-example plan contains 450 capability slots because one record may cover multiple capabilities. The audit therefore reports both:

- original primary `category` counts without rewriting metadata
- overlapping capability coverage inferred through documented rules

This prevents a record that teaches both validation and file handling from being forced into only one curriculum measurement.

## Automated decision states

The script emits one of:

- `structural_cleanup_required`
- `context_length_cleanup_required`
- `targeted_gap_batch_recommended`
- `ready_to_freeze_instruction_corpus_v0_1`

The decision is conservative. Leo's reviewed audit manifest remains authoritative.

## Validation performed by Ted

- audit-specific unit tests: 10 passed
- complete repository unit tests: 18 passed
- Python compilation across scripts, source, and tests: passed
- deterministic report generation: passed
- accepted JSONL/rendered-raw integrity: passed for all four admitted batches
- no training candidates added by this packet
- no raw training data modified
- no tokenizer rebuild
- no dataset preparation
- no model training

## What this packet intentionally does not do

- no Batch 06 candidates
- no edits to Batches 01–05
- no automatic duplicate deletion
- no direct import into `data/raw`
- no tokenizer training or rebuild
- no dataset preparation
- no checkpoint creation
- no model training
- no change to binary answer-only masking
- no change to model size
