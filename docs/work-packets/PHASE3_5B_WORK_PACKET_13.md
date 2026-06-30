# Phase 3.5B Work Packet 13 — Stage A Decoding, Filtering, Families, and Leakage Analysis

## Baseline

`2c74c34`

## Goal

Turn the locally staged repository-source files into **decoded, quality-filtered review candidates** while preserving a strict non-training boundary.

Work Packet 12 established that staged bytes were copied safely from pinned source archives. Work Packet 13 now asks whether those bytes contain useful, decodable, non-generated, non-duplicated material suitable for later human review. Surviving files are still not approved training data.

## Scope

Work Packet 13 adds:

- strict UTF-8, BOM-aware UTF-16/UTF-32, and guarded CP1252 decoding;
- binary/control-character rejection and decoded-text size limits;
- deterministic UTF-8/LF candidate copies while retaining original staged hashes and encoding metadata;
- source-aware path filtering for vendor, generated, snapshot, baseline, giant fixture, obsolete, and low-value files;
- generated-file header detection, repeated-template detection, low-information checks, extreme-line checks, and high-confidence secret rejection;
- explicit human-review flags for suspicious prompt-injection language, partial protected-evaluation overlap, and accepted-instruction overlap;
- exact normalized, template-normalized, and bounded SimHash near-duplicate analysis across all selected sources;
- conservative source-local document-family construction before any later train/dev split;
- protected-data comparison against every JSONL evaluation field, all accepted Phase 3.5B instruction fields, and cleaned legacy instruction-tag fields;
- per-source candidate, human-review, rejection, family, and compact summary manifests;
- verification that every surviving file and every metadata row remains non-training and internally consistent.

It does **not**:

- stream or extract GPT-NL rows;
- set `training_allowed=true`;
- approve any source or document for training;
- select the representative 500K-token corpus;
- train or choose a tokenizer;
- prepare binary datasets;
- start model training.

## GPT-NL boundary

GPT-NL remains fully deferred in this packet.

No GPT-NL row was staged in Work Packet 12, so Work Packet 13 creates only a zero-content deferred manifest for that source. A later dedicated packet must define:

- bounded and resumable row streaming;
- per-row source and license provenance;
- Hugging Face unsafe-file warning quarantine;
- row-level decoding and quality rules;
- deterministic sampling quotas;
- duplicate and evaluation-leakage checks before any text is retained.

Repository-source filtering must be reviewed first rather than mixing a large remote corpus into the same step.

## Tracked files

- `PHASE3_5B_WORK_PACKET_13.md`
- `APPLY_OVERLAY.md`
- `scripts/stage_a_filtering_common.py`
- `scripts/filter_stage_a_sources.py`
- `scripts/verify_stage_a_filtering.py`
- `tests/test_stage_a_source_filtering.py`
- `data/foundation/manifests/stage_a_sources_v1_filtering_plan.json`
- `data/reviews/stage_a_source_filtering_review_template.csv`
- `data/reviews/stage_a_work_packet_13_validation.json`
- updated `data/foundation/README.md`
- updated `.gitignore`

## Local-only outputs

```text
data/foundation/filtered/
  stage_a_sources_v1_filtering_summary.json
  <source_id>/
    files/<original logical path>
    candidates.jsonl
    review_queue.jsonl
    rejections.jsonl
    families.jsonl
    filtering_manifest.json
```

The candidate copies are decoded to UTF-8 with LF line endings for consistent downstream review. Their records retain:

- original staged SHA-256 and byte size;
- detected source encoding and BOM state;
- original newline style;
- decoded, normalized, template, and SimHash fingerprints;
- source ID, logical path, language hint, family ID, and family rule;
- explicit rejection or review reasons;
- protected-data overlap details;
- `training_allowed=false`.

Rejected files are not copied into the filtered candidate tree. Their metadata and reasons remain in `rejections.jsonl`.

## Quality decisions

### Automatic rejection

High-confidence automatic rejection includes:

- changed or missing staged bytes;
- undecodable or binary-looking content;
- explicit vendor/generated/snapshot/baseline paths;
- source-specific noisy fixture or obsolete paths;
- tiny or nearly empty material;
- extreme decoded size or line length;
- generated-file header markers;
- repeated template/snapshot domination;
- high-confidence private-key or production-token shapes;
- exact or near-complete protected evaluation wording;
- exact/template/high-confidence near-duplicates where an earlier deterministic survivor already exists.

### Human-review queue

Ambiguous cases are retained only in `review_queue.jsonl`, including:

- moderate repeated-line ratios;
- prompt-injection terminology appearing in legitimate docs/tests;
- accepted-instruction text overlap;
- partial evaluation overlap below the critical rejection threshold;
- possible, but not high-confidence, near-duplicates.

A review-queue record is not an approved training record.

## Document families

Families are constructed before future splitting so related implementation, documentation, SQL expected output, and tests can remain in the same split.

Examples include:

- CPython `Lib/json/`, `Lib/test/test_json.py`, and `Doc/library/json.rst`;
- Node.js `lib/fs.js` and `doc/api/fs.md`;
- PostgreSQL `src/test/regress/sql/transactions.sql` and `expected/transactions.out`;
- PowerShell cmdlet documentation and feature/test areas;
- pytest, Pester, TypeScript source/test/document sections.

Rules are intentionally conservative. Leo must inspect unusually large or unusually broad families before Packet 14.

## Protected-data policy

Evaluation protection is stricter than ordinary deduplication:

- exact evaluation field containment is critical and rejected;
- near-complete five-token-shingle containment is critical and rejected;
- meaningful partial evaluation overlap enters human review;
- accepted and legacy instruction overlap enters human review rather than being silently treated as foundation prose.

All evaluation JSONL fields are protected, not only the visible prompt field.

## Recommended local execution

Compile and test first:

```powershell
.\p -m py_compile scripts\stage_a_filtering_common.py scripts\filter_stage_a_sources.py scripts\verify_stage_a_filtering.py tests\test_stage_a_source_filtering.py
.\p -m unittest discover -s tests
```

Inspect the plan without touching staged files:

```powershell
.\p scripts\filter_stage_a_sources.py --all
```

Run filtering across all staged repository sources in one global dedup scope:

```powershell
.\p scripts\filter_stage_a_sources.py --all --execute
.\p scripts\verify_stage_a_filtering.py --require-all
```

Use `--force` only after reviewing an existing local filtered tree. The complete output is built in a temporary sibling directory and then atomically replaces the old tree.

## Review gates

For every source, Leo should review:

1. candidate, review, rejection, and family counts;
2. all decode/binary rejection categories;
3. samples from every path/generated/vendor rejection category;
4. every critical evaluation-leakage rejection;
5. every accepted-instruction overlap review;
6. duplicate survivors and a sample of rejected duplicate pairs;
7. very large families and cross-code/doc/test pairings;
8. whether source-specific path rules reject useful material or retain obvious noise;
9. whether generated markers create false positives in documentation;
10. that no record, family, manifest, or output grants training permission.

Complete `data/reviews/stage_a_source_filtering_review_template.csv` from the local reports.

## Validation performed by Ted

Ted did not possess Leo's local staged source trees because Work Packet 12 correctly kept them out of Git and out of the uploaded snapshot. Therefore, Ted did not claim to have filtered the real source pool.

Synthetic end-to-end validation covered:

- UTF-8 BOM, UTF-16 BOM, CP1252, and newline detection;
- binary/NUL/control-character rejection;
- vendor/generated path rejection;
- generated-header and repeated-template rejection;
- exact and near-duplicate fingerprints;
- critical evaluation leakage and instruction-overlap review;
- CPython, Node.js, and PostgreSQL family pairing;
- deferred GPT-NL with zero content rows;
- atomic local output creation;
- every row retaining `training_allowed=false`;
- filtered-file hash and extra-file tamper detection;
- dry-run operation without local staged data.

Full project suite: **55 tests passed**.

## Next packet

After Leo executes, verifies, and reviews Work Packet 13, Work Packet 14 should select a representative **approved 500K-token sample** from the reviewed survivors and prepare tokenizer-comparison inputs and measurement tooling.

Packet 14 must make training admission explicit and reviewable. It should not silently treat every Work Packet 13 survivor as approved.
