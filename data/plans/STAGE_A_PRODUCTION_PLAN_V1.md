# Neoma Stage A Production Plan v1

## Decision

Instruction Corpus v0.1 is frozen at baseline `19a93e1`. Stage A now becomes the active workstream. No more broad instruction batches are planned.

Neoma remains private, but Stage A uses production-grade provenance, reproducibility, safety, and evaluation controls. Private use relaxes distribution obligations; it does not justify unknown sources, silent version drift, or unreviewed data.

## Goal

Train a compact coding assistant that understands practical developer English, code structure, tests, errors, uncertainty, and operational consequences. Stage A is ordinary next-token training. It does not use answer-only masks.

The first production target is 50 million unique approved tokens:

- 45 million from ten pinned external sources;
- 5 million created inside the Neoma project;
- one pass as the first production budget, with checkpoint-based stopping if validation stops improving.

This is deliberately close to the data scale expected to be useful for a roughly 2.1M–2.5M parameter probe without bloating the source pool.

## Source freshness

- Technical releases must be stable releases from 2026.
- A 2025 release is allowed only when no newer stable release exists.
- Prerelease, beta, RC, nightly, and moving main-branch content are excluded.
- Rolling documentation repositories must be pinned to the exact 2026 acquisition commit.
- Dataset repositories must be pinned to an immutable revision.

The candidate manifest records the exact current release choices. Acquisition is not approved until Leo verifies the release, license, commit or archive hash, allowed paths, excluded paths, and a sampled quality review.

## Frozen source pool

The ten external sources are:

1. GPT-NL Public Corpus, selected English public-domain and academic subsets only;
2. CPython 3.14.6;
3. pytest 9.1.1;
4. TypeScript 6.0.3;
5. TypeScript Website at a pinned 2026 commit;
6. Node.js 24.18.0 LTS;
7. PowerShell 7.6.3;
8. PowerShell Docs at a pinned 2026 commit;
9. Pester 5.7.1, the latest stable release even though it is from 2025;
10. PostgreSQL 18.4.

Do not add sources after the v1 lock unless an evaluation identifies a concrete missing skill. A later addition must be a separately versioned skill pack.

## Internal data

The project will create rather than download:

- developer dialogue and follow-ups: 1.5M tokens;
- constraint-focused English: 1M;
- bug reports, reviews, and commit language: 1M;
- CLI, configuration, and errors: 750k;
- factual Neoma self-knowledge: 500k;
- verified source-linked transformations: 250k.

Internally created does not mean unverified synthetic data. Every record needs a family ID, creation method, source lineage where applicable, automated checks, deduplication, evaluation-leakage checks, and sampled human review. Hidden chain-of-thought is not collected.

## Evaluation-first rule

Before any source is downloaded or authored for training, lock the 48-prompt development suite and 48-prompt held-out suite included in Work Packet 10. These files and their answer keys are permanently excluded from tokenizer and model training.

The suites test negation, quantities, sequence, references, developer language, clarification judgment, follow-up state, uncertainty, and operational safety. They complement the existing coding evaluations rather than replacing them.

## Acquisition and quarantine

Raw sources are immutable and excluded from Git:

```
data/foundation/sources/raw/
data/foundation/sources/manifests/
data/foundation/staged/
data/foundation/approved/
data/foundation/rejected/
```

For every source, record:

- exact release, tag, commit, or dataset revision;
- acquisition timestamp;
- upstream location;
- archive SHA-256;
- license text and hash;
- allowed and excluded paths or subsets;
- raw, staged, accepted, and rejected counts;
- approved token total.

No raw source is a training source. Only approved records referenced by the final lock manifest may enter tokenization or training.

## Filtering

Reject generated, vendored, minified, binary, fixture-heavy, malformed, duplicated, secret-bearing, personal, obsolete, or low-information content. Prefer complete semantic units that connect code with tests, documentation, errors, or behavior.

Repository ingestion is allowlist-first. A repository archive is never admitted wholesale.

GPT-NL is streamed, not downloaded in full. Use only approved English subsets and preserve per-record license and source metadata.

If Hugging Face or acquisition tooling reports unsafe files in GPT-NL or any other external source, quarantine those files and keep them excluded until Leo explicitly reviews the warning, source path, hash, and sampled content.

## Deduplication

Deduplicate before splitting:

1. exact bytes;
2. normalized text;
3. normalized code tokens;
4. comment/whitespace-reduced code;
5. identifier-normalized structure;
6. Python AST structure where practical;
7. MinHash or token-shingle near-duplicates;
8. cross-source duplicates;
9. comparison against all Stage B records;
10. comparison against all evaluation fields.

Near-duplicates enter review. They are not deleted merely for sharing standard syntax.

## Split policy

Split by source family before segmentation:

- 96% training;
- 2% development;
- 2% locked validation.

Keep implementations, tests, docs, conversations, and close variants from one family in one split. Evaluation suites are separate and never become Stage A validation text.

## Segmentation

Segment at complete functions, classes, test cases, paragraphs, documentation sections, SQL statements, CLI sections, or conversation units. Do not split inside strings, signatures, statements, or sentences. Preserve source ID, family ID, segment index, license class, and content hash.

Stage A uses EOS between independent documents and supervises every token. It does not use `<instruction>` formatting or answer masking.

## Tokenizer decision

After the 500k representative approved corpus exists, train 2k, 4k, and 8k byte-level BPE candidates plus the current baseline. Use representative Stage A text and the frozen Stage B corpus, excluding all evaluations.

Select using:

- lossless round-trip;
- English and code compression;
- identifier, operator, indentation, and protocol fragmentation;
- actual Stage B record fit at 192, 256, 384, and 512 tokens;
- embedding parameter cost;
- CPU throughput;
- A1/A2 evaluation results.

Compression alone cannot select the tokenizer.

## Training ladder

- A0 25k: pipeline smoke test only.
- A1 250k: first English/code sanity probe.
- A2 500k: mixture and tokenizer comparison.
- A3 1M: first serious foundation checkpoint.
- Production: continue from the selected setup through the frozen training split, checkpointing at 5M, 10M, 20M, and the final token budget.

Do not use the old 20,000-step default. Configure by tokens seen and stop when validation and capability gains flatten.

## Stage B transition

The chosen Stage A checkpoint initializes Stage B with future `--init-from-model`, loading model weights only and creating a fresh optimizer and scheduler. `--resume` remains same-run continuation.

Stage B uses the frozen 331-record instruction source, binary answer-only masking, record-aware sampling, and real tokenizer-based context tiers.

## Self-improvement path

Neoma v1 may later propose candidate data for v2, but it may not be its own sole author, verifier, and judge. Early self-generated additions are capped at 10–15% of newly admitted data and require lineage, syntax or test checks, deduplication, leakage checks, privacy scans, and human or stronger-teacher review.

Readiness requires measurable English comprehension, syntax and test success, uncertainty calibration, low duplication, and no locked-suite regression.

## Stop and change control

After source lock, no new source is silently added. New skills are separate versioned packs with their own evaluation, provenance, overlap audit, and ablation. The base corpus stays reproducible.
