# Phase 3.5B Work Packet 13F - Safe English Alternatives and Internal English Seed Plan

## Baseline

`bf70cfb`

## Goal

Replace the blocked GPT-NL English role with a small, exact, checksum-verifiable source plan and define the first internal-authored English seed without downloading, admitting, tokenizing, preparing, or training anything.

## Decision

- Keep GPT-NL blocked.
- Do not use a manual queued-security override.
- Replace GPT-NL's 12M Stage A v1 target with three official Wikimedia 2026-06-01 dump candidates:
  - Simple English Wikipedia: 5M approved-token target;
  - English Wikibooks: 4M;
  - English Wikiversity: 3M.
- Retry GPT-NL only in a later packet after selected shards become security-safe and an ablation demonstrates unique value.

## Why the replacements are safer to operate

The alternatives are bounded official XML text snapshots with public completion state, official checksum manifests, stable page/revision IDs, and a single documented dump-license family. They can be parsed as inert data and locally hashed. This avoids executing or deserializing remote parquet shards whose security scans remain queued.

They are still untrusted community text. Every page remains quarantined and subject to decoding, markup stripping, quality filtering, privacy/secret scans, deduplication, protected-evaluation exclusion, and human review.

## Internal seed

The packet plans a 60K-token, 460-document seed across six components:

- developer dialogue and follow-ups;
- constraint-focused English;
- bug, review, and commit language;
- CLI, configuration, and errors;
- factual Neoma self-knowledge;
- verified source-linked transformations.

No seed documents are generated or admitted here.

## Files

- `data/foundation/manifests/stage_a_safe_english_alternatives_v1_candidate.json`
- `data/foundation/manifests/stage_a_internal_english_seed_v1_plan.json`
- `data/plans/STAGE_A_SAFE_ENGLISH_ALTERNATIVES_AND_INTERNAL_SEED_PLAN.md`
- `data/reviews/stage_a_safe_english_source_review_template.csv`
- `data/reviews/stage_a_internal_english_seed_review_template.csv`
- `scripts/validate_stage_a_english_alternatives.py`
- `tests/test_stage_a_english_alternatives.py`
- `data/reviews/stage_a_work_packet_13f_validation.json`

## Explicitly not done

- no external English download;
- no internal English generation;
- no training admission;
- no tokenizer rebuild or comparison;
- no dataset preparation;
- no model training.

Every plan and candidate remains `training_allowed=false`.
