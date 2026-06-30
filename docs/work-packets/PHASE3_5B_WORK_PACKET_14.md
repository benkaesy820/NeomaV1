# Phase 3.5B Work Packet 14 — Stage A Tokenizer-Sample Admission and Comparison

## Baseline

`35e6d17`

## Goal

Select a deterministic, representative, reviewable Stage A sample from the locally filtered source pools, approve it **for tokenizer comparison only**, and compare 2K, 4K, and 8K byte-level BPE candidates.

This packet does not prepare a model dataset and does not train Neoma.

## Why this packet exists

Stage A now has:

- nine filtered repository/documentation source pools;
- 33,110 clean Wikimedia English candidates;
- 2,625 Wikimedia human-review rows that remain excluded;
- about 12M Wikimedia proxy tokens;
- 50 factual self-knowledge candidates;
- the frozen 331-record Stage B instruction corpus.

The tokenizer must be trained on a representative vocabulary sample before the full Stage A corpus is admitted. The sample must represent English, code, documentation, tests, developer instructions, and Neoma-specific terms without silently becoming model-training data.

## Representative sample target

The plan targets approximately 500K proxy tokens:

| Component | Target |
|---|---:|
| Frozen Stage B instruction records without protected eval overlap | ~74.8K |
| Filtered repository code/docs/tests | ~218.7K |
| Filtered Wikimedia English | 206K |
| Stable Neoma self-knowledge | small complete allowlist |
| **Total permitted range** | **485K–505K** |

Repository quotas cover CPython, pytest, TypeScript, TypeScript Website, Node.js, PowerShell, PowerShell Docs, Pester, and PostgreSQL. Local execution showed that the filtered TypeScript Website pool is intentionally small, so the representative budget is filled from the already-filtered Wikimedia English pool rather than weakening quality gates.

Wikimedia quotas are:

- Simple English Wikipedia: 92K;
- English Wikibooks: 63K;
- English Wikiversity: 51K.

Four frozen Stage B records are excluded from tokenizer training because they have review-level overlap with protected evaluation wording. They remain part of the frozen instruction corpus and are still measured after tokenizer training.

The tokenizer sample is not intended to reproduce the final Stage A training mixture exactly. It is intended to expose the tokenizer to the vocabulary and formatting that the model will encounter.

## Selection controls

Only clean `filtered_candidate_not_admitted` rows can be selected from external source pools.

The selector excludes:

- every human-review queue row;
- every rejected row;
- critical or review-level evaluation leakage;
- cross-source exact, template, and high-confidence near duplicates;
- records outside source token quotas;
- excess members from one document family;
- oversized repository files unsuitable for the representative sample.

Wikimedia selection uses one record per family to maximize topic diversity. Repository selection allows up to two members per family so useful code/test or code/document relationships can be represented.

## Self-knowledge correction

The original 50-row seed mixed stable facts with transient project state. Packet 14 does not blindly admit all 50.

It allows 41 stable records and defers nine records concerning:

- Leo/Ted project-process identity;
- evaluation-governance wording;
- provisional model-size predictions;
- source-filtering state;
- tokenizer state;
- training state.

Transient statements such as “the tokenizer has not been selected” would become false after this packet. They must not be baked into the foundation model as permanent self-knowledge.

## Permission model

Candidate sample:

```text
training_allowed=false
model_training_allowed=false
tokenizer_training_allowed=false
```

After Leo reviews a stratified sample and signs a decision bound to the exact candidate-manifest SHA-256:

```text
training_allowed=false
model_training_allowed=false
tokenizer_training_allowed=true
status=approved_for_tokenizer_comparison_only
```

The generic training flag intentionally remains false. This prevents the approved tokenizer corpus from being accidentally passed into model dataset preparation.

## Tokenizer comparison

The packet prepares candidates for:

- 2K byte-level BPE;
- 4K byte-level BPE;
- 8K byte-level BPE.

All use:

- the `code` special-token preset;
- `min_frequency=2`;
- `max_token_length=32`;
- lossless ByteLevel encoding.

The comparison reports:

- exact round-trip failures;
- unknown-token count;
- atomic protocol tags;
- bytes per token by group and source;
- vocabulary utilization;
- singleton vocabulary entries;
- full Stage B record lengths;
- fit counts at 128, 192, 256, 384, and 512 tokens;
- embedding parameters;
- estimated total parameters for `code_phase3_cpu`.

Expected model sizes for that architecture are:

| Vocabulary | Approximate parameters |
|---:|---:|
| 2K | 2,155,200 |
| 4K | 2,539,200 |
| 8K | 3,307,200 |

The script may issue a transparent provisional recommendation, but tokenizer selection remains a review decision and must later be confirmed by the planned small Stage A probe.

## Added files

```text
scripts/build_stage_a_tokenizer_sample.py
scripts/verify_stage_a_tokenizer_sample.py
scripts/run_stage_a_tokenizer_comparison.py
tests/test_stage_a_tokenizer_admission.py

data/foundation/manifests/stage_a_tokenizer_sample_v0_1_plan.json
data/reviews/stage_a_tokenizer_sample_review_decision_template.json
data/reviews/stage_a_tokenizer_sample_review_template.csv
data/reviews/stage_a_work_packet_14_validation.json
```

## Explicitly not done

- no local filtered corpus was included in the overlay;
- no candidate sample was built in Ted’s environment;
- no source received model-training permission;
- no tokenizer was trained in Ted’s environment;
- no model dataset was prepared;
- no Stage A or Stage B model training occurred.

## Next step after Leo executes Packet 14

1. Build the candidate 500K sample locally.
2. Review the manifest, source totals, review sample, family balance, and self-knowledge allowlist.
3. Approve the exact candidate manifest for tokenizer comparison only.
4. Train and benchmark 2K/4K/8K tokenizers.
5. Record the tokenizer decision and real token lengths for all 331 Stage B records.
6. Only then prepare Work Packet 15 for Stage A corpus admission and the 25K/250K/500K training ladder.
