# Phase 3.5B Work Packet 12 — Stage A Allowed-Path Inventory and Staged Extraction

## Baseline

`f378d0a` — Add Stage A source acquisition tooling

## Goal

Inventory the ten locally quarantined Stage A source snapshots against explicit path and file-type policies, then copy only approved regular archive members into a local **staged, unreviewed, non-training** workspace.

This packet deliberately stops before content-quality filtering or training admission.

## Scope

Work Packet 12 adds:

- a reviewed staging policy for all ten Stage A sources;
- deterministic archive-root removal and logical-path inventory;
- explicit allowlist and exclusion matching, with exclusions taking precedence;
- inventory classifications for selected, excluded, outside-allowlist, unsupported, oversized, unsafe, and special members;
- byte-preserving extraction of selected regular files only;
- source/family/language hints for later filtering, without claiming final document-family pairing;
- per-file SHA-256 hashes and per-source staging manifests;
- end-to-end verification that staged bytes still match their manifests;
- a review template for Leo.

It does **not**:

- stream GPT-NL corpus rows;
- normalize, rewrite, parse, score, deduplicate, or approve staged content;
- set `training_allowed=true` anywhere;
- build a tokenizer;
- prepare a binary dataset;
- start model training.

## Source behavior

### Archive sources

The nine archive-backed sources are inventoried first. Only files that meet every rule below are staged:

1. archive member is a regular file;
2. path is traversal-safe after stripping one common archive root;
3. path matches an explicit allowed path;
4. path does not match an explicit exclusion;
5. file suffix/name is on the source's text/code allowlist;
6. individual and aggregate staged-byte safety limits are respected;
7. the source acquisition manifest has no security hold;
8. artifact bytes still match the acquisition SHA-256.

Staged files are copied as exact bytes. Newlines, encoding, whitespace, comments, and formatting are not changed.

### GPT-NL

Work Packet 12 inventories only the previously acquired selected-subset stream manifest. It records selected remote shard metadata but downloads and stages **zero corpus rows**. Any acquisition or Hugging Face warning continues to block later streaming until Leo explicitly clears it.

## Tracked files

- `PHASE3_5B_WORK_PACKET_12.md`
- `APPLY_OVERLAY.md`
- `scripts/stage_a_staging_common.py`
- `scripts/inventory_stage_a_sources.py`
- `scripts/stage_stage_a_sources.py`
- `scripts/verify_stage_a_staging.py`
- `tests/test_stage_a_source_staging.py`
- `data/foundation/manifests/stage_a_sources_v1_staging_plan.json`
- `data/reviews/stage_a_source_staging_review_template.csv`
- `data/reviews/stage_a_work_packet_12_validation.json`
- updated `data/foundation/README.md`
- updated `.gitignore`

## Local-only outputs

Inventory outputs:

```text
data/foundation/sources/inventory/
  <source_id>.inventory.jsonl
  <source_id>.inventory.summary.json
  stage_a_sources_v1_inventory_summary.json
```

Staged outputs:

```text
data/foundation/staged/
  <source_id>/
    files/<original logical path>
    files.jsonl
    staging_manifest.json
  stage_a_sources_v1_staging_summary.json
```

These directories remain ignored by Git.

## Recommended local execution

First compile and test the tooling:

```powershell
.\p -m py_compile scripts\stage_a_staging_common.py scripts\inventory_stage_a_sources.py scripts\stage_stage_a_sources.py scripts\verify_stage_a_staging.py tests\test_stage_a_source_staging.py
.\p -m unittest discover -s tests
```

Run dry runs:

```powershell
.\p scripts\inventory_stage_a_sources.py --all
.\p scripts\stage_stage_a_sources.py --all
```

Inventory every acquired source:

```powershell
.\p scripts\inventory_stage_a_sources.py --all --execute
```

Review the generated inventory summaries before extraction. Pay particular attention to:

- `unsafe_paths` and `special_members`;
- unexpectedly high `outside_allowlist` or `unsupported_type` counts;
- selected byte totals and safety limits;
- source security holds;
- GPT-NL security-warning metadata.

Then stage all sources:

```powershell
.\p scripts\stage_stage_a_sources.py --all --execute
.\p scripts\verify_stage_a_staging.py --require-all
```

A single source can be handled first:

```powershell
.\p scripts\inventory_stage_a_sources.py --source cpython_3_14_6 --execute
.\p scripts\stage_stage_a_sources.py --source cpython_3_14_6 --execute
.\p scripts\verify_stage_a_staging.py
```

Use `--force` only after reviewing an existing staged source. The replacement is created in a temporary sibling directory and atomically swapped into place.

## Review gates

For each source, Leo should verify:

1. acquisition artifact SHA-256 still matches;
2. acquisition manifest remains `training_allowed=false`;
3. no unresolved security hold exists;
4. archive root stripping is correct;
5. allowlisted and excluded roots match the intended source areas;
6. selected extensions do not admit binaries or generated artifacts;
7. selected file count and bytes are plausible;
8. staged file hashes verify;
9. GPT-NL has no rows staged;
10. no staged source is treated as approved training data.

Complete `data/reviews/stage_a_source_staging_review_template.csv` from the local reports.

## Validation performed by Ted

Ted did not possess Leo's local quarantined archives, because Work Packet 11 intentionally kept them out of Git and out of the uploaded repository snapshot. Therefore, the real ten-source extraction was not falsely reported as completed.

The packet was validated with synthetic archives covering:

- path traversal rejection;
- symlink/special-member rejection;
- exclusions overriding allowed paths;
- unsupported and oversized file classification;
- archive-root stripping;
- exact byte preservation, including CRLF;
- per-file and artifact hashes;
- security-hold refusal;
- duplicate logical-path refusal;
- metadata-only GPT-NL staging;
- post-staging tamper detection;
- all manifests retaining `training_allowed=false`.

Full project suite: **43 tests passed**.

## Next packet

After Leo inventories, stages, verifies, and reviews all ten sources, Work Packet 13 should perform first-pass content decoding, quality/security filtering, generated/vendor/template detection, document-family construction, and rejection manifests. It must still keep all surviving records non-training until a separate admission packet.
