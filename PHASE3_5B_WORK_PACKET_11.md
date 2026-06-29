# Phase 3.5B Work Packet 11 — Stage A Source Acquisition and Immutable Hashing

## Baseline

`8066b1f` — Add Stage A readiness and English eval packet

## Goal

Acquire the ten approved Stage A v1 external source snapshots into a local quarantine, pin their exact identities, calculate immutable SHA-256 hashes, inventory archives without extracting them, and produce per-source manifests for Leo review.

This packet does **not** admit any source content into training.

## Sources

1. GPT-NL selected English-subset stream manifest — exact Hugging Face revision resolved at acquisition.
2. CPython 3.14.6 official XZ source archive.
3. pytest 9.1.1 official PyPI source distribution.
4. TypeScript 6.0.3 exact Git tag resolved to a full commit.
5. TypeScript Website acquisition-day default-branch commit.
6. Node.js 24.18.0 LTS official source archive.
7. PowerShell 7.6.3 exact Git tag resolved to a full commit.
8. PowerShell Docs acquisition-day default-branch commit.
9. Pester 5.7.1 exact Git tag resolved to a full commit.
10. PostgreSQL 18.4 official source archive.

## Included files

- `scripts/acquire_stage_a_sources.py`
- `scripts/verify_stage_a_acquisitions.py`
- `tests/test_stage_a_source_acquisition.py`
- `data/foundation/manifests/stage_a_sources_v1_acquisition_plan.json`
- `data/reviews/stage_a_source_acquisition_review_template.csv`
- `data/reviews/stage_a_work_packet_11_validation.json`
- updated `data/foundation/README.md`

## Safety and immutability rules

- Acquisition is dry-run unless `--execute` is supplied.
- HTTPS and explicit host allowlists are mandatory.
- Files stream to `.part`, are hashed while downloading, flushed, and atomically renamed.
- Published SHA-256 values are enforced for CPython, pytest, Node.js, and PostgreSQL.
- GitHub tags are resolved through the API to full commit IDs before downloading a commit-addressed archive.
- Rolling documentation repositories are pinned to their exact acquisition-time commit.
- Archive paths, symlinks, special members, member counts, declared sizes, suspicious binary names, and license files are inspected without extracting content.
- Every artifact is stored under local quarantine.
- Every manifest and source keeps `training_allowed=false`.
- GPT-NL receives metadata/stream-manifest acquisition only. No corpus rows are downloaded in this packet.
- Any Hugging Face or acquisition security warning creates a security hold for Leo review.
- Acquired files and local manifests remain ignored by Git.

## Official fixed checksums

| Artifact | SHA-256 |
|---|---|
| `Python-3.14.6.tar.xz` | `143b1dddefaec3bd2e21e3b839b34a2b7fb9842272883c576420d605e9f30c63` |
| `pytest-9.1.1.tar.gz` | `1088fbde8f2b49d95a549a195707afa7a76a3ce9bcadc26b6d71f0ffda5fe313` |
| `node-v24.18.0.tar.xz` | `e94afde24db08e0c564ee7110a2d5aab51ee0059382c9fd8233c54eec47b28f9` |
| `postgresql-18.4.tar.bz2` | `81a81ec695fb0c7901407defaa1d2f7973617154cf27ba74e3a7ab8e64436094` |

## Recommended local execution order

Run one source first:

```powershell
.\p scripts\acquire_stage_a_sources.py --source cpython_3_14_6 --execute
.\p scripts\verify_stage_a_acquisitions.py
```

Then acquire the remaining sources, preferably one at a time so failures and warnings are easy to review:

```powershell
.\p scripts\acquire_stage_a_sources.py --source pytest_9_1_1 --execute
.\p scripts\acquire_stage_a_sources.py --source typescript_6_0_3 --execute
.\p scripts\acquire_stage_a_sources.py --source typescript_website_2026 --execute
.\p scripts\acquire_stage_a_sources.py --source node_24_18_0 --execute
.\p scripts\acquire_stage_a_sources.py --source powershell_7_6_3 --execute
.\p scripts\acquire_stage_a_sources.py --source powershell_docs_2026 --execute
.\p scripts\acquire_stage_a_sources.py --source pester_5_7_1 --execute
.\p scripts\acquire_stage_a_sources.py --source postgresql_18_4 --execute
.\p scripts\acquire_stage_a_sources.py --source gptnl_english_2026 --execute
.\p scripts\verify_stage_a_acquisitions.py --require-all
```

`GITHUB_TOKEN` and `HF_TOKEN` are optional environment variables for authenticated API access. They must never be written into manifests or committed.

## Leo review gates

For every source:

1. Confirm expected version and resolved full commit/revision.
2. Confirm artifact SHA-256 and any published checksum.
3. Review archive warnings, suspicious-file samples, path safety, and special members.
4. Confirm the declared license and hashed license files.
5. For GPT-NL, review every Hugging Face security warning and selected-file metadata before later row streaming.
6. Keep the source quarantined and `training_allowed=false`.
7. Complete `stage_a_source_acquisition_review_template.csv`.
8. Commit only compact plans, tooling, tests, and a reviewed lock manifest—not raw archives.

## Explicitly not included

- no content extraction into staged or approved corpora;
- no filtering or token quotas filled;
- no internal 5M-token authoring;
- no tokenizer rebuild;
- no dataset preparation;
- no model training;
- no training permission for any acquired source.

## Validation before handoff

- acquisition plan: 10 unique sources;
- PowerShell pinned to `v7.6.3`;
- all source and summary records retain `training_allowed=false`;
- four official source checksums pinned;
- safe URL, GitHub resolution, PyPI metadata, archive scan, license hash, GPT-NL quarantine, and summary tests passed;
- full project test suite: 33 passed;
- Python compilation passed;
- all-source dry-run passed.

Ted's sandbox could not perform the real network downloads because outbound DNS is unavailable. Leo must execute acquisition locally and return the generated per-source manifests and review decisions.

## Next packet

After all ten acquisitions are reviewed, Work Packet 12 should perform source inventory, allowed-path extraction into a **staged but still non-training** corpus, document-family identification, and first-pass quality/security filtering. It must not build a tokenizer or start training yet.
