# Phase 3.5B Work Packet 13E - GPT-NL English Streaming and Quarantine

## Baseline

`f319c02`

## Goal

Resolve the missing GPT-NL English-corpus step without weakening the safety boundary. This packet adds bounded GPT-NL streaming tooling, but the local execution remains blocked while Hugging Face reports selected parquet shards as security `queued` rather than `safe`.

## What This Packet Does

- inspects selected GPT-NL shard metadata from the pinned dataset revision;
- records shard size, LFS SHA-256, last commit, and Hugging Face security status;
- refuses row streaming unless shard security is safe or a future run uses an explicit override;
- defines bounded row-download, row-filtering, provenance, deduplication, and leakage checks for the eventual safe stream;
- keeps every output `training_allowed=false`;
- commits only compact tooling, plans, tests, and review summaries.

## What It Does Not Do

- no tokenizer rebuild;
- no dataset preparation;
- no model training;
- no training admission;
- no silent GPT-NL row download from queued-security shards.

## Local Output

Ignored local outputs live under:

```text
data/foundation/gptnl_streaming/
```

The current local result is a security/quarantine report, not an English corpus.

## Current Finding

The planned GPT-NL parquet shards are visible at the pinned revision, but sampled selected shards report:

```text
safe=false
security.status=queued
```

That means the correct state is:

```text
metadata inspected: yes
actual corpus rows downloaded: 0
english text filtered: 0
training_allowed: false
status: blocked_security_queued
```

## Next Step

If Hugging Face later marks the shards safe, rerun the streaming tool with a bounded budget. If we decide queued-security shards are acceptable for private local review, that must be an explicit user-approved override, not the default.
