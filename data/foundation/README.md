# Stage A Foundation Workspace

This directory separates planning, quarantined source acquisition, staged extraction, approval, tokenizer experiments, and later training data.

Tracked planning files:

- `manifests/stage_a_sources_v1_candidate.json`: reviewed source candidates; no acquisition or training permission.
- `manifests/stage_a_sources_v1_acquisition_plan.json`: exact acquisition methods, hosts, versions, filenames, size limits, and fixed checksums.
- `manifests/stage_a_internal_components_v1_candidate.json`: internal authoring targets only.
- `manifests/stage_a_source_manifest.schema.json`: source-planning manifest shape.

Local-only ignored directories:

- `sources/raw/quarantine/`: immutable downloaded archives and GPT-NL stream metadata; never training data.
- `sources/manifests/`: generated per-source acquisition manifests and verification summaries.
- `sources/inventory/`: Work Packet 12 member inventories and allowed-path summaries; still not approved.
- `staged/`: byte-preserving allowed-path extraction plus per-file hashes; still unreviewed and not approved.
- `approved/`: future reviewed Stage A records only.
- `rejected/`: retained rejection metadata or quarantined extracts.
- `tokenizers/`: future tokenizer candidates and reports.

Work Packet 11 acquires source snapshots only. Work Packet 12 inventories allowed paths and copies selected regular text/code members into local staging. Neither packet grants training permission, prepares a dataset, trains a tokenizer, or starts a model run. GPT-NL rows are not streamed in Work Packet 12.
