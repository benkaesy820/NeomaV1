# Stage A Foundation Workspace

This directory separates planning, quarantined source acquisition, staged extraction, filtering, approval, tokenizer experiments, and later training data.

Tracked planning files:

- `manifests/stage_a_sources_v1_candidate.json`: reviewed source candidates; no acquisition or training permission.
- `manifests/stage_a_sources_v1_acquisition_plan.json`: exact acquisition methods, hosts, versions, filenames, size limits, and fixed checksums.
- `manifests/stage_a_sources_v1_staging_plan.json`: allowed-path inventory and exact-byte staging rules.
- `manifests/stage_a_sources_v1_filtering_plan.json`: decoding, quality, deduplication, leakage, and family-construction rules.
- `manifests/stage_a_internal_components_v1_candidate.json`: internal authoring targets only.
- `manifests/stage_a_source_manifest.schema.json`: source-planning manifest shape.

Local-only ignored directories:

- `sources/raw/quarantine/`: immutable downloaded archives and GPT-NL stream metadata; never training data.
- `sources/manifests/`: generated per-source acquisition manifests and verification summaries.
- `sources/inventory/`: Work Packet 12 archive-member inventories and allowed-path summaries; still not approved.
- `staged/`: byte-preserving allowed-path extraction plus per-file hashes; still unreviewed and not approved.
- `filtered/`: Work Packet 13 decoded UTF-8/LF candidates, review queues, rejection reports, family manifests, and hashes; still not approved.
- `approved/`: future explicitly reviewed Stage A records only.
- `rejected/`: retained rejection metadata or quarantined extracts.
- `tokenizers/`: future tokenizer candidates and reports.

Packet boundaries:

- Work Packet 11 acquires immutable source snapshots only.
- Work Packet 12 inventories allowed paths and copies selected regular files into local staging.
- Work Packet 13 decodes, quality-filters, deduplicates, checks protected-data overlap, and constructs document families.

None of these packets grants training permission, prepares a dataset, trains a tokenizer, or starts a model run. GPT-NL rows remain deferred through Work Packet 13.
