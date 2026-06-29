# Stage A Foundation Workspace

This directory contains manifests and plans. Raw or processed corpora must not be committed to Git.

- `manifests/stage_a_sources_v1_candidate.json`: source candidates only; no acquisition or training permission.
- `manifests/stage_a_internal_components_v1_candidate.json`: internal component targets only.
- `manifests/stage_a_source_manifest.schema.json`: documented manifest shape.

Future local-only directories are ignored:

- `sources/raw/`
- `sources/manifests/`
- `staged/`
- `approved/`
- `rejected/`
- `tokenizers/`

Only compact lock manifests, reports, and source-free planning files belong in version control.
