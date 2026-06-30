# Phase 3.5B Work Packet 13G - Wikimedia English Acquisition

## Baseline

`82b994f`

## Goal

Acquire the exact safe-English replacement sources approved in Work Packet 13F, without extracting text, approving training data, rebuilding a tokenizer, preparing a dataset, or training a model.

## Scope

Acquire only these official Wikimedia `20260601` sources:

- Simple English Wikipedia;
- English Wikibooks;
- English Wikiversity.

For each source, the acquisition must:

- confirm `articlesmultistreamdump` is exactly `done`;
- download the pages-articles multistream XML archive;
- download the matching multistream index;
- download SHA-1 and MD5 checksum manifests;
- download and hash the Wikimedia dump legal page;
- verify official SHA-1 for the archive and index;
- compute local SHA-256 for every stored artifact;
- write local quarantine manifests;
- keep `training_allowed=false`.

## Files

- `scripts/acquire_wikimedia_english_sources.py`
- `scripts/verify_wikimedia_english_acquisitions.py`
- `tests/test_wikimedia_english_acquisition.py`
- `data/reviews/stage_a_safe_english_acquisition_results_v1.json`

## Explicitly not done

- no XML parsing;
- no wikitext stripping;
- no page selection;
- no training admission;
- no tokenizer comparison or rebuild;
- no dataset preparation;
- no model training.
