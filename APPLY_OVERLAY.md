# Apply Work Packet 13 Overlay

Apply this overlay at repository root on baseline `2c74c34`.

## Files added or updated

- `PHASE3_5B_WORK_PACKET_13.md`
- `scripts/stage_a_filtering_common.py`
- `scripts/filter_stage_a_sources.py`
- `scripts/verify_stage_a_filtering.py`
- `tests/test_stage_a_source_filtering.py`
- `data/foundation/manifests/stage_a_sources_v1_filtering_plan.json`
- `data/reviews/stage_a_source_filtering_review_template.csv`
- `data/reviews/stage_a_work_packet_13_validation.json`
- `data/foundation/README.md`
- `.gitignore`

## Validate before using local staged sources

```powershell
.\p -m py_compile scripts\stage_a_filtering_common.py scripts\filter_stage_a_sources.py scripts\verify_stage_a_filtering.py tests\test_stage_a_source_filtering.py
.\p -m unittest discover -s tests
.\p scripts\filter_stage_a_sources.py --all
.\p scripts\verify_stage_a_filtering.py
```

The last two commands are non-mutating dry/empty-local checks.

## Filter and verify locally

```powershell
.\p scripts\filter_stage_a_sources.py --all --execute
.\p scripts\verify_stage_a_filtering.py --require-all
```

Review the local output under `data/foundation/filtered/` and complete `data/reviews/stage_a_source_filtering_review_template.csv`.

Do not copy survivors into `approved/`, do not alter any `training_allowed` flag, and do not run tokenizer, dataset-preparation, or training commands during this packet.

GPT-NL remains deferred and must contain zero filtered text rows.
