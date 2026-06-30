# Apply Work Packet 12 Overlay

Apply this overlay at repository root on baseline `f378d0a`.

## Files added or updated

- `PHASE3_5B_WORK_PACKET_12.md`
- `scripts/stage_a_staging_common.py`
- `scripts/inventory_stage_a_sources.py`
- `scripts/stage_stage_a_sources.py`
- `scripts/verify_stage_a_staging.py`
- `tests/test_stage_a_source_staging.py`
- `data/foundation/manifests/stage_a_sources_v1_staging_plan.json`
- `data/reviews/stage_a_source_staging_review_template.csv`
- `data/reviews/stage_a_work_packet_12_validation.json`
- `data/foundation/README.md`
- `.gitignore`

## Validate before using local archives

```powershell
.\p -m py_compile scripts\stage_a_staging_common.py scripts\inventory_stage_a_sources.py scripts\stage_stage_a_sources.py scripts\verify_stage_a_staging.py tests\test_stage_a_source_staging.py
.\p -m unittest discover -s tests
.\p scripts\inventory_stage_a_sources.py --all
.\p scripts\stage_stage_a_sources.py --all
```

The last two commands are dry runs.

## Inventory and stage

```powershell
.\p scripts\inventory_stage_a_sources.py --all --execute
```

Review the local inventory summaries under `data/foundation/sources/inventory/`, then run:

```powershell
.\p scripts\stage_stage_a_sources.py --all --execute
.\p scripts\verify_stage_a_staging.py --require-all
```

Do not copy anything into `approved/`, do not alter any `training_allowed` flag, and do not run tokenizer, dataset-preparation, or training commands during this packet.

Raw acquisitions, inventories, and staged files remain local and ignored by Git. Commit only tooling, plans, tests, and Leo's compact review record.
