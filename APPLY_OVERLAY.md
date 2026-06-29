# Apply Work Packet 11 Overlay

Apply the overlay at repository root on baseline `8066b1f`.

## Files added or updated

- `PHASE3_5B_WORK_PACKET_11.md`
- `scripts/acquire_stage_a_sources.py`
- `scripts/verify_stage_a_acquisitions.py`
- `tests/test_stage_a_source_acquisition.py`
- `data/foundation/manifests/stage_a_sources_v1_acquisition_plan.json`
- `data/reviews/stage_a_source_acquisition_review_template.csv`
- `data/reviews/stage_a_work_packet_11_validation.json`
- `data/foundation/README.md`

## Before network acquisition

```powershell
.\p -m py_compile scripts\acquire_stage_a_sources.py scripts\verify_stage_a_acquisitions.py tests\test_stage_a_source_acquisition.py
.\p -m unittest discover -s tests
.\p scripts\acquire_stage_a_sources.py --all
```

The final command is a dry run.

## Acquire and verify

Start with one fixed-checksum source:

```powershell
.\p scripts\acquire_stage_a_sources.py --source cpython_3_14_6 --execute
.\p scripts\verify_stage_a_acquisitions.py
```

After reviewing that result, acquire the other sources one at a time or run:

```powershell
.\p scripts\acquire_stage_a_sources.py --all --execute
.\p scripts\verify_stage_a_acquisitions.py --require-all
```

Local artifacts are written under:

```text
data/foundation/sources/raw/quarantine/
data/foundation/sources/manifests/
```

Both locations are ignored by Git. Do not move artifacts into `approved/`, do not set `training_allowed=true`, and do not extract them into training inputs during this packet.

## Review

Fill `data/reviews/stage_a_source_acquisition_review_template.csv` from the generated per-source acquisition manifests. Any security warning, checksum mismatch, missing license, unexpected version, or GPT-NL unsafe-file warning remains blocked pending explicit review.
