# Apply Work Packet 13G Overlay

Apply this overlay at repository root on baseline `82b994f`.

## Validate

```powershell
.\p -m py_compile scripts\acquire_wikimedia_english_sources.py scripts\verify_wikimedia_english_acquisitions.py tests\test_wikimedia_english_acquisition.py
.\p scripts\acquire_wikimedia_english_sources.py --all
.\p -m unittest discover -s tests
```

## Acquire Locally

```powershell
.\p scripts\acquire_wikimedia_english_sources.py --all --execute
.\p scripts\verify_wikimedia_english_acquisitions.py --require-all
```

Review:

- `data/reviews/stage_a_safe_english_acquisition_results_v1.json`
- local ignored manifests under `data/foundation/sources/manifests/`
- local ignored artifacts under `data/foundation/sources/raw/quarantine/`

Do not parse XML, extract pages, change `training_allowed`, build a tokenizer, prepare a dataset, or train a model during this packet.
