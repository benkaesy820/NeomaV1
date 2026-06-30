# Apply Work Packet 13H Overlay

Apply this overlay at repository root on baseline `59afb7c`.

## Validate

```powershell
.\p -m py_compile scripts\filter_wikimedia_english_sources.py scripts\verify_wikimedia_english_filtering.py scripts\build_neoma_self_knowledge_seed.py tests\test_wikimedia_english_filtering.py tests\test_neoma_self_knowledge_seed.py
.\p scripts\filter_wikimedia_english_sources.py --all
.\p scripts\build_neoma_self_knowledge_seed.py
.\p -m unittest discover -s tests
```

## Filter Locally

```powershell
.\p scripts\filter_wikimedia_english_sources.py --all --execute --force
.\p scripts\verify_wikimedia_english_filtering.py --require-all
```

Review:

- `data/reviews/stage_a_wikimedia_filtering_results_v1.json`
- `data/foundation/internal_seed/neoma_self_knowledge_v0_1_candidates.jsonl`
- local ignored filtered candidates under `data/foundation/filtered/wikimedia_english_20260601/`

Do not admit candidates to training, change `training_allowed`, build a tokenizer, prepare a dataset, or train a model during this packet.
