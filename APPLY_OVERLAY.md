# Apply Work Packet 13F Overlay

Apply this overlay at repository root on baseline `bf70cfb`.

## Validate

```powershell
.\p -m py_compile scripts\validate_stage_a_english_alternatives.py tests\test_stage_a_english_alternatives.py
.\p scripts\validate_stage_a_english_alternatives.py
.\p -m unittest discover -s tests
```

Review:

- `data/foundation/manifests/stage_a_safe_english_alternatives_v1_candidate.json`
- `data/foundation/manifests/stage_a_internal_english_seed_v1_plan.json`
- `data/plans/STAGE_A_SAFE_ENGLISH_ALTERNATIVES_AND_INTERNAL_SEED_PLAN.md`
- both review-template CSV files.

Do not download any source during this packet. Do not create internal seed content, change `training_allowed`, build a tokenizer, prepare a dataset, or train a model.

The next packet may acquire and hash the three exact `20260601` Wikimedia dumps only after Leo approves the plan and confirms each official `articlesmultistreamdump` job reports `done`.
