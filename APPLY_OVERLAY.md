# Apply Work Packet 10 Overlay

Apply this overlay to repository baseline `19a93e1`.

Review all replacements and additions before committing. In particular, verify the current stable source versions and licenses independently at acquisition time.

Suggested checks:

```powershell
.\p -m py_compile scripts\validate_stage_a_readiness.py scripts\score_stage_a_english_eval.py tests\test_stage_a_readiness.py
.\p scripts\validate_stage_a_readiness.py --repo .
.\p scripts\check_training_data.py --raw data\raw --eval data\eval\stage_a_english_dev_v1.jsonl
.\p scripts\check_training_data.py --raw data\raw --eval data\eval\stage_a_english_locked_v1.jsonl
$env:PYTHONPATH = "$PWD\src"
.\p -m unittest discover -s tests

git diff --check
git status --short
```

The source and internal manifests remain `training_allowed=false`. Do not download, tokenize, prepare, or train until Leo approves the packet and creates the next baseline.
