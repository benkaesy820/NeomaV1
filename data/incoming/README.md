# Incoming Synthetic Data

Put JSONL files from other models here.

Do not paste them directly into `data/raw`.
UTF-8 and UTF-8-with-BOM files are accepted, which keeps normal PowerShell
exports usable.

Expected format:

```json
{"id":"example_id","language":"python","category":"function","difficulty":"basic","instruction":"Write a Python function...","constraints":["..."],"answer":"def example():\n    pass","bad_code":"","reasoning":"","edge_cases":["..."],"quality_notes":["..."]}
```

Then import:

```powershell
.\p scripts/import_instruction_jsonl.py data/incoming/examples.jsonl --out data/raw/imported_examples.txt
.\p scripts/check_training_data.py --raw data/raw --eval data/eval/code_prompts.jsonl
```
