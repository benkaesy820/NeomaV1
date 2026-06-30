# Apply Work Packet 14 Overlay

Apply this overlay at repository root on baseline `35e6d17`.

## Validate tooling

```powershell
.\p -m py_compile scripts\build_stage_a_tokenizer_sample.py scripts\verify_stage_a_tokenizer_sample.py scripts\run_stage_a_tokenizer_comparison.py tests\test_stage_a_tokenizer_admission.py
.\p scripts\build_stage_a_tokenizer_sample.py build
.\p -m unittest discover -s tests
```

The normal Neoma environment should have the `tokenizers` package and should run the expected 82 tests. Ted’s sandbox passed 79 available tests; the existing `test_prepare_dataset` module could not import because `tokenizers` is unavailable there.

## Build the local candidate sample

```powershell
.\p scripts\build_stage_a_tokenizer_sample.py build --execute
.\p scripts\verify_stage_a_tokenizer_sample.py
```

Review:

```text
data/foundation/approved/stage_a_tokenizer_sample_v0_1_candidate/manifest.json
data/foundation/approved/stage_a_tokenizer_sample_v0_1_candidate/review_sample.csv
```

Copy the committed review-decision template to a local decision file. Set:

```text
candidate_manifest_sha256=<SHA-256 of candidate manifest.json>
status=approved
approved_for_tokenizer_comparison=true
reviewed_utc=<actual UTC timestamp>
```

Keep `model_training_allowed=false`.

## Approve tokenizer use only

```powershell
.\p scripts\build_stage_a_tokenizer_sample.py approve `
  --review-decision path\to\completed_review_decision.json

.\p scripts\verify_stage_a_tokenizer_sample.py `
  --root data\foundation\approved\stage_a_tokenizer_sample_v0_1 `
  --require-approved
```

## Train and compare tokenizer candidates

```powershell
.\p scripts\run_stage_a_tokenizer_comparison.py all
```

Review:

```text
data/foundation/tokenizers/stage_a_v0_1/training_manifest.json
data/foundation/tokenizers/stage_a_v0_1/comparison_report.json
```

Do not pass the tokenizer sample to `prepare_dataset.py`. Do not prepare model tokens or start model training in this packet.
