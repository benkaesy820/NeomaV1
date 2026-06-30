# Apply Work Packet 16 Overlay

Baseline required: `321f0f2`

From the repository root, copy the overlay files into the project while preserving paths. Review the diff before committing.

This packet adds the bounded Stage A 250K probe. It does not contain local corpora, tokenizer binaries, prepared datasets, checkpoints, or run outputs.

After applying:

```powershell
.\p -m py_compile scripts\stage_a_probe_common.py scripts\build_stage_a_250k_slice.py scripts\prepare_stage_a_250k_dataset.py scripts\run_stage_a_250k_probe.py scripts\verify_stage_a_250k_probe.py scripts\train.py tests\test_stage_a_250k_probe.py
.\p -m unittest discover -s tests
```

Expected tracked-suite result in Ted's environment: **93 passed**.

Then follow `PHASE3_5B_WORK_PACKET_16.md`. Do not approve or run the 250K probe until the generated candidate sample and exact manifest hash have been reviewed.
