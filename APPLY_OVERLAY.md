# Apply Work Packet 15 Overlay

Baseline: `32d3ef4`

1. Extract this overlay into the root of the NeomaV1 repository.
2. Review `PHASE3_5B_WORK_PACKET_15.md` and the new smoke-probe plan.
3. Run:

```powershell
.\p -m py_compile scripts\stage_a_smoke_common.py scripts\build_stage_a_smoke_slice.py scripts\prepare_stage_a_smoke_dataset.py scripts\run_stage_a_smoke_probe.py scripts\verify_stage_a_smoke_probe.py scripts\train.py scripts\generate.py tests\test_stage_a_smoke_probe.py
.\p -m unittest discover -s tests
```

4. Build and review the local 30K–48K-token candidate slice.
5. Approve it only with a Leo decision bound to the exact candidate-manifest SHA-256.
6. Prepare and verify the family-disjoint dataset.
7. Run the two-phase smoke probe and verify the generated report.

No local corpus, tokenizer binary, processed dataset, checkpoint, or run output belongs in Git.
