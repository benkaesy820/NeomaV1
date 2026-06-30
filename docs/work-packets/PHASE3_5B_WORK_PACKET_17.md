# Work Packet 17 - Stage A 250K Extended Diagnostic

Baseline: `22ad99c`

This packet keeps the approved 250K Stage A corpus fixed and trains a separate diagnostic run for 2,000 steps. It does not admit new data, rebuild the tokenizer, prepare a larger dataset, or authorize a capability claim.

## Purpose

Work Packet 16 proved that the pipeline can train the 3.3M parameter model for 500 steps without leakage or resume failure. Work Packet 17 asks a narrower question:

```text
Does longer training on the same 250K corpus improve loss and generation evidence,
or does it mainly create repetition and overfitting?
```

## Boundaries

- Same approved 250K corpus.
- Same provisional 8K tokenizer.
- Same 3,307,200 parameter architecture.
- Same Stage A next-token objective.
- Separate run directory: `runs/stage_a_250k_extended_8k`.
- Milestone checkpoints and generation samples at steps 500, 1000, 1500, and 2000.
- `capability_claim_authorized=false`.
- `expansion_authorized=false`.

## Training Configuration

| Setting | Value |
|---|---:|
| Dataset | `stage_a_250k_probe_v0_1` |
| Tokens in dataset | 248,250 |
| Context | 128 |
| Tokens per optimizer step | 512 |
| Steps | 2,000 |
| Tokens processed | 1,024,000 |
| Checkpoint milestones | 500, 1000, 1500, 2000 |

## Local Commands

```powershell
.\p -m py_compile scripts\run_stage_a_250k_extended_probe.py scripts\verify_stage_a_250k_extended_probe.py tests\test_stage_a_250k_extended_probe.py
.\p -m unittest discover -s tests

.\p scripts\run_stage_a_250k_extended_probe.py
.\p scripts\run_stage_a_250k_extended_probe.py --execute --force

.\p scripts\verify_stage_a_250k_extended_probe.py --require-slice --require-dataset --require-run
```

## Decision Rule

If the final validation loss improves without a large validation gap or severe repetition, the next move can be a 500K Stage A corpus. If validation stalls, rises, or generation becomes more repetitive, improve the data mixture before scaling.
