# Phase 3.5 Engineering Patch

For the complete file-by-file engineering handoff, phase-label clarification,
and remaining decision points, see `UPDATES.md`.

This patch strengthens NeomaV1 before larger CPU training runs.

It should be called Phase 3.5, not completed Phase 4. Phase 4 should begin only
after we have a larger, reviewed capability dataset, stronger evaluation
coverage, and evidence that training improves real coding behavior instead of
only memorizing the seed corpus.

## Changes

- Added resumable checkpoints with optimizer state, step, best loss, and PyTorch RNG state.
- Added `--resume` and `--auto-resume` to `scripts/train.py`.
- Added safe checkpoint saving through a temporary file and atomic replacement.
- Added automatic checkpoint saving when training is interrupted with `Ctrl+C`.
- Corrected the one-based warm-up learning-rate calculation.
- Added token-per-second reporting.
- Precomputed one shared RoPE cache per model instead of rebuilding it in every layer and forward pass.
- Uses native PyTorch grouped-query attention when available, with a compatibility fallback.
- Added GPT-style residual projection scaling at initialization.
- Added optional target loss masks to the model and training loop.
- Changed dataset preparation to split complete records/examples instead of slicing instruction text at an arbitrary token boundary.
- Added optional answer-only instruction masks with `--instruction-loss-mask`.
- Added a dataset manifest showing the exact train/validation records and token totals.
- Changed the tokenizer's default preset from `code` to `base`; code workflows now request `--preset code` explicitly.
- Added EOS-aware generation and deterministic greedy generation with `--temperature 0`.
- Made generation restore the model's previous train/eval state.
- Expanded tests for causal attention, masked loss, RoPE checkpoint behavior, record splitting, and instruction masks.

## Data warning

The included Phase 3 corpus is still a pipeline-validation dataset, not enough
data for the 20,000-step `code_tiny_cpu` run. Expand and rebalance the corpus
before attempting the largest configuration.

## Decisions still needed before Phase 4

- Capability dataset scope: which coding languages, task types, and difficulty bands come first.
- Data acceptance rules: how strict we are about synthetic examples from other models.
- Evaluation scoring: which prompts and pass/fail checks decide whether the model improved.
- Training target: whether to continue with answer-only masks for all instruction records or mix some full-record supervision.
- Tokenizer retest point: how much new data must exist before retraining and re-benchmarking the tokenizer.
