# Leo to Ted — NeomaV1 Phase 3.5B Handoff

NeomaV1 is a small CPU-friendly coding model trained from random weights. The frozen engineering baseline is `phase-3.5-engineering-reviewed`. Phase 3.5B must prioritize clean task design, strong held-out evaluation, leakage prevention, and measured behavior rather than a large unreviewed corpus or an early `code_tiny_cpu` run.

Work Packet 01 adds planning and a locked 80-prompt evaluation suite only. It deliberately adds no training examples and authorizes no large training run. Review the master packet and quality gates before producing Work Packet 02.

## Work Packet 14 handoff

Baseline `35e6d17` now has tooling prepared for deterministic selection of an approximately 500K-token representative corpus, hash-bound human approval for tokenizer use only, and 2K/4K/8K tokenizer comparison. The local filtered corpora remain outside Git, so Leo must build and review the sample locally. Do not prepare model data or train Neoma until the tokenizer decision and real Stage B context-length report are recorded.

Leo's local execution rebalanced the sample plan after `typescript_website_2026` and stable self-knowledge underfilled their provisional quotas. The approved tokenizer-only sample has 1,818 records and 500,104 proxy tokens: 327 non-leaking frozen Stage B records, 827 repository-source records, 623 Wikimedia English records, and 41 stable self-knowledge records. Model-training permission remains false.

The 2K, 4K, and 8K tokenizer comparison passed hard gates for all candidates: zero round-trip failures, zero unknown-token hits, atomic protocol tags, and zero overlong tokens. The provisional recommendation is the 8K tokenizer for the next small Stage A probe because it gives the best frozen Stage B context fit, but it is not final until the probe confirms the capability/parameter trade-off.
