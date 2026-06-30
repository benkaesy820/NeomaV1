# Phase 3.5B Work Packet 10 — Stage A Foundation Readiness and English Evaluation

## Baseline

`19a93e1` — Admit Phase 3.5B batch 07 examples

## Scope

This packet begins Stage A without downloading or admitting training data. It freezes the evaluation boundary, defines the production source pool, creates provenance and review structures, and adds deterministic validation tooling.

## Included

- 48-prompt English/code development suite;
- 48-prompt locked English/code suite;
- deterministic choice scorer;
- source-manifest schema and candidate manifest;
- internal authoring component manifest;
- source review template;
- Stage A production plan and internal authoring plan;
- Stage A workspace README and Git exclusions;
- corrected Instruction Corpus v0.1 baseline;
- readiness tests and validation report.

## Explicitly not included

- no external source download;
- no internally authored training corpus;
- no tokenizer rebuild;
- no processed dataset;
- no training run;
- no change to the 331 frozen instruction records.

## Leo review gates

1. Confirm all 96 evaluation prompts are independent, clear, and permanently excluded from training.
2. Confirm source versions, release channels, allowed paths, exclusions, and token targets.
3. Verify licenses and acquisition methods before setting any source to training-allowed.
4. Confirm the 50M-token target remains appropriate for available CPU time and the first model size.
5. Approve the internal authoring protocol before candidate generation begins.
6. Commit only the reviewed planning, evaluation, and tooling files.

## Next packet after admission

Work Packet 11 should acquire and hash the ten source snapshots or stream manifests, without yet building the final corpus. Acquisition and content admission remain separate decisions.
