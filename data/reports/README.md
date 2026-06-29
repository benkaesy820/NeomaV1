# Corpus Audit Reports

`pre_batch05_baseline/` is a reproducible preview generated from baseline `b771d01`, before Work Packet 06 admission. It proves the audit tooling runs and gives Leo concrete findings to inspect while reviewing Work Packets 06 and 07 together.

Recommended order:

1. Reproduce the baseline audit.
2. Review and resolve or document blocking exact duplicates.
3. Audit and admit only approved Batch 05 rows.
4. Regenerate reports under `data/reports/post_batch05/`.
5. Complete the generated `phase3_5b_human_review_queue.csv`.
6. Choose at most one final targeted batch from measured gaps.

Run:

```powershell
.\p scripts\audit_instruction_corpus.py --out-dir data\reports\post_batch05 --expected-context 256
```

When a project tokenizer exists, pass it explicitly and generate a separate report:

```powershell
.\p scripts\audit_instruction_corpus.py --out-dir data\reports\post_batch05_tokenizer --tokenizer data\tokenizer.json --expected-context 256
```

The audit command never trains or mutates a tokenizer. Lexical-proxy token counts are provisional and must not be used alone to rewrite or reject records.
