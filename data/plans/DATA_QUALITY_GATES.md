# Phase 3.5B Data Quality Gates

A candidate enters training only after every mandatory gate passes. Warnings require review; they are not automatically ignored.

## Gate 1 — Schema and identity

- valid UTF-8 JSONL, one object per line;
- unique normalized ID across all batches;
- allowed language, category, and difficulty;
- non-empty instruction and answer;
- lists contain strings only;
- length limits and no control-character corruption.

## Gate 2 — Safety

Reject detected secrets, private keys, realistic credentials, personal data, destructive filesystem/database/system actions, hidden subprocess execution, and unmarked network dependencies. Examples that explain unsafe code may contain it only in `bad_code`, with an explicitly safe answer.

## Gate 3 — Syntax and static validity

- Python: `ast.parse`;
- JavaScript/TypeScript: parser or compiler check when the runtime is installed;
- PowerShell: parser check without execution;
- SQL: dialect recorded and parser/linter check where possible;
- text answers: formatting and length checks.

A parser pass does not prove correctness; it only admits the candidate to later gates.

## Gate 4 — Executable behavior

Run only approved self-contained tests in a restricted subprocess:

- network disabled;
- temporary working directory;
- sanitized environment;
- 3-second default timeout;
- output and memory caps;
- no access to user files;
- reject flaky or time-dependent behavior.

## Gate 5 — Duplicate and leakage controls

Normalize case and whitespace, then compute:

- exact normalized match;
- token 3-gram Jaccard;
- character 5-gram cosine or MinHash approximation;
- identifier-normalized structural fingerprint for code;
- top-five nearest held-out eval prompts and training examples.

Default decisions:

| Comparison | Reject | Manual review |
|---|---:|---:|
| Eval token 3-gram Jaccard | >= 0.80 | 0.65–0.80 |
| Eval character 5-gram cosine | >= 0.88 | 0.78–0.88 |
| Training near-duplicate similarity | >= 0.90 | 0.80–0.90 |

These thresholds are deliberately conservative starting points. Human semantic review overrides them in either direction. Variable renaming or constant changes do not make an example new.

## Gate 6 — Correctness review

Review the answer against every constraint and edge case. Check mutation behavior, error types, async completion, SQL fan-out, null handling, path boundaries, data leakage, algorithmic claims, and test quality. Revise rather than rationalize questionable examples.

## Gate 7 — Balance

After every batch, report counts by language, primary category, difficulty, answer length, test availability, and source. Block admission when a category is unintentionally more than 20% over its cumulative target or a planned category is below 70% of its target without a written reason.

## Gate 8 — Provenance and manifest

Record candidate ID, content hash, generator/source, reviewer, gate results, nearest eval IDs, revision count, acceptance decision, and destination file. Generated material is marked generated even after review.

## Eval protection

The held-out suite directory must be excluded from every tokenizer, collector, importer, and training glob. CI should fail when a held-out ID or normalized prompt appears under training paths.

## Binary masking decision

Keep Stage B answer masks binary for the first baseline. Weighted supervision is a later controlled experiment, not an implicit data feature. If introduced, store per-token or span weights explicitly and test that total supervised weight is nonzero.
