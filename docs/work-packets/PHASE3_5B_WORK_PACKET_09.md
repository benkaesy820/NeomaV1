# Phase 3.5B Work Packet 09

## Engineering Judgment and Task Decision Fundamentals

### Baseline

This packet was prepared against repository baseline `7eadf9c` and was also checked against all 48 Work Packet 08 candidates. It is designed so Leo can audit Work Packets 08 and 09 together before either candidate batch is admitted.

### Scope

Work Packet 09 contains:

- 48 review-only candidates;
- a maximum intended admission of 40;
- no direct addition to `data/raw`;
- no tokenizer rebuild;
- no dataset preparation;
- no training run;
- a separate read-only English-and-Code Stage A foundation plan;
- eight illustrative planning samples marked `training_allowed=false`.

This is the final planned instruction batch before a complete post-admission audit and an Instruction Corpus v0.1 freeze decision.

## Why this packet exists

Earlier batches teach implementation, files, debugging, tests, security, and efficiency. This packet teaches the decision layer that makes a coding assistant behave like a careful junior engineer:

- ask only when missing information materially changes correctness or safety;
- otherwise proceed directly or state a reversible assumption;
- prefer the smallest correct change;
- gather evidence before claiming a bug cause;
- distinguish blocking defects from style preferences;
- scale test scope to risk;
- avoid optimization without evidence;
- protect user work before destructive or broad operations;
- plan cross-system, migration, and compatibility changes;
- calibrate uncertainty instead of inventing facts.

## Distribution

| Language | Candidates |
|---|---:|
| Text / judgment | 14 |
| PowerShell | 10 |
| PostgreSQL | 8 |
| Python | 6 |
| TypeScript | 5 |
| JavaScript | 5 |
| **Total** | **48** |

## Response-mode balance

| Mode | Candidates |
|---|---:|
| Proceed directly | 12 |
| Ask a material clarification | 8 |
| Plan before acting | 6 |
| Proceed with a stated assumption | 6 |
| Investigate evidence first | 5 |
| Warn or request confirmation | 5 |
| Decline or refuse an unsafe/impossible request | 3 |
| Recommend no change | 3 |

This balance is intentional. The packet must not train Neoma to answer every request with “please clarify” or “it depends.”

## Files

- `PHASE3_5B_WORK_PACKET_09.md`
- `APPLY_OVERLAY.md`
- `data/incoming/phase3_5b_batch07_candidates.jsonl`
- `data/reviews/phase3_5b_batch07_candidate_validation.json`
- `data/reviews/phase3_5b_batch07_review_template.csv`
- `data/plans/ENGLISH_CODE_FOUNDATION_CORPUS_PLAN.md`
- `data/plans/english_code_foundation_planning_sample.jsonl`

## Validation summary

- 48 records and 48 unique IDs.
- Importer schema validation passed.
- Code-bearing Python snippets passed `ast.parse`.
- Code-bearing JavaScript snippets passed `node --check`.
- Code-bearing TypeScript snippets passed strict `tsc --noEmit` with TypeScript 5.8.3; Leo must rerun with 6.0.3.
- Four selected deterministic behavior checks passed.
- PowerShell requires Leo's local `Parser.ParseInput` check.
- PostgreSQL requires Leo's final dialect and semantic review.
- Exact instruction overlap: zero.
- Exact answer overlap: zero.
- References included 251 admitted records, both eval suites and auxiliary fields, and all Work Packet 08 candidates.
- Maximum instruction trigram overlap: about 0.067.
- Maximum answer trigram overlap: about 0.172.
- All 48 records fit within 192 lexical proxy tokens; final tokenizer counts may differ.
- No secret-shaped values, network calls in answers, or executable destructive commands in answers were found.
- Temporary combined gates checked 347 records with zero warnings against both evaluation suites.
- Existing project tests: 18 passed.
- Admitted-corpus audit remained at 251 records, proving candidate files and planning samples are excluded.

## Leo review priorities

1. Reject answers that are vague, excessively cautious, or unable to commit to a decision.
2. Confirm each clarification is truly necessary.
3. Confirm every assumption is low-risk, explicit, and reversible.
4. Confirm destructive scenarios preview or protect data before mutation.
5. Check PowerShell parsing and PostgreSQL semantics locally.
6. Rerun TypeScript with 6.0.3.
7. Admit no more than 40; accepting fewer is correct.
8. After admission, rerun the complete corpus audit before freezing Instruction Corpus v0.1.

## Stage A planning status

The included foundation plan is read-only. It defines the mixture, source rules, leakage controls, segmentation, tokenizer comparison, and 25k/250k/500k/1M-token ladder. The eight sample rows illustrate source styles only. They are not approved training or tokenizer data.
