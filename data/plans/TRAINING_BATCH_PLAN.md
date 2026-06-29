# Phase 3.5B Training-Data Batch Plan

## Target

Create **400 reviewed instruction examples** as ten batches of 40. Stop at 300 if evaluation already identifies a clear bottleneck; extend toward 600 only when additional coverage is justified. Generation is not admission: every line must pass automated gates and human/model review before import.

## Overall language budget for 400 examples

| Language | Count | Share |
|---|---:|---:|
| Python | 120 | 30% |
| TypeScript | 80 | 20% |
| JavaScript | 70 | 17.5% |
| PowerShell | 50 | 12.5% |
| SQL | 50 | 12.5% |
| Text/explanation | 30 | 7.5% |
| **Total** | **400** | **100%** |

## Category budget

| Capability | Target |
|---|---:|
| Functions and data transformations | 90 |
| Validation and boundary handling | 55 |
| Debugging and repair | 55 |
| Tests | 55 |
| Files, APIs, and automation | 50 |
| SQL/database work | 45 |
| Security and safe defaults | 35 |
| Efficiency and optimization | 35 |
| Concise explanations | 30 |
| **Total** | **450 slots** |

The category total exceeds 400 because a reviewed example may carry one primary category and one secondary coverage label. The import schema retains one primary `category`; the review manifest may store secondary labels separately.

## Ten batches

### Batch 01 — Core functions and exact validation
40 small tasks: input normalization, strict numeric parsing, required fields, stable transformations, explicit errors. Avoid all locked eval task formulations.

### Batch 02 — Files and structured data
40 tasks: UTF-8 text, JSON, CSV, JSONL, path handling, non-destructive filesystem inspection. No path-traversal task may mirror the held-out suite.

### Batch 03 — Debugging fundamentals
40 broken snippets with one or two local defects: control flow, mutation, async return, join fan-out, PowerShell exit handling. Require `bad_code` and short practical `reasoning`.

### Batch 04 — Tests
40 tasks that write tests for previously unseen helpers. Cover boundaries, invalid inputs, immutability, determinism, and failure messages. Prefer built-in test tools.

### Batch 05 — Data structures and lookup behavior
40 tasks teaching dictionaries/maps/sets, grouping, indexing, counters, stable ordering, and clear duplicate policies. Include correctness-first versions before optimization variants.

### Batch 06 — APIs and boundary security
40 framework-light tasks: parse unknown request bodies, safe public errors, authentication assumptions, redaction, parameter binding, and explicit timeouts. Network execution is not required.

### Batch 07 — SQL and persistence
40 SQL tasks across schema constraints, joins, aggregation, transactions, indexes, pagination, and data-quality queries. Every record must state its SQL dialect.

### Batch 08 — PowerShell and local automation
40 non-destructive tasks: validation, pipeline behavior, object output, filesystem reporting, native command checks, and safe previews. No deletion, registry changes, or service control.

### Batch 09 — Efficiency curriculum
40 tasks where the naive and improved solutions are both understandable. Teach the practical bottleneck, not slogans. Require `bad_code` and `reasoning`; include cases where optimization is unnecessary.

### Batch 10 — Mixed integration and concise explanations
40 small multi-constraint tasks that combine validation, implementation, tests, and explanation without becoming framework-heavy. Use this batch to fill measured coverage gaps, not arbitrary quotas.

## Per-batch admission workflow

1. Generate 40–50 candidates to admit no more than 40.
2. Validate JSONL and IDs.
3. Run secret/destructive/network scans.
4. Parse or compile code when tooling exists.
5. Execute approved local tests in isolation.
6. Compare against all eval prompts and admitted training examples.
7. Review correctness, clarity, balance, and provenance.
8. Import only accepted rows.
9. Write a batch manifest with accepted, rejected, and revised counts.
10. Freeze the admitted batch before the next model comparison.

## Example quality rules

- One main teaching objective, with at most one supporting objective.
- Small complete answer; no ellipses or “implementation omitted.”
- Explicit invalid-input behavior.
- Standard library or built-ins unless a dependency is the point of the task.
- No secrets, personal data, destructive operations, or hidden network use.
- No copy, paraphrase, solution, or edge-case bundle from the held-out suite.
- Do not multiply one task by renaming variables or changing constants.
- Efficiency examples must state the workload that makes the optimization relevant.
