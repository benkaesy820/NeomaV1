# NeomaV1 Phase 3.5B Corpus Audit

**Decision:** `structural_cleanup_required`

Critical leakage or exact duplicate flags must be resolved before adding more data.

## Scope

- Total instruction records: **213**
- Accepted Phase 3.5B JSONL records: **160**
- Legacy protocol records: **53**
- Non-instruction foundation seed files inventoried: **2**
- Length metric: **proxy_lexical_tokens**

## Languages

| Language | Count | Actual share | Target share | Status |
|---|---|---|---|---|
| powershell | 21 | 9.9% | 12.5% | under |
| text | 11 | 5.2% | 7.5% | under |
| sql | 25 | 11.7% | 12.5% | near_target |
| typescript | 44 | 20.7% | 20.0% | near_target |
| javascript | 39 | 18.3% | 17.5% | near_target |
| python | 73 | 34.3% | 30.0% | over |

## Primary categories

| Category | Count |
|---|---|
| data | 22 |
| database | 8 |
| debugging | 58 |
| efficiency | 1 |
| explanation | 9 |
| files | 26 |
| function | 14 |
| security | 2 |
| tests | 47 |
| validation | 26 |

## Overlapping capability coverage

Targets come from the 400-record plan and intentionally total 450 slots because one record may cover more than one capability.

| Capability | Covered records | Target slots | Progress |
|---|---|---|---|
| efficiency | 7 | 35 | 20.0% |
| security | 10 | 35 | 28.6% |
| explanation | 18 | 30 | 60.0% |
| sql_database | 29 | 45 | 64.4% |
| files_apis_automation | 49 | 50 | 98.0% |
| functions_and_data | 93 | 90 | 103.3% |
| debugging | 64 | 55 | 116.4% |
| tests | 64 | 55 | 116.4% |
| validation | 111 | 55 | 201.8% |

## Length and context fit

| Threshold | Fits | Exceeds |
|---|---|---|
| 128 | 29 | 184 |
| 192 | 72 | 141 |
| 256 | 136 | 77 |
| 512 | 213 | 0 |

Full-record length statistics:

```json
{
  "count": 213,
  "min": 83,
  "median": 225,
  "mean": 225.883,
  "p90": 320.2,
  "p95": 361.2,
  "max": 458
}
```

## Supervision

- Records with zero supervised tokens: **0**
- Median supervised percentage: **48.23%**
- Mean supervised percentage: **46.34%**

## Duplicate and leakage review

- Duplicate flags: **20**
- Exact duplicate flags: **2**
- Structural-similarity flags: **6**
- Critical leakage flags: **0**
- Review leakage flags: **0**
- Informational term overlaps: **88**

See the CSV reports before removing, merging, or rewriting any record. Flags require human review; they are not automatic deletion decisions.

## Integrity

| JSONL | Rows | Raw exists | Raw matches |
|---|---|---|---|
| data/incoming/phase3_5b_batch01_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch02_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch03_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch04_accepted.jsonl | 40 | True | True |

## Recommended order

1. Review every critical/review leakage flag.
2. Review exact and high structural duplicate flags.
3. Inspect records over the intended 256-token context.
4. Confirm language-by-category gaps.
5. Choose one targeted correction batch only from measured gaps.
6. Rerun this audit after that batch, then decide whether to freeze Instruction Corpus v0.1.
