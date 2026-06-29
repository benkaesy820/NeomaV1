# NeomaV1 Phase 3.5B Corpus Audit

**Decision:** `targeted_gap_batch_recommended`

The curriculum remains below the 300-example lower bound; add one measured gap batch, not another broad batch.

## Scope

- Total instruction records: **291**
- Accepted Phase 3.5B JSONL records: **240**
- Legacy protocol records: **51**
- Non-instruction foundation seed files inventoried: **2**
- Length metric: **proxy_lexical_tokens**

## Languages

| Language | Count | Actual share | Target share | Status |
|---|---|---|---|---|
| text | 18 | 6.2% | 7.5% | near_target |
| javascript | 48 | 16.5% | 17.5% | near_target |
| typescript | 56 | 19.2% | 20.0% | near_target |
| python | 88 | 30.2% | 30.0% | near_target |
| sql | 40 | 13.8% | 12.5% | near_target |
| powershell | 41 | 14.1% | 12.5% | near_target |

## Primary categories

| Category | Count |
|---|---|
| data | 22 |
| database | 10 |
| debugging | 60 |
| efficiency | 34 |
| explanation | 16 |
| files | 26 |
| function | 14 |
| security | 36 |
| tests | 47 |
| validation | 26 |

## Overlapping capability coverage

Targets come from the 400-record plan and intentionally total 450 slots because one record may cover more than one capability.

| Capability | Covered records | Target slots | Progress |
|---|---|---|---|
| explanation | 29 | 30 | 96.7% |
| functions_and_data | 106 | 90 | 117.8% |
| sql_database | 53 | 45 | 117.8% |
| efficiency | 44 | 35 | 125.7% |
| tests | 76 | 55 | 138.2% |
| security | 50 | 35 | 142.9% |
| files_apis_automation | 74 | 50 | 148.0% |
| debugging | 102 | 55 | 185.5% |
| validation | 157 | 55 | 285.4% |

## Length and context fit

| Threshold | Fits | Exceeds |
|---|---|---|
| 128 | 28 | 263 |
| 192 | 81 | 210 |
| 256 | 179 | 112 |
| 512 | 290 | 1 |

Full-record length statistics:

```json
{
  "count": 291,
  "min": 83,
  "median": 233,
  "mean": 238.069,
  "p90": 335.0,
  "p95": 398.5,
  "max": 524
}
```

## Supervision

- Records with zero supervised tokens: **0**
- Median supervised percentage: **47.35%**
- Mean supervised percentage: **45.23%**

## Duplicate and leakage review

- Duplicate flags: **14**
- Exact duplicate flags: **0**
- Structural-similarity flags: **5**
- Critical leakage flags: **0**
- Review leakage flags: **0**
- Informational term overlaps: **110**

See the CSV reports before removing, merging, or rewriting any record. Flags require human review; they are not automatic deletion decisions.

## Integrity

| JSONL | Rows | Raw exists | Raw matches |
|---|---|---|---|
| data/incoming/phase3_5b_batch01_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch02_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch03_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch04_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch05_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch06_accepted.jsonl | 40 | True | True |

## Recommended order

1. Review every critical/review leakage flag.
2. Review exact and high structural duplicate flags.
3. Inspect records over the intended 256-token context.
4. Confirm language-by-category gaps.
5. Choose one targeted correction batch only from measured gaps.
6. Rerun this audit after that batch, then decide whether to freeze Instruction Corpus v0.1.
