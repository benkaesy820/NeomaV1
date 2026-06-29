# NeomaV1 Phase 3.5B Corpus Audit

**Decision:** `context_length_cleanup_required`

Too many records exceed the intended 256-token context and need shortening, splitting, or deferral.

## Scope

- Total instruction records: **331**
- Accepted Phase 3.5B JSONL records: **280**
- Legacy protocol records: **51**
- Non-instruction foundation seed files inventoried: **2**
- Length metric: **proxy_lexical_tokens**

## Languages

| Language | Count | Actual share | Target share | Status |
|---|---|---|---|---|
| python | 93 | 28.1% | 30.0% | near_target |
| typescript | 60 | 18.1% | 20.0% | near_target |
| javascript | 53 | 16.0% | 17.5% | near_target |
| text | 28 | 8.5% | 7.5% | near_target |
| sql | 47 | 14.2% | 12.5% | near_target |
| powershell | 50 | 15.1% | 12.5% | over |

## Primary categories

| Category | Count |
|---|---|
| data | 22 |
| database | 15 |
| debugging | 70 |
| efficiency | 36 |
| explanation | 21 |
| files | 29 |
| function | 14 |
| security | 46 |
| tests | 49 |
| validation | 29 |

## Overlapping capability coverage

Targets come from the 400-record plan and intentionally total 450 slots because one record may cover more than one capability.

| Capability | Covered records | Target slots | Progress |
|---|---|---|---|
| explanation | 35 | 30 | 116.7% |
| functions_and_data | 109 | 90 | 121.1% |
| efficiency | 47 | 35 | 134.3% |
| sql_database | 65 | 45 | 144.4% |
| tests | 85 | 55 | 154.6% |
| security | 60 | 35 | 171.4% |
| files_apis_automation | 96 | 50 | 192.0% |
| debugging | 119 | 55 | 216.4% |
| validation | 169 | 55 | 307.3% |

## Length and context fit

| Threshold | Fits | Exceeds |
|---|---|---|
| 128 | 33 | 298 |
| 192 | 121 | 210 |
| 256 | 219 | 112 |
| 512 | 330 | 1 |

Full-record length statistics:

```json
{
  "count": 331,
  "min": 83,
  "median": 222,
  "mean": 226.628,
  "p90": 326.0,
  "p95": 382.0,
  "max": 524
}
```

## Supervision

- Records with zero supervised tokens: **0**
- Median supervised percentage: **43.16%**
- Mean supervised percentage: **43.80%**

## Duplicate and leakage review

- Duplicate flags: **14**
- Exact duplicate flags: **0**
- Structural-similarity flags: **5**
- Critical leakage flags: **0**
- Review leakage flags: **0**
- Informational term overlaps: **116**

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
| data/incoming/phase3_5b_batch07_accepted.jsonl | 40 | True | True |

## Recommended order

1. Review every critical/review leakage flag.
2. Review exact and high structural duplicate flags.
3. Inspect records over the intended 256-token context.
4. Confirm language-by-category gaps.
5. Choose one targeted correction batch only from measured gaps.
6. Rerun this audit after that batch, then decide whether to freeze Instruction Corpus v0.1.
