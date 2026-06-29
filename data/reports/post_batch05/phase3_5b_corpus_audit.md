# NeomaV1 Phase 3.5B Corpus Audit

**Decision:** `targeted_gap_batch_recommended`

The curriculum remains below the 300-example lower bound; add one measured gap batch, not another broad batch.

## Scope

- Total instruction records: **251**
- Accepted Phase 3.5B JSONL records: **200**
- Legacy protocol records: **51**
- Non-instruction foundation seed files inventoried: **2**
- Length metric: **proxy_lexical_tokens**

## Languages

| Language | Count | Actual share | Target share | Status |
|---|---|---|---|---|
| powershell | 26 | 10.4% | 12.5% | under |
| text | 14 | 5.6% | 7.5% | near_target |
| sql | 31 | 12.3% | 12.5% | near_target |
| javascript | 45 | 17.9% | 17.5% | near_target |
| typescript | 52 | 20.7% | 20.0% | near_target |
| python | 83 | 33.1% | 30.0% | over |

## Primary categories

| Category | Count |
|---|---|
| data | 22 |
| database | 8 |
| debugging | 60 |
| efficiency | 1 |
| explanation | 12 |
| files | 25 |
| function | 14 |
| security | 36 |
| tests | 47 |
| validation | 26 |

## Overlapping capability coverage

Targets come from the 400-record plan and intentionally total 450 slots because one record may cover more than one capability.

| Capability | Covered records | Target slots | Progress |
|---|---|---|---|
| efficiency | 8 | 35 | 22.9% |
| explanation | 24 | 30 | 80.0% |
| sql_database | 36 | 45 | 80.0% |
| functions_and_data | 100 | 90 | 111.1% |
| debugging | 66 | 55 | 120.0% |
| tests | 73 | 55 | 132.7% |
| files_apis_automation | 68 | 50 | 136.0% |
| security | 48 | 35 | 137.1% |
| validation | 144 | 55 | 261.8% |

## Length and context fit

| Threshold | Fits | Exceeds |
|---|---|---|
| 128 | 28 | 223 |
| 192 | 77 | 174 |
| 256 | 147 | 104 |
| 512 | 250 | 1 |

Full-record length statistics:

```json
{
  "count": 251,
  "min": 83,
  "median": 238,
  "mean": 239.378,
  "p90": 345.0,
  "p95": 405.5,
  "max": 524
}
```

## Supervision

- Records with zero supervised tokens: **0**
- Median supervised percentage: **50.70%**
- Mean supervised percentage: **47.54%**

## Duplicate and leakage review

- Duplicate flags: **14**
- Exact duplicate flags: **0**
- Structural-similarity flags: **5**
- Critical leakage flags: **0**
- Review leakage flags: **0**
- Informational term overlaps: **98**

See the CSV reports before removing, merging, or rewriting any record. Flags require human review; they are not automatic deletion decisions.

## Integrity

| JSONL | Rows | Raw exists | Raw matches |
|---|---|---|---|
| data/incoming/phase3_5b_batch01_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch02_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch03_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch04_accepted.jsonl | 40 | True | True |
| data/incoming/phase3_5b_batch05_accepted.jsonl | 40 | True | True |

## Recommended order

1. Review every critical/review leakage flag.
2. Review exact and high structural duplicate flags.
3. Inspect records over the intended 256-token context.
4. Confirm language-by-category gaps.
5. Choose one targeted correction batch only from measured gaps.
6. Rerun this audit after that batch, then decide whether to freeze Instruction Corpus v0.1.
