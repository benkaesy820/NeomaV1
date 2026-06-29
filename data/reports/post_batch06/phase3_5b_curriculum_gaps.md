# NeomaV1 Curriculum Gap Analysis

This report is diagnostic. It does not authorize generation or admission of another batch.

## Language balance

| Language | Count | Delta at current size | Status |
|---|---|---|---|
| text | 18 | -3.82 | near_target |
| javascript | 48 | -2.92 | near_target |
| typescript | 56 | -2.2 | near_target |
| python | 88 | 0.7 | near_target |
| sql | 40 | 3.62 | near_target |
| powershell | 41 | 4.62 | near_target |

## Overlapping capability coverage

| Capability | Covered records | Remaining target slots |
|---|---|---|
| explanation | 29 | 1 |
| functions_and_data | 106 | 0 |
| sql_database | 53 | 0 |
| efficiency | 44 | 0 |
| tests | 76 | 0 |
| security | 50 | 0 |
| files_apis_automation | 74 | 0 |
| debugging | 102 | 0 |
| validation | 157 | 0 |

## Language × primary-category matrix

| Language | data | database | debugging | efficiency | explanation | files | function | security | tests | validation |
|---|---|---|---|---|---|---|---|---|---|---|
| javascript | 5 | 0 | 12 | 3 | 0 | 4 | 5 | 6 | 9 | 4 |
| powershell | 2 | 0 | 6 | 14 | 0 | 5 | 1 | 5 | 5 | 3 |
| python | 5 | 0 | 22 | 6 | 0 | 13 | 6 | 11 | 16 | 9 |
| sql | 3 | 9 | 8 | 7 | 0 | 0 | 0 | 6 | 5 | 2 |
| text | 0 | 1 | 0 | 0 | 16 | 0 | 0 | 0 | 1 | 0 |
| typescript | 7 | 0 | 12 | 4 | 0 | 4 | 2 | 8 | 11 | 8 |

## Decision rule

Current automated recommendation: **`targeted_gap_batch_recommended`**.

The curriculum remains below the 300-example lower bound; add one measured gap batch, not another broad batch.

The final gap batch topic must be selected after human review of duplicate flags, context outliers, leakage flags, and language-category coverage. Do not choose it from counts alone.
