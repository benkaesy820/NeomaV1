# NeomaV1 Curriculum Gap Analysis

This report is diagnostic. It does not authorize generation or admission of another batch.

## Language balance

| Language | Count | Delta at current size | Status |
|---|---|---|---|
| powershell | 26 | -5.38 | under |
| text | 14 | -4.82 | near_target |
| sql | 31 | -0.38 | near_target |
| javascript | 45 | 1.08 | near_target |
| typescript | 52 | 1.8 | near_target |
| python | 83 | 7.7 | over |

## Overlapping capability coverage

| Capability | Covered records | Remaining target slots |
|---|---|---|
| efficiency | 8 | 27 |
| explanation | 24 | 6 |
| sql_database | 36 | 9 |
| functions_and_data | 100 | 0 |
| debugging | 66 | 0 |
| tests | 73 | 0 |
| files_apis_automation | 68 | 0 |
| security | 48 | 0 |
| validation | 144 | 0 |

## Language × primary-category matrix

| Language | data | database | debugging | efficiency | explanation | files | function | security | tests | validation |
|---|---|---|---|---|---|---|---|---|---|---|
| javascript | 5 | 0 | 12 | 0 | 0 | 4 | 5 | 6 | 9 | 4 |
| powershell | 2 | 0 | 6 | 0 | 0 | 4 | 1 | 5 | 5 | 3 |
| python | 5 | 0 | 22 | 1 | 0 | 13 | 6 | 11 | 16 | 9 |
| sql | 3 | 7 | 8 | 0 | 0 | 0 | 0 | 6 | 5 | 2 |
| text | 0 | 1 | 0 | 0 | 12 | 0 | 0 | 0 | 1 | 0 |
| typescript | 7 | 0 | 12 | 0 | 0 | 4 | 2 | 8 | 11 | 8 |

## Decision rule

Current automated recommendation: **`targeted_gap_batch_recommended`**.

The curriculum remains below the 300-example lower bound; add one measured gap batch, not another broad batch.

The final gap batch topic must be selected after human review of duplicate flags, context outliers, leakage flags, and language-category coverage. Do not choose it from counts alone.
