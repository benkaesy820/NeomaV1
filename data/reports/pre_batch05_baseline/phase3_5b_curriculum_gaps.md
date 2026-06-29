# NeomaV1 Curriculum Gap Analysis

This report is diagnostic. It does not authorize generation or admission of another batch.

## Language balance

| Language | Count | Delta at current size | Status |
|---|---|---|---|
| powershell | 21 | -5.62 | under |
| text | 11 | -4.97 | under |
| sql | 25 | -1.62 | near_target |
| typescript | 44 | 1.4 | near_target |
| javascript | 39 | 1.73 | near_target |
| python | 73 | 9.1 | over |

## Overlapping capability coverage

| Capability | Covered records | Remaining target slots |
|---|---|---|
| efficiency | 7 | 28 |
| security | 10 | 25 |
| explanation | 18 | 12 |
| sql_database | 29 | 16 |
| files_apis_automation | 49 | 1 |
| functions_and_data | 93 | 0 |
| debugging | 64 | 0 |
| tests | 64 | 0 |
| validation | 111 | 0 |

## Language × primary-category matrix

| Language | data | database | debugging | efficiency | explanation | files | function | security | tests | validation |
|---|---|---|---|---|---|---|---|---|---|---|
| javascript | 5 | 0 | 11 | 0 | 0 | 5 | 5 | 0 | 9 | 4 |
| powershell | 2 | 0 | 6 | 0 | 0 | 4 | 1 | 0 | 5 | 3 |
| python | 5 | 0 | 22 | 1 | 0 | 13 | 6 | 1 | 16 | 9 |
| sql | 3 | 7 | 8 | 0 | 0 | 0 | 0 | 0 | 5 | 2 |
| text | 0 | 1 | 0 | 0 | 9 | 0 | 0 | 0 | 1 | 0 |
| typescript | 7 | 0 | 11 | 0 | 0 | 4 | 2 | 1 | 11 | 8 |

## Decision rule

Current automated recommendation: **`structural_cleanup_required`**.

Critical leakage or exact duplicate flags must be resolved before adding more data.

The final gap batch topic must be selected after human review of duplicate flags, context outliers, leakage flags, and language-category coverage. Do not choose it from counts alone.
