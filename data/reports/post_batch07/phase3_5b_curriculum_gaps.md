# NeomaV1 Curriculum Gap Analysis

This report is diagnostic. It does not authorize generation or admission of another batch.

## Language balance

| Language | Count | Delta at current size | Status |
|---|---|---|---|
| python | 93 | -6.3 | near_target |
| typescript | 60 | -6.2 | near_target |
| javascript | 53 | -4.92 | near_target |
| text | 28 | 3.18 | near_target |
| sql | 47 | 5.62 | near_target |
| powershell | 50 | 8.62 | over |

## Overlapping capability coverage

| Capability | Covered records | Remaining target slots |
|---|---|---|
| explanation | 35 | 0 |
| functions_and_data | 109 | 0 |
| efficiency | 47 | 0 |
| sql_database | 65 | 0 |
| tests | 85 | 0 |
| security | 60 | 0 |
| files_apis_automation | 96 | 0 |
| debugging | 119 | 0 |
| validation | 169 | 0 |

## Language × primary-category matrix

| Language | data | database | debugging | efficiency | explanation | files | function | security | tests | validation |
|---|---|---|---|---|---|---|---|---|---|---|
| javascript | 5 | 0 | 14 | 3 | 1 | 4 | 5 | 7 | 9 | 5 |
| powershell | 2 | 0 | 8 | 15 | 0 | 7 | 1 | 9 | 5 | 3 |
| python | 5 | 0 | 24 | 6 | 0 | 14 | 6 | 12 | 17 | 9 |
| sql | 3 | 13 | 9 | 7 | 0 | 0 | 0 | 8 | 5 | 2 |
| text | 0 | 2 | 1 | 1 | 19 | 0 | 0 | 2 | 2 | 1 |
| typescript | 7 | 0 | 14 | 4 | 1 | 4 | 2 | 8 | 11 | 9 |

## Decision rule

Current automated recommendation: **`context_length_cleanup_required`**.

Too many records exceed the intended 256-token context and need shortening, splitting, or deferral.

The final gap batch topic must be selected after human review of duplicate flags, context outliers, leakage flags, and language-category coverage. Do not choose it from counts alone.
