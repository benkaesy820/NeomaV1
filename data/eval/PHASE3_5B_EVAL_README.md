# Phase 3.5B Held-Out Evaluation Suite

`phase3_5b_heldout_v1.jsonl` contains **80 locked evaluation prompts**.

## Hard rule

These prompts, their test code, expected-behavior notes, close paraphrases, and direct solutions must never be imported into `data/raw`, `data/incoming`, tokenizer training examples, foundation corpus files, or instruction-training batches.

The file is an evaluation asset, not training material. Every row therefore includes:

- `split: "heldout"`
- `training_allowed: false`
- a stable unique `id`
- language and category labels
- optional machine-checkable test code
- lightweight required/forbidden-term checks
- expected behavior notes for human review

## Suite composition

| Language | Count |
|---|---:|
| Python | 24 |
| TypeScript | 14 |
| JavaScript | 12 |
| PowerShell | 10 |
| SQL | 12 |
| Text/explanation | 8 |
| **Total** | **80** |

The suite deliberately covers implementation, files, data handling, debugging, tests, validation, security, efficiency, database work, and concise explanations. It avoids copies of the original 12 evaluation prompts and the existing 53 training instructions.

## Scoring layers

1. **Generation health:** non-empty output, EOS completion, no protocol leakage, no pathological repetition.
2. **Language shape:** requested language, expected syntax markers, forbidden unsafe markers.
3. **Parse/static checks:** Python AST, PowerShell parser, TypeScript/JavaScript parser when installed, SQL parser or dialect-aware linting.
4. **Executable checks:** run only explicitly approved local test code in a restricted subprocess with network disabled and a short timeout.
5. **Human rubric:** correctness, constraint following, clarity, security, and practical efficiency.

`required_terms` and `forbidden_terms` are weak signals, not a complete correctness oracle. A valid alternative solution must not be rejected solely because it uses different wording unless the term is essential to the task.

## Versioning

Do not silently edit a locked suite after results exist. Corrections create a new file such as `phase3_5b_heldout_v2.jsonl`, with a changelog and a mapping of changed IDs. Preserve prior outputs for comparison.
