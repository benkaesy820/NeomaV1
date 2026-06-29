# Data Collection Prompts

Use this when asking another model to create training examples for this project.

The goal is not to get flashy answers. The goal is to collect small, correct,
boring, high-quality coding examples that teach the tiny model how English
requests map to useful code.

## Important Rules

- Ask for JSONL only.
- One example per line.
- Do not ask for huge examples.
- Prefer small functions, tests, bug fixes, and explanations.
- Do not include secrets, real API keys, private tokens, personal data, or copied
  proprietary code.
- Prefer original examples created for this dataset.
- Keep examples easy to verify.
- Include edge cases deliberately.
- Avoid unnecessary dependencies.
- Avoid framework-heavy examples unless the task is specifically about a
  framework.

## Best Use

Ask another model for one batch at a time:

```text
20 Python examples
20 TypeScript examples
20 JavaScript examples
10 PowerShell examples
10 SQL examples
10 debugging examples
10 test-writing examples
10 efficiency examples
10 security/boundary-validation examples
```

Bring the JSONL back into this project under:

```text
data/incoming/
```

Then import it with:

```powershell
.\p scripts/import_instruction_jsonl.py data/incoming/examples.jsonl --out data/raw/imported_examples.txt
.\p scripts/check_training_data.py --raw data/raw --eval data/eval/code_prompts.jsonl
```

## Prompt For Other Models

Copy this whole prompt into another model:

```text
You are creating original training data for a tiny from-scratch coding language model.

Return JSONL only. Do not use Markdown fences. Do not add commentary before or after the JSONL.

Each line must be one valid JSON object with this schema:

{
  "id": "short_unique_snake_case_id",
  "language": "python|typescript|javascript|powershell|sql|text",
  "category": "function|files|api|tests|debugging|data|database|security|efficiency|explanation",
  "difficulty": "basic|intermediate",
  "instruction": "Clear English coding task.",
  "constraints": ["constraint 1", "constraint 2"],
  "answer": "Complete, correct answer code or explanation.",
  "bad_code": "Optional broken code for debugging/refactor examples only.",
  "reasoning": "Optional short reason for debugging/efficiency/security examples.",
  "edge_cases": ["edge case 1", "edge case 2"],
  "quality_notes": ["why this example is good"]
}

Rules:
- Produce original examples written for this request.
- Do not copy proprietary code or real project code.
- Do not include secrets, API keys, tokens, passwords, cookies, private URLs, or personal data.
- Keep each answer small enough to inspect.
- Prefer standard libraries and built-in language features.
- Use clear names.
- Validate external input at boundaries.
- Use explicit errors for invalid input.
- Do not log sensitive values.
- Include tests when the category is tests.
- Include bad_code and reasoning when the category is debugging or efficiency.
- Keep reasoning short and practical.
- Use idempotent/retry-safe patterns where relevant.
- Avoid overengineering.

Coverage request:
- Generate 30 examples.
- Include at least 8 Python examples.
- Include at least 6 TypeScript examples.
- Include at least 5 JavaScript examples.
- Include at least 3 PowerShell examples.
- Include at least 3 SQL examples.
- Include at least 3 debugging examples.
- Include at least 3 test-writing examples.
- Include at least 3 efficiency examples.
- Include at least 3 security or input-boundary examples.

Do not repeat the same task with different names.
Do not include tasks about machine learning, tokenizers, or language models.
Focus on practical software engineering tasks.
```

## Prompt For Python-Only Batch

```text
You are creating original Python training data for a tiny from-scratch coding language model.

Return JSONL only. No Markdown fences. No commentary.

Each line must be one valid JSON object with:
id, language, category, difficulty, instruction, constraints, answer, bad_code, reasoning, edge_cases, quality_notes.

Generate 40 Python examples covering:
- string validation
- pathlib file handling
- JSON loading and saving
- CSV parsing
- dictionaries and grouping
- unit tests with unittest
- bug fixes
- efficient membership checks
- avoiding repeated sorting
- logging redaction
- CLI boundary error handling
- retry/idempotency examples

Use only the Python standard library.
Keep each answer small and complete.
Include bad_code and reasoning for debugging/efficiency examples.
Do not include secrets or personal data.
```

## Prompt For TypeScript And JavaScript Batch

```text
You are creating original TypeScript and JavaScript training data for a tiny from-scratch coding language model.

Return JSONL only. No Markdown fences. No commentary.

Each line must be one valid JSON object with:
id, language, category, difficulty, instruction, constraints, answer, bad_code, reasoning, edge_cases, quality_notes.

Generate 40 examples:
- 24 TypeScript
- 16 JavaScript

Cover:
- Result<T> types
- unknown input parsing
- request handlers
- JSON responses
- fetch with timeout
- grouping and indexing arrays
- async/await mistakes
- retries
- redacting logs
- authentication boundary checks
- tests or test-shaped examples

Prefer built-in APIs.
Avoid framework-specific code unless the task says Node-style handler.
Keep each answer small and complete.
Include bad_code and reasoning for debugging/efficiency examples.
Do not include secrets or personal data.
```

## Prompt For PowerShell And SQL Batch

```text
You are creating original PowerShell and SQL training data for a tiny from-scratch coding language model.

Return JSONL only. No Markdown fences. No commentary.

Each line must be one valid JSON object with:
id, language, category, difficulty, instruction, constraints, answer, bad_code, reasoning, edge_cases, quality_notes.

Generate 30 examples:
- 15 PowerShell
- 15 SQL

PowerShell coverage:
- Test-Path with LiteralPath
- Get-ChildItem file summaries
- parameter validation
- JSON read/write
- clear errors
- no destructive commands

SQL coverage:
- table schemas
- primary keys
- foreign keys
- indexes for common filters
- simple joins
- aggregation
- avoiding N+1 style access by using joins

Keep all examples small.
Do not include destructive shell commands.
Do not include secrets or personal data.
```

## What I Will Do With The Data

When you bring JSONL back, I will:

1. Save it under `data/incoming/`.
2. Import it to protocol text.
3. Run the quality gate.
4. Inspect examples for correctness and duplicates.
5. Fix or reject weak examples.
6. Rebuild tokenizer and token data.
7. Run a small training/eval pass.
8. Only keep data that improves the model.
