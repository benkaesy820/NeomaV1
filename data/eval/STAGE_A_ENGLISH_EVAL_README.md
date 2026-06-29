# Stage A English-and-Code Evaluation v1

## Purpose

These suites measure whether Neoma understands short developer English, constraints, references, follow-ups, and engineering uncertainty before and after Stage A foundation training.

- `stage_a_english_dev_v1.jsonl`: 48 development prompts. It may be evaluated frequently.
- `stage_a_english_locked_v1.jsonl`: 48 held-out prompts. Use only at named checkpoints.

Every record has `training_allowed=false`. Neither file, its answer keys, rationales, wording, nor derived paraphrases may enter tokenizer training, Stage A text, Stage B data, synthetic-data prompts, examples, or documentation used as training material.

## Scoring

Version 1 uses short multiple-choice responses so a tiny model can be scored deterministically without a judge model. Normalize the generated answer, accept the first standalone choice letter, and compare it with `accepted_answers`.

The suites test eight balanced capability groups:

1. negation and valid falsy values;
2. quantity and boundaries;
3. order and sequence;
4. reference resolution;
5. developer language;
6. clarification judgment;
7. follow-up state preservation;
8. uncertainty and operational safety.

A high choice score is necessary but not sufficient. Generation quality, code behavior, and the existing coding evaluations remain separate requirements.
