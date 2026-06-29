# Neoma English-and-Code Foundation Corpus Plan

## Status

This is a read-only planning document. It authorizes no download, corpus admission, tokenizer rebuild, dataset preparation, or training run.

The purpose of Stage A is to teach Neoma the local statistical structure of English and code before Stage B teaches request-to-answer behavior. Stage A uses ordinary next-token loss. Stage B remains instruction training with binary answer-only masking.

## Target behavior

Stage A should improve Neoma's ability to:

- understand short, naturally phrased developer requests;
- connect comments, names, errors, tests, and code behavior;
- follow negation, ordering, quantity, and boundary language;
- read concise technical prose and simple dialogue;
- produce grammatical short explanations;
- recognize common code and configuration structure.

It is not intended to create a general-purpose chatbot, encyclopedic knowledge model, or long-form storyteller.

## Proposed mixture

| Component | Target share | Purpose |
|---|---:|---|
| Clean code, tests, comments, and docstrings | 35% | Syntax, program structure, local semantics |
| Developer-oriented technical English | 25% | Documentation, explanations, API contracts |
| Simple general English | 15% | Grammar, reference, negation, sequence, common vocabulary |
| Bug reports, commit messages, and review comments | 10% | Developer intent, symptoms, change descriptions |
| Short developer dialogue and Q&A | 10% | Natural requests, clarification, follow-up language |
| CLI help, configuration, and error messages | 5% | Operational language and concise diagnostics |

These are starting ratios, not permanent quotas. The 250k- and 500k-token probes should measure whether a component improves held-out behavior before it is expanded.

## Token ladder

| Stage | Tokens | Purpose | Advancement rule |
|---|---:|---|---|
| A0 | 25k | Ingestion, loss, checkpoint, and generation sanity | Pipeline is deterministic and loss decreases |
| A1 | 250k | First English/code comprehension probe | Beats random baseline on local held-out checks |
| A2 | 500k | Mixture and tokenizer comparison | Improves both English and code checks without severe regression |
| A3 | 1M | First serious foundation checkpoint | Stable learning curves and useful Stage B initialization |
| Later | 5M+ | Only after evidence | Additional tokens produce measurable benefit per CPU hour |

The 25k run is not a capability target. It is an engineering sanity run.

## Document unit and segmentation

- Keep source documents intact when they fit the context target.
- Split long files at semantic boundaries: function, class, test case, section, paragraph, or dialogue turn.
- Preserve enough neighboring context for names, comments, and tests to make sense.
- Do not split in the middle of a string, code block, function signature, SQL statement, or sentence.
- Record the source document ID and segment index so related segments can stay in the same train/validation partition.
- Do not concatenate unrelated tiny fragments merely to fill 256 tokens.

Stage A may learn from overlapping windows inside longer documents because every token is supervised. Stage B records should still be designed so the request and answer fit together.

## Source admission rules

Every source must have:

- a stable source ID;
- origin and retrieval date;
- license or ownership status;
- allowed-use decision recorded before admission;
- content hash;
- language and component label;
- document-family or project identifier;
- automated scan results;
- human-review status for sampled documents.

Prefer:

- original project-created material;
- user-owned code and documentation;
- permissively licensed repositories with clear provenance;
- creator-published educational or technical corpora whose terms are verified at acquisition time;
- small complete modules and their tests;
- concise documentation that matches the code.

Do not assume that public availability grants training permission. License and terms must be checked when the source is acquired.

## Quality filters

Reject or quarantine:

- secrets, credentials, private keys, access tokens, or personal data;
- generated dependency trees, lockfile noise, minified bundles, vendored libraries, build output, or binary dumps;
- code that cannot be assigned a clear license or ownership basis;
- malformed encoding, truncated files, and mixed unrelated page fragments;
- spam, SEO text, scraped navigation, repeated boilerplate, or low-information lists;
- unsafe destructive instructions without educational context;
- unresolved merge markers and mechanically corrupted code;
- documents dominated by unsupported frameworks or obsolete APIs;
- copied held-out evaluation prompts, tests, fixtures, required terms, or expected answers.

## Deduplication

Run deduplication before splitting:

1. Exact byte and normalized-text hashes.
2. Near-duplicate text shingles.
3. Code-token similarity after comment and whitespace normalization.
4. Python AST structural comparison where practical.
5. Same project/file family grouping.
6. Repeated boilerplate and license-header suppression without deleting required attribution from provenance records.

Near-duplicate flags require review. Do not automatically delete two examples merely because they share a common API pattern.

## Evaluation leakage protection

The locked coding suite and future English-understanding suite must be excluded from:

- Stage A text;
- Stage B instructions;
- tokenizer training text;
- source-selection prompts;
- examples used to generate synthetic data;
- documentation samples and tests.

Compare against prompt text, required and forbidden terms, test code, behavior notes, function names, fixtures, edge cases, and expected messages. Split by source family before sampling validation data.

## English coverage

Stage A English should emphasize language that changes coding behavior:

- negation: do not mutate, do not sort, not missing;
- quantity and boundaries: at most, exactly, inclusive, first, every;
- conditionals: unless, only when, otherwise;
- reference: this value, the previous result, that file;
- ordering and sequence: before, after, then, preserve order;
- uncertainty and assumptions;
- concise cause-and-effect explanations;
- developer vocabulary used in errors, reviews, tests, and docs.

Avoid making story prose the dominant source. Simple general English supports grammar; developer language remains the center of the mixture.

## Code balance

Start with the project languages: Python, TypeScript, JavaScript, PowerShell, and PostgreSQL. Balance should be measured by tokens and structural variety, not file counts.

Include:

- functions and small modules;
- tests paired with implementations;
- comments and docstrings that accurately describe behavior;
- file, parsing, database, validation, debugging, and efficiency patterns;
- command help and configuration examples;
- short before/after changes when both versions are clearly labeled.

Avoid huge frameworks and generated application scaffolds in the first corpus.

## Tokenizer measurement plan

Build tokenizer candidates only after representative Stage A and frozen Stage B text exist. Compare 2k, 4k, and 8k vocabularies using the same train split.

Measure:

- tokens per character and per line by component and language;
- percentage of Stage B records fitting 192, 256, 384, and 512 tokens;
- fragmentation of identifiers, operators, indentation, protocol tags, and common English words;
- embedding-parameter cost;
- tokenizer training determinism and round-trip correctness;
- CPU tokens per second in the same 2.1M-model probe;
- downstream English and coding evaluation, not compression alone.

Do not select a tokenizer only because it creates fewer tokens. A larger vocabulary consumes model parameters and may reduce learning in a tiny model.

## Training and validation split

- Split by repository, document family, conversation, or source document before segmentation.
- Keep related code, tests, docs, and variants in one split.
- Reserve a fixed Stage A validation set before training.
- Record exact hashes and manifests for every experiment.
- Never use the locked instruction evaluation suite as the Stage A validation set.

## English-understanding evaluation required before A1

Create a separate locked suite that measures:

- paraphrase understanding;
- negation and valid falsy values;
- singular/plural and quantity constraints;
- order and boundary language;
- pronoun and short-reference resolution;
- two-constraint instructions;
- concise summaries and explanations;
- clarification when information is materially missing;
- one- and two-turn developer follow-ups.

The suite should be small, machine-checkable where possible, and excluded from both tokenizer and model training.

## Stage transition

`--resume` continues the same optimizer, scheduler, and training run.

The future `--init-from-model` operation starts a new stage by loading model weights only, creating a fresh optimizer and scheduler, and recording the parent checkpoint. Stage B must not masquerade as a resumed Stage A run.

Binary answer-only masking should remain the Stage B default. Weighted prompt or reasoning losses should be tested only after a specific evaluation failure justifies the added variable.

## Stop criteria

Do not expand Stage A merely because more text is available. Advance only when:

- validation loss and held-out behavior improve;
- English gains do not erase code capability;
- additional tokens provide useful gain per CPU hour;
- source quality and provenance remain auditable;
- the 2.1M model has not clearly saturated from capacity rather than data.

## What not to include yet

- a multi-million-token unreviewed web scrape;
- broad trivia, news, social-media chatter, or personality data;
- long multi-turn conversations that exceed the target context;
- chain-of-thought or hidden reasoning corpora;
- generated synthetic data without source and quality controls;
- dependencies, compiled assets, minified code, or repository dumps;
- held-out evaluation material;
- RL, preference optimization, LoRA, or external pretrained checkpoints.

## Immediate next artifact after approval

Produce a source manifest template, the locked English-understanding evaluation suite, and a reviewed 25k-token A0 corpus. Do not jump directly to 1M tokens.
