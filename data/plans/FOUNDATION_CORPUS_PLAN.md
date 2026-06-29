# Phase 3.5B Stage A Foundation Corpus Plan

## Purpose

Teach Neoma code syntax, local structure, common library patterns, tests, comments, and technical language before instruction following. Stage A uses ordinary next-token loss and starts from random weights.

## Token ladder

| Step | Clean tokens | Purpose |
|---|---:|---|
| A0 | 25k–50k | pipeline and loss sanity only |
| A1 | 250k | first behavior probe |
| A2 | 500k | coverage and tokenizer comparison |
| A3 | about 1M | first serious foundation checkpoint |
| Later | up to 2M | only when evaluation shows useful continued gains |

Do not create a 2M-token corpus before the 250k and 500k probes demonstrate that quality and mixture are sound.

## Suggested 1M-token composition

| Material | Share |
|---|---:|
| Small source files and functions | 45% |
| Tests and test fixtures | 20% |
| Concise module/API documentation and comments | 10% |
| PowerShell scripts and command examples | 8% |
| SQL schemas, migrations, and queries | 10% |
| Configuration/serialization examples | 7% |

Approximate language shares: Python 35%, TypeScript 20%, JavaScript 17%, PowerShell 10%, SQL 10%, technical text/config 8%. These are starting targets, not immutable quotas.

## Source admission

Every file needs a manifest entry containing source type, owner/license, original path or origin, language, hash, review status, and exclusion reason when rejected. Prefer original project-written material, user-owned code, or clearly permissive/public-domain sources.

Exclude:

- generated dependencies, lockfile noise, minified bundles, build output, vendored libraries;
- secrets, credentials, personal data, private endpoints, machine-specific paths;
- incomplete snippets that cannot be understood locally;
- duplicated templates and generated boilerplate;
- large framework internals;
- eval prompts, solutions, and close paraphrases.

## Document framing

Use stable file-boundary markers and language/path metadata, but do not add verbose boilerplate around every tiny snippet. Preserve meaningful comments and tests. Normalize line endings without reformatting code in ways that could introduce errors.

## Split policy

Split by project/file family before tokenization. Keep related files from one small project in the same split to reduce leakage. Use a held-out foundation validation set distinct from the 80 behavioral eval prompts.

## Stage transition

The future `--init-from-model` operation must load model weights only, validate architecture/tokenizer compatibility, initialize a fresh optimizer and schedule, reset stage step counters, and record the parent checkpoint hash. It is not resume training.
