# Stage A Safe English Alternatives and Internal Seed Plan

## Decision

Keep GPT-NL blocked. Do not override Hugging Face's queued shard-security state and do not manually fetch the parquet files.

For Stage A v1, replace GPT-NL's planned 12 million approved English tokens with three bounded official Wikimedia text dumps:

| Source | Role | Approved-token target |
|---|---|---:|
| Simple English Wikipedia, 2026-06-01 | Clear general English | 5M |
| English Wikibooks, 2026-06-01 | Tutorial and technical English | 4M |
| English Wikiversity, 2026-06-01 | Educational reasoning and explanation | 3M |
| **Total** | | **12M** |

These are source pools, not automatically approved training data.

GPT-NL may be reconsidered only after all selected shards report `security_status=safe`, a new packet records exact hashes and row-level provenance, and an ablation shows that it adds useful diversity beyond the safer sources. It should not silently re-enter Stage A v1.

## Why these sources are more controllable

Each alternative is a single official Wikimedia XML snapshot with:

- a fixed 2026 snapshot identifier;
- a public dump job status;
- official SHA-1 and MD5 manifests;
- a local SHA-256 added by Neoma;
- stable page and revision identifiers;
- text-oriented XML and wikitext that can be parsed as inert data;
- a bounded archive rather than many remote parquet shards;
- one documented Wikimedia text-license family.

This does not make community content automatically trustworthy. It makes acquisition, provenance, rejection, and reproduction simpler and auditable.

## Acquisition boundary

Work Packet 13F does not download these dumps.

A later acquisition packet must, for each source:

1. request the exact `20260601` dump page;
2. confirm the `articlesmultistreamdump` job status is exactly `done`;
3. download only the pages-articles multistream archive and its index;
4. verify the official SHA-1 entry;
5. calculate and store a local SHA-256;
6. retain the dump legal page and hash its content;
7. write an immutable local acquisition manifest;
8. keep `training_allowed=false`.

Do not use a moving `latest` URL as the locked provenance. Do not silently fall back to another date. A missing snapshot or a non-`done` `articlesmultistreamdump` job is a review-blocking condition.

## Extraction and filtering

Parse XML and wikitext as data only. Never execute templates, Lua, JavaScript, shell fragments, links, or embedded examples.

Keep only main-namespace pages. Preserve source ID, project, snapshot, page ID, revision ID, revision timestamp, title, section path, license class, and content hashes.

Common rejects:

- redirects and disambiguation-only pages;
- navigation, category, user, talk, administration, and discussion pages;
- template-, list-, table-, infobox-, citation-, or metadata-dominated content;
- malformed markup, broken Unicode, vandalism indicators, and prompt-injection text;
- personal information, secret-shaped values, and copied credentials;
- empty outlines, answer keys without context, repeated boilerplate, and low-information stubs;
- obsolete technical procedures presented as current;
- near-duplicate paragraphs or sections across projects;
- any overlap with protected evaluations.

## Source-specific balance

### Simple English Wikipedia

Use it for clear grammar, reference resolution, definitions, causes, comparisons, and short explanations. Cap biography, geography, dates, entertainment, and repeated article templates. Do not let simplified encyclopedic prose dominate the final foundation corpus.

### English Wikibooks

Prefer complete, coherent sections from computing, mathematics, science, writing, and practical problem-solving books. Keep each book as a document family. Reject abandoned or obsolete books and repeated book navigation.

### English Wikiversity

Prefer lessons, definitions, worked explanations, and problem-solving discussions. Reject course administration, enrollment, talk, speculative notes, and quiz-answer-only pages. Keep each course or learning project in one split family.

## Deduplication and leakage

Before any admission:

1. exact bytes and exact normalized text;
2. markup-stripped and whitespace-normalized hashes;
3. paragraph shingles and MinHash-style near-duplicate candidates;
4. template-normalized duplicate detection;
5. cross-source comparison among all three alternatives;
6. comparison with the nine filtered repository/documentation sources;
7. comparison with all 331 instruction records;
8. comparison with every prompt, option, answer, explanation, and auxiliary field in every evaluation suite.

Any evaluation overlap is rejected, not merely reviewed. Instruction overlap is reviewed because ordinary programming phrases can be legitimate.

## Internal English seed

Do not attempt the full five-million-token internal corpus yet. First create a 60,000-token seed that proves the record schema, verification rules, family design, and writing quality.

| Component | Seed tokens | Target documents |
|---|---:|---:|
| Developer dialogue and follow-ups | 18K | 90 |
| Constraint-focused English | 12K | 120 |
| Bug, review, and commit language | 12K | 90 |
| CLI, configuration, and errors | 8K | 70 |
| Neoma self-knowledge | 6K | 50 |
| Verified source-linked transformations | 4K | 40 |
| **Total** | **60K** | **460** |

The seed is candidate material only. It must be authored in small batches, checked, reviewed, and later admitted by a separate manifest.

## Making Neoma understand itself

Self-knowledge is factual product documentation, not consciousness or hidden introspection. Generate it from a versioned model-card/build manifest so the model can accurately state:

- its model name and checkpoint lineage;
- architecture and context limits;
- supported programming-language focus;
- Stage A and Stage B roles;
- lack of internet, files, tools, or persistent memory unless explicitly provided;
- known weaknesses and uncertainty rules;
- that it may propose candidate data for a successor but cannot independently approve its own data.

Unknown model-card fields must fail generation rather than being invented. Self-knowledge must be regenerated whenever the released model changes.

## Anti-bloat rules

- Do not maximize token count before quality is demonstrated.
- No rename-only or slot-filled template families.
- No long essays when a short precise document teaches the same distinction.
- Cap each source, book, course, topic, and authored template family.
- Prefer cause/effect, constraints, comparisons, evidence, and decisions over trivia.
- Track marginal vocabulary and capability coverage per added family.
- Stop a component when new batches mostly duplicate existing lessons.

## Packet order after 13F

1. Audit and accept this planning packet.
2. Acquire and hash the three exact Wikimedia snapshots in quarantine.
3. Parse and filter a small review sample from each source.
4. Create the 60K internal seed in small audited batches.
5. Review source balance, prose quality, duplicates, and protected-data overlap.
6. Build a representative 500K candidate sample.
7. Admit that sample only in a separate packet.
8. Prepare tokenizer comparisons after approval.

No tokenizer, dataset, or model training belongs in Work Packet 13F.
