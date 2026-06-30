# Phase 3.5B Work Packet 13H - Wikimedia Filtering and Neoma Self-Knowledge Seed

## Baseline

`59afb7c`

## Goal

Turn the acquired Wikimedia English dumps into local review candidates and create a factual Neoma self-knowledge seed from a versioned model card. This strengthens Stage A English understanding while preserving the rule that no candidate is training data until a later admission packet.

## Scope

- Parse Wikimedia XML as inert data.
- Strip wikitext markup without executing templates, links, Lua, JavaScript, or examples.
- Keep main-namespace pages only.
- Reject redirects, disambiguation pages, list/table/template-heavy pages, stubs, markup residue, secrets, and protected evaluation leakage.
- Segment coherent paragraphs into bounded English documents.
- Deduplicate candidates and cap source/family concentration.
- Preserve page ID, revision ID, timestamp, title, source, hashes, and review reasons.
- Create 50 review-only Neoma self-knowledge records from `neoma_model_card_v0_1_candidate`.

## Explicitly not done

- no training admission;
- no tokenizer comparison or rebuild;
- no dataset preparation;
- no model training.

Every output remains `training_allowed=false`.
