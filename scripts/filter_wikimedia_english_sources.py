#!/usr/bin/env python3
"""Parse and filter quarantined Wikimedia English dumps into review candidates.

The command treats XML and wikitext as inert data. It produces local-only
candidate JSONL files under ``data/foundation/filtered/wikimedia_english_20260601``
and a small tracked review summary with counts and hashes. No output is admitted
to training.
"""

from __future__ import annotations

import argparse
import bz2
from collections import Counter, defaultdict
import html
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Iterator

from stage_a_filtering_common import (
    HIGH_CONFIDENCE_SECRET_PATTERNS,
    PROMPT_INJECTION_PATTERNS,
    build_protected_index,
    leakage_findings,
    lexical_tokens,
    load_protected_items,
    normalize_text,
    record_fingerprints,
    shingle_hashes,
)
from stage_a_staging_common import StagingError, atomic_write_json, atomic_write_jsonl, sha256_file, utc_now

BASELINE = "59afb7c"
EXPECTED_SOURCE_IDS = ("simplewiki_20260601", "enwikibooks_20260601", "enwikiversity_20260601")
DEFAULT_FILTER_ROOT = Path("data/foundation/filtered/wikimedia_english_20260601")
DEFAULT_REVIEW_OUT = Path("data/reviews/stage_a_wikimedia_filtering_results_v1.json")

NAMESPACE_RE = re.compile(r"^\{.*\}")
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
REF_RE = re.compile(r"<ref\b[^>/]*/>|<ref\b[^>]*>.*?</ref>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
EXTERNAL_LINK_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+(?:\s+([^\]]+))?\]")
INTERNAL_LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
HEADING_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$")
TABLE_LINE_RE = re.compile(r"^\s*(?:\{\||\|\}|\|-|!|\|)")
LIST_MARKER_RE = re.compile(r"^\s*[*#:;]+\s*")
SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
SENTENCE_END_RE = re.compile(r"[.!?]['\")\]]?$")
DISAMBIG_RE = re.compile(r"(?i)\{\{\s*(?:disambiguation|disambig|dab|geodis)\b")
REDIRECT_RE = re.compile(r"(?i)^\s*#redirect\b")
STUB_RE = re.compile(r"(?i)\b(stub|short description|authority control)\b")
BAD_TITLE_RE = re.compile(r"(?i)(?:^|/)(?:list of|index of|glossary of|outline of)\b")
WEAK_TITLE_RE = re.compile(r"^(?:[A-Za-z]|\d{1,4}|[IVXLCDM]+)$")
CODE_MARKER_RE = re.compile(
    r"(?i)(?:#include|</?\w+>|[{};]{2,}|\b(?:function|return|var|let|const|def|class|public|private|namespace|printf|scanf)\b)"
)
TECHNICAL_CONTEXT_RE = re.compile(
    r"(?i)\b(?:computer|program|algorithm|logic|mathematics|science|engineering|data|system|software|test|function|method|language)\b"
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StagingError(f"expected JSON object: {path}")
    return value


def tag_name(tag: str) -> str:
    return NAMESPACE_RE.sub("", tag)


def child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if tag_name(child.tag) == name:
            return child.text or ""
    return ""


def first_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if tag_name(child.tag) == name:
            return child
    return None


def page_record(element: ET.Element) -> dict[str, Any]:
    revision = first_child(element, "revision")
    return {
        "title": child_text(element, "title"),
        "ns": child_text(element, "ns"),
        "page_id": child_text(element, "id"),
        "revision_id": child_text(revision, "id") if revision is not None else "",
        "revision_timestamp": child_text(revision, "timestamp") if revision is not None else "",
        "text": child_text(revision, "text") if revision is not None else "",
    }


def iter_pages(archive_path: Path) -> Iterator[dict[str, Any]]:
    with bz2.open(archive_path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, element in context:
            if tag_name(element.tag) == "page":
                yield page_record(element)
                element.clear()


def strip_balanced(text: str, opener: str, closer: str, max_passes: int = 8) -> str:
    for _ in range(max_passes):
        start = text.find(opener)
        if start < 0:
            return text
        result: list[str] = []
        index = 0
        changed = False
        while index < len(text):
            if text.startswith(opener, index):
                depth = 1
                index += len(opener)
                while index < len(text) and depth:
                    if text.startswith(opener, index):
                        depth += 1
                        index += len(opener)
                    elif text.startswith(closer, index):
                        depth -= 1
                        index += len(closer)
                    else:
                        index += 1
                result.append(" ")
                changed = True
            else:
                result.append(text[index])
                index += 1
        text = "".join(result)
        if not changed:
            return text
    return text


def convert_internal_link(match: re.Match[str]) -> str:
    payload = match.group(1)
    lowered = payload.lower()
    if lowered.startswith(("file:", "image:", "category:", "template:", "help:", "wikipedia:")):
        return " "
    parts = payload.split("|")
    label = parts[-1] if len(parts) > 1 else parts[0]
    label = label.split("#", 1)[0] if len(parts) == 1 else label
    return label.replace("_", " ")


def clean_wikitext(raw: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not raw.strip():
        return "", ["empty_page_text"]
    if REDIRECT_RE.search(raw):
        return "", ["redirect_page"]
    if DISAMBIG_RE.search(raw):
        reasons.append("disambiguation_template")
    template_count = raw.count("{{")
    table_count = raw.count("{|")
    text = COMMENT_RE.sub(" ", raw)
    text = REF_RE.sub(" ", text)
    text = strip_balanced(text, "{|", "|}")
    text = strip_balanced(text, "{{", "}}")
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]+\]\]", " ", text, flags=re.I)
    text = INTERNAL_LINK_RE.sub(convert_internal_link, text)
    text = EXTERNAL_LINK_RE.sub(lambda match: match.group(1) or " ", text)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("'''", "").replace("''", "")
    text = text.replace("__TOC__", " ").replace("__NOTOC__", " ")
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    lines: list[str] = []
    list_lines = 0
    table_lines = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if TABLE_LINE_RE.match(line):
            table_lines += 1
            continue
        if line.startswith(("{{", "}}", "|", "!")):
            table_lines += 1
            continue
        if LIST_MARKER_RE.match(line):
            list_lines += 1
            line = LIST_MARKER_RE.sub("", line).strip()
        heading = HEADING_RE.match(line)
        if heading:
            line = heading.group(1).strip()
        line = re.sub(r"\s+\([,;:]\)", " ", line)
        line = SPACE_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    cleaned = BLANK_RE.sub("\n\n", "\n".join(lines)).strip()
    total_lines = max(1, len([line for line in lines if line.strip()]))
    if list_lines / total_lines > 0.45:
        reasons.append("list_dominated")
    if table_lines / total_lines > 0.25 or table_count >= 4:
        reasons.append("table_dominated")
    if template_count >= 80:
        reasons.append("template_heavy_source")
    return cleaned, reasons


def paragraphs_from_cleaned(text: str) -> list[str]:
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = SPACE_RE.sub(" ", " ".join(line.strip() for line in block.splitlines() if line.strip())).strip()
        tokens = lexical_tokens(block)
        alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
        if len(alpha_tokens) < 12:
            continue
        if len(block) < 80:
            continue
        paragraphs.append(block)
    return paragraphs


def make_segments(paragraphs: list[str], min_tokens: int, max_tokens: int, max_segments: int) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        count = len(lexical_tokens(paragraph))
        if count > max_tokens:
            continue
        if current and current_tokens + count > max_tokens:
            if current_tokens >= min_tokens:
                segments.append("\n\n".join(current))
                if len(segments) >= max_segments:
                    return segments
            current = []
            current_tokens = 0
        current.append(paragraph)
        current_tokens += count
    if current and current_tokens >= min_tokens and len(segments) < max_segments:
        segments.append("\n\n".join(current))
    return segments


def source_policy(source_id: str) -> dict[str, Any]:
    if source_id == "simplewiki_20260601":
        return {"component": "simple_general_english", "min_tokens": 90, "max_tokens": 420, "max_segments_per_page": 2, "candidate_token_budget": 5_000_000}
    if source_id == "enwikibooks_20260601":
        return {"component": "tutorial_and_technical_english", "min_tokens": 110, "max_tokens": 520, "max_segments_per_page": 4, "candidate_token_budget": 4_000_000}
    if source_id == "enwikiversity_20260601":
        return {"component": "educational_reasoning_and_explanation", "min_tokens": 110, "max_tokens": 520, "max_segments_per_page": 4, "candidate_token_budget": 3_000_000}
    raise StagingError(f"unknown source policy: {source_id}")


def title_family(source_id: str, title: str) -> str:
    root = title.split("/", 1)[0].strip().lower()
    root = re.sub(r"[^a-z0-9]+", "-", root).strip("-") or "root"
    return f"{source_id}:page-family:{root}"


def quality_score(title: str, text: str, source_id: str) -> tuple[int, list[str]]:
    tokens = lexical_tokens(text)
    token_count = len(tokens)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    labels: list[str] = []
    score = 0
    if 140 <= token_count <= 420:
        score += 20
        labels.append("compact_segment")
    if len(paragraphs) >= 2:
        score += 15
        labels.append("multi_paragraph")
    sentence_like = sum(1 for p in paragraphs for s in re.split(r"(?<=[.!?])\s+", p) if SENTENCE_END_RE.search(s.strip()))
    if sentence_like >= 3:
        score += 15
        labels.append("sentence_rich")
    lower = text.lower()
    for marker in ("because", "therefore", "however", "if ", "when ", "before ", "after ", "unless ", "example"):
        if marker in lower:
            score += 4
    if source_id != "simplewiki_20260601" and any(term in lower for term in ("function", "data", "program", "method", "test", "system", "logic")):
        score += 12
        labels.append("technical_language")
    if BAD_TITLE_RE.search(title):
        score -= 30
        labels.append("list_or_outline_title")
    number_ratio = sum(token.isdigit() for token in tokens) / max(1, token_count)
    if number_ratio > 0.12:
        score -= 15
        labels.append("number_heavy")
    return score, labels


def rejection_for_page(title: str, raw: str, cleaned: str, cleanup_reasons: list[str], paragraphs: list[str]) -> list[str]:
    reasons = list(cleanup_reasons)
    if BAD_TITLE_RE.search(title):
        reasons.append("list_outline_or_glossary_title")
    if WEAK_TITLE_RE.fullmatch(title.strip()):
        reasons.append("weak_single_token_title")
    if STUB_RE.search(raw) and len(lexical_tokens(cleaned)) < 180:
        reasons.append("stub_or_metadata_dominated")
    if len(paragraphs) < 2:
        reasons.append("too_few_coherent_paragraphs")
    if len(lexical_tokens(cleaned)) < 80:
        reasons.append("too_short_after_cleaning")
    if sum(cleaned.count(char) for char in "{}[]|") / max(1, len(cleaned)) > 0.015:
        reasons.append("markup_residue")
    for label, pattern in HIGH_CONFIDENCE_SECRET_PATTERNS:
        if pattern.search(cleaned):
            reasons.append(f"possible_secret:{label}")
    return sorted(set(reasons))


def segment_rejection_reasons(title: str, text: str, source_id: str) -> list[str]:
    reasons: list[str] = []
    stripped = text.strip()
    tokens = lexical_tokens(stripped)
    if not stripped:
        return ["empty_segment"]
    first = stripped[0]
    if first.isalpha() and first.islower():
        reasons.append("starts_with_lowercase_fragment")
    if stripped.count("$") >= 3:
        reasons.append("currency_or_budget_fragment")
    code_markers = len(CODE_MARKER_RE.findall(stripped))
    if code_markers / max(1, len(tokens)) > 0.035:
        reasons.append("code_fragment_dominated")
    bracket_residue = sum(stripped.count(char) for char in "{}[]|")
    if bracket_residue / max(1, len(stripped)) > 0.01:
        reasons.append("markup_or_code_residue")
    digit_tokens = sum(token.isdigit() for token in tokens)
    if digit_tokens / max(1, len(tokens)) > 0.14:
        reasons.append("number_table_or_date_heavy")
    lower = stripped.lower()
    if source_id == "simplewiki_20260601" and " was born " in lower and " died " in lower and not TECHNICAL_CONTEXT_RE.search(stripped):
        reasons.append("biography_trivia_without_technical_context")
    if WEAK_TITLE_RE.fullmatch(title.strip()):
        reasons.append("weak_single_token_title")
    return sorted(set(reasons))


def build_candidate(
    source: dict[str, Any],
    page: dict[str, Any],
    text: str,
    segment_index: int,
    score: int,
    labels: list[str],
    protected: Any,
) -> dict[str, Any]:
    source_id = source["source_id"]
    record_id = f"{source_id}:page:{page['page_id']}:rev:{page['revision_id']}:seg:{segment_index}"
    leakage = leakage_findings(text, protected)
    critical = [row for row in leakage if row["severity"] == "critical"]
    review = [row for row in leakage if row["severity"] == "review"]
    review_reasons = ["protected_overlap_review"] if review else []
    rejection_reasons = ["protected_evaluation_leakage"] if critical else []
    if any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS):
        review_reasons.append("prompt_injection_language")
    fingerprints = record_fingerprints(text)
    return {
        "schema_version": "1.0",
        "record_id": record_id,
        "source_id": source_id,
        "source_name": source["name"],
        "component": source_policy(source_id)["component"],
        "snapshot": source["snapshot"],
        "title": page["title"],
        "page_id": page["page_id"],
        "revision_id": page["revision_id"],
        "revision_timestamp": page["revision_timestamp"],
        "segment_index": segment_index,
        "family_id": title_family(source_id, page["title"]),
        "text": text,
        "language_hint": "english",
        "quality_score": score,
        "quality_labels": sorted(set(labels)),
        "leakage_findings": leakage,
        "review_reasons": sorted(set(review_reasons)),
        "rejection_reasons": sorted(set(rejection_reasons)),
        "status": "rejected_not_admitted" if rejection_reasons else ("human_review_required_not_admitted" if review_reasons else "filtered_candidate_not_admitted"),
        "training_allowed": False,
        **fingerprints,
    }


def compact_counts(counter: Counter[str], limit: int = 40) -> dict[str, int]:
    if len(counter) <= limit:
        return dict(sorted(counter.items()))
    most_common = counter.most_common(limit)
    shown = {key for key, _ in most_common}
    other = sum(value for key, value in counter.items() if key not in shown)
    result = dict(most_common)
    result["other_reasons_compacted"] = other
    return result


def load_acquisition_manifest(manifest_root: Path, source_id: str) -> dict[str, Any]:
    path = manifest_root / f"{source_id}.wikimedia_acquisition.json"
    if not path.is_file():
        raise StagingError(f"missing acquisition manifest: {source_id}")
    manifest = load_json(path)
    if manifest.get("training_allowed") is not False:
        raise StagingError(f"{source_id}: acquisition manifest grants training permission")
    if manifest.get("status") != "acquired_quarantined_pending_review":
        raise StagingError(f"{source_id}: not acquired cleanly")
    if manifest.get("dumpstatus", {}).get("articles_multistream_status") != "done":
        raise StagingError(f"{source_id}: dump status is not done")
    return manifest


def deduplicate(records: list[dict[str, Any]]) -> None:
    seen_normalized: dict[str, str] = {}
    seen_shingles: dict[int, str] = {}
    for record in sorted(records, key=lambda item: (-int(item["quality_score"]), item["record_id"])):
        if record["status"] == "rejected_not_admitted":
            continue
        prior = seen_normalized.get(record["normalized_sha256"])
        if prior:
            record["status"] = "rejected_not_admitted"
            record.setdefault("rejection_reasons", []).append(f"exact_duplicate_of:{prior}")
            continue
        shingles = shingle_hashes(record["text"], width=7, max_shingles=2000)
        overlap_hits = Counter(seen_shingles.get(shingle) for shingle in shingles)
        overlap_hits.pop(None, None)
        if overlap_hits:
            duplicate_id, count = overlap_hits.most_common(1)[0]
            if shingles and count / len(shingles) >= 0.86:
                record["status"] = "rejected_not_admitted"
                record.setdefault("rejection_reasons", []).append(f"near_duplicate_of:{duplicate_id}")
                continue
        seen_normalized[record["normalized_sha256"]] = record["record_id"]
        for shingle in shingles:
            seen_shingles.setdefault(shingle, record["record_id"])


def select_candidates(records: list[dict[str, Any]], token_budget: int) -> None:
    selected_tokens = 0
    family_counts: Counter[str] = Counter()
    ordered = sorted(records, key=lambda item: (-int(item["quality_score"]), item["record_id"]))
    for record in ordered:
        if record["status"] == "rejected_not_admitted":
            continue
        token_count = int(record["token_count_proxy"])
        family = record["family_id"]
        if selected_tokens + token_count > token_budget:
            record["status"] = "rejected_not_admitted"
            record.setdefault("rejection_reasons", []).append("candidate_token_budget_overflow")
            continue
        family_limit = 160 if record["source_id"] != "simplewiki_20260601" else 80
        if family_counts[family] >= family_limit:
            record["status"] = "human_review_required_not_admitted"
            record.setdefault("review_reasons", []).append("family_concentration_limit")
        selected_tokens += token_count
        family_counts[family] += 1


def atomic_replace_directory(temp: Path, target: Path, force: bool) -> None:
    if target.exists():
        if not force:
            raise StagingError(f"output exists; rerun with --force after review: {target}")
        backup = target.with_name(target.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(target, backup)
        try:
            os.replace(temp, target)
        except Exception:
            os.replace(backup, target)
            raise
        shutil.rmtree(backup)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, target)


def process_source(
    source: dict[str, Any],
    manifest_root: Path,
    protected: Any,
    page_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_id = source["source_id"]
    manifest = load_acquisition_manifest(manifest_root, source_id)
    archive = manifest["artifacts"]["archive"]
    archive_path = Path(archive["path"])
    if not archive_path.is_file() or sha256_file(archive_path) != archive["sha256"]:
        raise StagingError(f"{source_id}: archive missing or SHA-256 mismatch")
    policy = source_policy(source_id)
    records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    pages_seen = 0
    pages_main = 0
    for page in iter_pages(archive_path):
        pages_seen += 1
        if page_limit is not None and pages_seen > page_limit:
            break
        if page["ns"] != "0":
            counters["non_main_namespace"] += 1
            continue
        pages_main += 1
        cleaned, cleanup_reasons = clean_wikitext(page["text"])
        paragraphs = paragraphs_from_cleaned(cleaned)
        page_reasons = rejection_for_page(page["title"], page["text"], cleaned, cleanup_reasons, paragraphs)
        if page_reasons:
            counters.update(page_reasons)
            continue
        segments = make_segments(
            paragraphs,
            min_tokens=int(policy["min_tokens"]),
            max_tokens=int(policy["max_tokens"]),
            max_segments=int(policy["max_segments_per_page"]),
        )
        if not segments:
            counters["no_usable_segment"] += 1
            continue
        for index, segment in enumerate(segments, 1):
            segment_reasons = segment_rejection_reasons(page["title"], segment, source_id)
            if segment_reasons:
                counters.update(segment_reasons)
                continue
            score, labels = quality_score(page["title"], segment, source_id)
            record = build_candidate(source, page, segment, index, score, labels, protected)
            records.append(record)
    deduplicate(records)
    select_candidates(records, int(policy["candidate_token_budget"]))
    status_counts = Counter(record["status"] for record in records)
    reason_counts = Counter(reason for record in records for reason in record.get("rejection_reasons", []))
    review_counts = Counter(reason for record in records for reason in record.get("review_reasons", []))
    manifest_summary = {
        "schema_version": "1.0",
        "baseline": BASELINE,
        "source_id": source_id,
        "source_name": source["name"],
        "snapshot": source["snapshot"],
        "status": "filtered_candidates_not_admitted",
        "pages_seen": pages_seen,
        "main_namespace_pages": pages_main,
        "record_count": len(records),
        "candidate_count": status_counts["filtered_candidate_not_admitted"],
        "review_count": status_counts["human_review_required_not_admitted"],
        "rejected_count": status_counts["rejected_not_admitted"],
        "candidate_token_count_proxy": sum(int(r["token_count_proxy"]) for r in records if r["status"] != "rejected_not_admitted"),
        "page_rejection_reason_counts": compact_counts(counters),
        "record_rejection_reason_counts": compact_counts(reason_counts),
        "record_review_reason_counts": compact_counts(review_counts),
        "training_allowed": False,
    }
    return records, manifest_summary


def write_source_output(root: Path, source_id: str, records: list[dict[str, Any]], summary: dict[str, Any], rejection_sample_limit: int) -> dict[str, Any]:
    source_dir = root / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    candidates = [record for record in records if record["status"] == "filtered_candidate_not_admitted"]
    reviews = [record for record in records if record["status"] == "human_review_required_not_admitted"]
    rejections = [record for record in records if record["status"] == "rejected_not_admitted"][:rejection_sample_limit]
    families: dict[str, list[str]] = defaultdict(list)
    for record in candidates + reviews:
        families[record["family_id"]].append(record["record_id"])
    family_rows = [
        {
            "schema_version": "1.0",
            "source_id": source_id,
            "family_id": family_id,
            "member_count": len(member_ids),
            "member_ids": sorted(member_ids),
            "status": "family_candidate_not_admitted",
            "training_allowed": False,
        }
        for family_id, member_ids in sorted(families.items())
    ]
    atomic_write_jsonl(source_dir / "candidates.jsonl", candidates)
    atomic_write_jsonl(source_dir / "review_queue.jsonl", reviews)
    atomic_write_jsonl(source_dir / "rejections_sample.jsonl", rejections)
    atomic_write_jsonl(source_dir / "families.jsonl", family_rows)
    summary = dict(summary)
    summary["family_count"] = len(family_rows)
    summary["rejection_sample_count"] = len(rejections)
    atomic_write_json(source_dir / "filtering_manifest.json", summary)
    return summary


def review_summary(source_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for row in source_summaries:
        for key in ("pages_seen", "main_namespace_pages", "record_count", "candidate_count", "review_count", "rejected_count", "candidate_token_count_proxy", "family_count"):
            totals[key] += int(row.get(key, 0))
    return {
        "schema_version": "1.0",
        "review_id": "stage_a_wikimedia_filtering_results_v1",
        "baseline": BASELINE,
        "generated_utc": utc_now(),
        "scope": "Wikimedia English XML parsing, wikitext cleanup, quality filtering, deduplication, and leakage review",
        "status": "filtered_candidates_not_admitted",
        "raw_artifacts_committed": False,
        "filtered_text_committed": False,
        "training_allowed": False,
        "totals": dict(totals),
        "sources": source_summaries,
        "review_decision": {
            "status": "filtering_complete_quarantined_pending_human_review",
            "reason": "Wikimedia pages were parsed as inert data, noisy pages were rejected, candidates retain provenance and hashes, and no output is admitted to training.",
        },
        "next_step": "Review source samples and combine approved Wikimedia candidates with the internal English seed before tokenizer comparison.",
    }


def run_filtering(
    repo_root: Path,
    plan_path: Path,
    manifest_root: Path,
    filter_root: Path,
    review_out: Path,
    selected_ids: list[str],
    execute: bool,
    force: bool,
    page_limit: int | None,
    rejection_sample_limit: int,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    sources = [source for source in plan["sources"] if source["source_id"] in selected_ids]
    if [source["source_id"] for source in sources] != selected_ids:
        raise StagingError("selected source order mismatch")
    protected = build_protected_index(load_protected_items(repo_root))
    if not execute:
        return {
            "schema_version": "1.0",
            "mode": "dry_run",
            "selected_source_ids": selected_ids,
            "protected_item_count": len(protected.items),
            "training_allowed": False,
        }
    temp = Path(tempfile.mkdtemp(prefix=filter_root.name + ".tmp-", dir=filter_root.parent))
    try:
        summaries: list[dict[str, Any]] = []
        for source in sources:
            records, summary = process_source(source, manifest_root, protected, page_limit)
            written_summary = write_source_output(temp, source["source_id"], records, summary, rejection_sample_limit)
            summaries.append(written_summary)
        summary = review_summary(summaries)
        atomic_write_json(temp / "stage_a_wikimedia_filtering_summary.json", summary)
        atomic_replace_directory(temp, filter_root, force)
        atomic_write_json(review_out, summary)
        return summary
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, default=Path("data/foundation/manifests/stage_a_safe_english_alternatives_v1_candidate.json"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/foundation/sources/manifests"))
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--review-out", type=Path, default=DEFAULT_REVIEW_OUT)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--page-limit", type=int)
    parser.add_argument("--rejection-sample-limit", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all and args.source:
        raise StagingError("use either --all or --source, not both")
    selected = list(EXPECTED_SOURCE_IDS) if args.all else list(args.source)
    if not selected:
        print("Select --all or --source SOURCE_ID")
        return 0
    unknown = sorted(set(selected) - set(EXPECTED_SOURCE_IDS))
    if unknown:
        raise StagingError(f"unknown source IDs: {unknown}")
    report = run_filtering(
        repo_root=args.repo_root.resolve(),
        plan_path=args.plan,
        manifest_root=args.manifest_root,
        filter_root=args.filter_root,
        review_out=args.review_out,
        selected_ids=selected,
        execute=args.execute,
        force=args.force,
        page_limit=args.page_limit,
        rejection_sample_limit=args.rejection_sample_limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True) + "\n", end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StagingError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
