from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REQUIRED_EVAL_FIELDS = {
    "id": str,
    "language": str,
    "category": str,
    "difficulty": str,
    "prompt": str,
    "scoring": str,
    "accepted_answers": list,
    "rationale": str,
    "suite": str,
    "split": str,
    "training_allowed": bool,
    "source_family": str,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def trigrams(text: str) -> set[tuple[str, str, str]]:
    tokens = normalize(text).split()
    return set(zip(tokens, tokens[1:], tokens[2:])) if len(tokens) >= 3 else set()


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate_eval(path: Path, expected_suite: str, expected_split: str) -> tuple[list[dict], list[str]]:
    rows = load_jsonl(path)
    errors: list[str] = []
    ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        for field, kind in REQUIRED_EVAL_FIELDS.items():
            if field not in row or not isinstance(row[field], kind):
                errors.append(f"{path}:{index}: missing or invalid {field}")
        row_id = row.get("id")
        if isinstance(row_id, str):
            if row_id in ids:
                errors.append(f"{path}:{index}: duplicate id {row_id}")
            ids.add(row_id)
        if row.get("suite") != expected_suite:
            errors.append(f"{path}:{index}: wrong suite")
        if row.get("split") != expected_split:
            errors.append(f"{path}:{index}: wrong split")
        if row.get("training_allowed") is not False:
            errors.append(f"{path}:{index}: training_allowed must be false")
        answers = row.get("accepted_answers")
        if not answers or not all(isinstance(value, str) and value.strip() for value in answers):
            errors.append(f"{path}:{index}: accepted_answers must be non-empty strings")
        if row.get("scoring") == "choice" and any(value not in {"A", "B", "C", "D"} for value in answers or []):
            errors.append(f"{path}:{index}: invalid choice answer")
    return rows, errors


def existing_texts(repo: Path, exclude: set[Path]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for path in sorted((repo / "data" / "raw").glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"<instruction>\s*(.*?)\s*</instruction>", text, re.S):
            items.append((str(path.relative_to(repo)), match.group(1)))
    for path in sorted((repo / "data" / "eval").glob("*.jsonl")):
        if path.resolve() in exclude:
            continue
        for row in load_jsonl(path):
            prompt = row.get("prompt")
            if isinstance(prompt, str):
                items.append((str(path.relative_to(repo)), prompt))
    return items


def validate_manifest(path: Path) -> tuple[dict, list[str]]:
    obj = load_json(path)
    errors: list[str] = []
    required = ["schema_version", "manifest_id", "baseline", "status", "freshness_policy", "external_target_tokens", "internal_target_tokens", "total_target_tokens", "training_allowed", "sources"]
    for field in required:
        if field not in obj:
            errors.append(f"{path}: missing {field}")
    if obj.get("training_allowed") is not False:
        errors.append(f"{path}: training_allowed must be false")
    sources = obj.get("sources") or []
    ids: set[str] = set()
    target = 0
    for index, source in enumerate(sources, 1):
        for field in ["source_id", "name", "upstream", "version_policy", "freshness_status", "license", "target_tokens", "training_allowed", "status"]:
            if field not in source:
                errors.append(f"{path}: source {index} missing {field}")
        source_id = source.get("source_id")
        if source_id in ids:
            errors.append(f"{path}: duplicate source_id {source_id}")
        ids.add(source_id)
        if source.get("training_allowed") is not False:
            errors.append(f"{path}: {source_id} training_allowed must be false before admission")
        if source.get("release_channel") not in {"stable", "stable_lts", "active_lts", "published corpus", "rolling documentation"}:
            errors.append(f"{path}: {source_id} has unapproved release channel")
        freshness = str(source.get("freshness_status", ""))
        if not (freshness.startswith("2026") or freshness == "2025_latest_stable"):
            errors.append(f"{path}: {source_id} violates freshness policy")
        target += int(source.get("target_tokens", 0))
    if target != obj.get("external_target_tokens"):
        errors.append(f"{path}: source targets total {target}, expected {obj.get('external_target_tokens')}")
    if obj.get("external_target_tokens", 0) + obj.get("internal_target_tokens", 0) != obj.get("total_target_tokens"):
        errors.append(f"{path}: external + internal does not equal total")
    return obj, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("data/reviews/stage_a_work_packet_10_validation.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    dev_path = repo / "data/eval/stage_a_english_dev_v1.jsonl"
    locked_path = repo / "data/eval/stage_a_english_locked_v1.jsonl"
    manifest_path = repo / "data/foundation/manifests/stage_a_sources_v1_candidate.json"

    dev, errors = validate_eval(dev_path, "stage_a_english_dev_v1", "development")
    locked, more = validate_eval(locked_path, "stage_a_english_locked_v1", "heldout")
    errors.extend(more)
    manifest, more = validate_manifest(manifest_path)
    errors.extend(more)

    all_rows = dev + locked
    ids = [row["id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids across development and locked suites")
    normalized_prompts = [normalize(row["prompt"]) for row in all_rows]
    if len(normalized_prompts) != len(set(normalized_prompts)):
        errors.append("duplicate normalized prompts across new suites")

    existing = existing_texts(repo, {dev_path.resolve(), locked_path.resolve()})
    existing_norm = {normalize(text) for _, text in existing if normalize(text)}
    exact = [row["id"] for row in all_rows if normalize(row["prompt"]) in existing_norm]
    if exact:
        errors.append(f"exact prompt overlap: {exact}")

    existing_tri = [(source, text, trigrams(text)) for source, text in existing]
    max_ref = {"score": 0.0, "candidate_id": None, "source": None}
    for row in all_rows:
        tri = trigrams(row["prompt"])
        for source, text, other in existing_tri:
            score = jaccard(tri, other)
            if score > max_ref["score"]:
                max_ref = {"score": round(score, 9), "candidate_id": row["id"], "source": source, "source_preview": text[:160]}

    max_within = {"score": 0.0, "left": None, "right": None}
    for index, left in enumerate(all_rows):
        ltri = trigrams(left["prompt"])
        for right in all_rows[index + 1:]:
            score = jaccard(ltri, trigrams(right["prompt"]))
            if score > max_within["score"]:
                max_within = {"score": round(score, 9), "left": left["id"], "right": right["id"]}

    report = {
        "packet": "Phase 3.5B Work Packet 10",
        "baseline": manifest.get("baseline"),
        "status": "passed" if not errors else "failed",
        "development_records": len(dev),
        "locked_records": len(locked),
        "unique_ids": len(set(ids)),
        "category_counts": dict(sorted(Counter(row["category"] for row in all_rows).items())),
        "language_counts": dict(sorted(Counter(row["language"] for row in all_rows).items())),
        "answer_label_counts": dict(sorted(Counter(row["accepted_answers"][0] for row in all_rows).items())),
        "all_training_allowed_false": all(row.get("training_allowed") is False for row in all_rows),
        "exact_overlap_with_existing_raw_or_eval": exact,
        "max_prompt_trigram_jaccard_against_existing": max_ref,
        "max_prompt_trigram_jaccard_within_new_suites": max_within,
        "source_count": len(manifest.get("sources", [])),
        "external_target_tokens": manifest.get("external_target_tokens"),
        "internal_target_tokens": manifest.get("internal_target_tokens"),
        "total_target_tokens": manifest.get("total_target_tokens"),
        "errors": errors,
        "notes": [
            "Lexical similarity is a review aid, not proof of semantic independence.",
            "The source manifest is candidate-only. Acquisition hashes, exact rolling-repository commits, and license approvals remain required before any source becomes training-allowed.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
