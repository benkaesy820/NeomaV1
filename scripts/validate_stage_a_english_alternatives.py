from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL = ROOT / "data/foundation/manifests/stage_a_safe_english_alternatives_v1_candidate.json"
DEFAULT_INTERNAL = ROOT / "data/foundation/manifests/stage_a_internal_english_seed_v1_plan.json"
EXPECTED_BASELINE = "bf70cfb"
EXPECTED_SOURCE_IDS = {
    "simplewiki_20260601",
    "enwikibooks_20260601",
    "enwikiversity_20260601",
}
EXPECTED_COMPONENT_IDS = {
    "developer_dialogue",
    "constraint_english",
    "bug_review_commit_language",
    "cli_config_errors",
    "neoma_self_knowledge",
    "verified_transformations",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_external(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(plan.get("baseline") == EXPECTED_BASELINE, "external baseline mismatch", errors)
    _require(plan.get("training_allowed") is False, "external plan must keep training_allowed=false", errors)
    _require(plan.get("replacement_target_tokens") == 12_000_000, "replacement target must be 12M", errors)
    controls = plan.get("common_controls", {})
    _require(
        controls.get("required_articles_multistream_status") == "done",
        "Wikimedia articles multistream job status must be pinned to done",
        errors,
    )

    gptnl = plan.get("gptnl_policy", {})
    _require(gptnl.get("base_v1_status") == "blocked_not_overridden", "GPT-NL must remain blocked", errors)
    _require(gptnl.get("automatic_retry") is False, "GPT-NL automatic retry must be false", errors)
    _require(gptnl.get("allow_manual_download") is False, "GPT-NL manual download must be false", errors)
    _require(gptnl.get("training_allowed") is False, "GPT-NL training permission must be false", errors)

    sources = plan.get("sources")
    _require(isinstance(sources, list), "sources must be a list", errors)
    if not isinstance(sources, list):
        return errors

    ids = {item.get("source_id") for item in sources if isinstance(item, dict)}
    _require(ids == EXPECTED_SOURCE_IDS, f"unexpected external source IDs: {sorted(ids)}", errors)
    total = 0
    for item in sources:
        if not isinstance(item, dict):
            errors.append("source entry must be an object")
            continue
        source_id = item.get("source_id", "<unknown>")
        _require(item.get("snapshot") == "20260601", f"{source_id}: snapshot must be 20260601", errors)
        _require(item.get("training_allowed") is False, f"{source_id}: training_allowed must be false", errors)
        _require(item.get("status") == "candidate_not_acquired", f"{source_id}: status must be candidate_not_acquired", errors)
        _require(str(item.get("archive_url", "")).startswith("https://dumps.wikimedia.org/"), f"{source_id}: archive must use official HTTPS host", errors)
        _require("20260601" in str(item.get("archive_url", "")), f"{source_id}: archive URL must pin snapshot", errors)
        _require(str(item.get("sha1_url", "")).endswith("sha1sums.txt"), f"{source_id}: missing official SHA-1 manifest", errors)
        _require("CC BY-SA 4.0" in str(item.get("license", "")), f"{source_id}: license statement missing CC BY-SA 4.0", errors)
        _require(bool(item.get("quality_filters")), f"{source_id}: quality filters required", errors)
        _require(bool(item.get("dedup_and_leakage")), f"{source_id}: dedup/leakage rules required", errors)
        _require(bool(item.get("risks")), f"{source_id}: risks required", errors)
        target = item.get("expected_token_target")
        _require(isinstance(target, int) and target > 0, f"{source_id}: positive token target required", errors)
        if isinstance(target, int):
            total += target
    _require(total == 12_000_000, f"external source targets total {total}, expected 12000000", errors)
    return errors


def validate_internal(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(plan.get("baseline") == EXPECTED_BASELINE, "internal baseline mismatch", errors)
    _require(plan.get("training_allowed") is False, "internal plan must keep training_allowed=false", errors)
    _require(plan.get("status") == "planning_only_not_generated", "internal status must remain planning-only", errors)
    _require(plan.get("seed_target_tokens") == 60_000, "internal seed target must be 60K", errors)

    schema = plan.get("record_schema", {})
    required_fields = set(schema.get("required_fields", []))
    for field in {"id", "family_id", "component_id", "text", "content_sha256", "training_allowed"}:
        _require(field in required_fields, f"internal schema missing required field {field}", errors)
    _require(schema.get("training_allowed_required_value") is False, "internal schema must require training_allowed=false", errors)

    components = plan.get("components")
    _require(isinstance(components, list), "components must be a list", errors)
    if not isinstance(components, list):
        return errors
    ids = {item.get("component_id") for item in components if isinstance(item, dict)}
    _require(ids == EXPECTED_COMPONENT_IDS, f"unexpected component IDs: {sorted(ids)}", errors)
    total = 0
    documents = 0
    for item in components:
        if not isinstance(item, dict):
            errors.append("component entry must be an object")
            continue
        component_id = item.get("component_id", "<unknown>")
        _require(item.get("training_allowed") is False, f"{component_id}: training_allowed must be false", errors)
        _require(bool(item.get("document_types")), f"{component_id}: document types required", errors)
        _require(bool(item.get("capabilities")), f"{component_id}: capabilities required", errors)
        _require(bool(item.get("verification")), f"{component_id}: verification rules required", errors)
        tokens = item.get("target_tokens")
        docs = item.get("target_documents")
        _require(isinstance(tokens, int) and tokens > 0, f"{component_id}: positive target_tokens required", errors)
        _require(isinstance(docs, int) and docs > 0, f"{component_id}: positive target_documents required", errors)
        if isinstance(tokens, int):
            total += tokens
        if isinstance(docs, int):
            documents += docs
    _require(total == 60_000, f"internal component targets total {total}, expected 60000", errors)
    _require(documents == 460, f"internal document targets total {documents}, expected 460", errors)
    return errors


def validate(external_path: Path, internal_path: Path) -> dict[str, Any]:
    external = load_json(external_path)
    internal = load_json(internal_path)
    errors = validate_external(external) + validate_internal(internal)
    return {
        "ok": not errors,
        "errors": errors,
        "external_source_count": len(external.get("sources", [])),
        "external_target_tokens": sum(x.get("expected_token_target", 0) for x in external.get("sources", []) if isinstance(x, dict)),
        "internal_component_count": len(internal.get("components", [])),
        "internal_seed_target_tokens": internal.get("seed_target_tokens"),
        "training_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage A safe-English alternative and internal-seed plans.")
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--internal", type=Path, default=DEFAULT_INTERNAL)
    args = parser.parse_args()
    report = validate(args.external, args.internal)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
