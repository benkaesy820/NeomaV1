from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover - optional runtime enhancement
    Tokenizer = None  # type: ignore[assignment]


LANGUAGE_TARGET_SHARES = {
    "python": 0.30,
    "typescript": 0.20,
    "javascript": 0.175,
    "powershell": 0.125,
    "sql": 0.125,
    "text": 0.075,
}

CAPABILITY_TARGETS = {
    # These are coverage slots from TRAINING_BATCH_PLAN.md. They intentionally
    # total 450 for a 400-record corpus because one record may cover a primary
    # category and one or more secondary capabilities.
    "functions_and_data": 90,
    "validation": 55,
    "debugging": 55,
    "tests": 55,
    "files_apis_automation": 50,
    "sql_database": 45,
    "security": 35,
    "efficiency": 35,
    "explanation": 30,
}

GENERIC_EVAL_TERMS = {
    "valueerror",
    "typeerror",
    "return",
    "raise",
    "def",
    "function",
    "string",
    "number",
    "array",
    "list",
    "dict",
    "set",
    "true",
    "false",
    "null",
    "undefined",
    "select",
    "where",
    "unittest",
    "specific",
    "deterministic",
    "parameter",
    "membership",
    "integration",
    "exact",
    "message",
    "error",
    "stable",
    "sorted",
    "validation",
    "invalid",
    "empty",
    "exception",
    "input",
    "output",
}

PATTERNS = {
    "value_error": r"\bValueError\b",
    "type_error": r"\bTypeError\b",
    "truthiness_guard": r"\bif\s+not\s+[A-Za-z_(]",
    "explicit_none_or_null": r"\b(?:is\s+None|is\s+not\s+None|===\s+null|!==\s+null|===\s+undefined|!==\s+undefined)\b",
    "object_create_null": r"Object\.create\(null\)",
    "array_is_array": r"Array\.isArray\(",
    "unittest": r"\bunittest\b",
    "subtest": r"\bsubTest\b",
    "pytest": r"\bpytest\b",
    "pester": r"\b(?:Describe|It|Should)\b",
    "transaction_begin": r"\bBEGIN\b",
    "transaction_commit": r"\bCOMMIT\b",
    "transaction_rollback": r"\bROLLBACK\b",
    "parameter_placeholder": r"(?:\$\d+|%s|\?|:\w+)",
    "strip_or_trim": r"\.(?:strip|trim)\(",
    "sort_operation": r"\b(?:sorted\(|\.sort\()",
    "set_membership": r"\bset\(|new\s+Set\(",
    "async_await": r"\b(?:async|await|Promise<)\b",
    "pathlib_or_path": r"\b(?:Path|pathlib|Resolve-Path|Join-Path)\b",
    "json_parse": r"\b(?:json\.loads?|JSON\.parse|ConvertFrom-Json)\b",
    "explicit_size_limit": r"\b(?:max_|limit|length|size|bytes?)\b",
}

SECONDARY_TAG_RULES: dict[str, tuple[str, ...]] = {
    "validation": (r"\bvalidat", r"\breject", r"\brequired\b", r"\bparse\b", r"\bboundar"),
    "files": (r"\bfile", r"\bpath", r"\bcsv\b", r"\bjsonl\b", r"\bnewline", r"\bbom\b"),
    "data": (r"\bjson\b", r"\bcsv\b", r"\bmap\b", r"\bdict", r"\bgroup", r"\btransform"),
    "debugging": (r"\bfix\b", r"\bbug\b", r"\bregression", r"<bad_code>"),
    "tests": (r"\btest", r"\bunittest\b", r"\bpester\b", r"\bassert"),
    "database": (r"\bsql\b", r"\bpostgres", r"\btransaction", r"\bselect\b", r"\binsert\b", r"\bupdate\b"),
    "security": (r"\bauthori", r"\bauthentic", r"\bsecret", r"\btoken", r"\bcsrf", r"\btenant", r"\btraversal", r"\bredact"),
    "efficiency": (r"\befficient", r"\blinear time", r"\bcomplexity", r"\bavoid repeated", r"\bbounded"),
    "api": (r"\brequest\b", r"\bresponse\b", r"\bhttp\b", r"\bapi\b", r"\bheader\b"),
    "async": (r"\basync\b", r"\bawait\b", r"\bpromise\b", r"\btimeout\b"),
    "explanation": (r"\bexplain", r"\bwhy\b", r"\bdifference between\b"),
    "transactions": (r"\btransaction", r"\bbegin\b", r"\bcommit\b", r"\brollback\b"),
}

PAIR_TAGS = ("instruction", "constraints", "bad_code", "reasoning", "answer")


@dataclass(frozen=True)
class AuditRecord:
    record_id: str
    source_kind: str
    source_path: str
    language: str
    category: str
    difficulty: str
    instruction: str
    constraints: tuple[str, ...]
    answer: str
    bad_code: str
    reasoning: str
    edge_cases: tuple[str, ...]
    quality_notes: tuple[str, ...]
    rendered: str


@dataclass(frozen=True)
class DuplicateFlag:
    left_id: str
    right_id: str
    field: str
    score: float
    reason: str
    left_source: str
    right_source: str


@dataclass(frozen=True)
class LeakageFlag:
    record_id: str
    eval_id: str
    severity: str
    field: str
    score: float
    reason: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def canonical_text(text: str) -> str:
    """Normalize text for containment checks while preserving code tokens."""
    return " ".join(token.lower() for token in lexical_tokens(text))


def lexical_tokens(text: str) -> list[str]:
    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==={0,1}|!==|==|!=|<=|>=|=>|::|\$\d+|[^\s]",
        text,
    )


def trigrams(text: str) -> set[tuple[str, str, str]]:
    tokens = [token.lower() for token in lexical_tokens(text)]
    return set(zip(tokens, tokens[1:], tokens[2:])) if len(tokens) >= 3 else set()


def jaccard(left: set[Any], right: set[Any]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0.0, "mean": 0.0, "p90": 0.0, "p95": 0.0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 3),
        "p90": round(percentile(values, 0.90), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": max(values),
    }


def bullet_block(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_jsonl_row(row: dict[str, Any]) -> str:
    parts = [
        f"<instruction>\n{row['instruction'].strip()}\n</instruction>",
        f"<constraints>\n{bullet_block(row.get('constraints', []))}\n</constraints>",
    ]
    bad_code = str(row.get("bad_code", "") or "").strip()
    reasoning = str(row.get("reasoning", "") or "").strip()
    if bad_code:
        parts.append(f"<bad_code>\n{bad_code}\n</bad_code>")
    if reasoning:
        parts.append(f"<reasoning>\n{reasoning}\n</reasoning>")
    parts.append(f"<answer>\n{row['answer'].strip()}\n</answer>")
    notes: list[str] = []
    edge_cases = [str(item).strip() for item in row.get("edge_cases", []) if str(item).strip()]
    quality_notes = [str(item).strip() for item in row.get("quality_notes", []) if str(item).strip()]
    if edge_cases:
        notes.append("Edge cases:")
        notes.extend(f"- {item}" for item in edge_cases)
    if quality_notes:
        notes.append("Quality notes:")
        notes.extend(f"- {item}" for item in quality_notes)
    if notes:
        parts.append(f"<reasoning>\n{chr(10).join(notes)}\n</reasoning>")
    return "\n".join(parts)


def extract_tag(chunk: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", chunk, flags=re.IGNORECASE | re.DOTALL)
    return "\n\n".join(match.strip() for match in matches if match.strip())


def extract_bullets(text: str) -> tuple[str, ...]:
    items = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*[-*]\s+", "", line).strip()
        if cleaned:
            items.append(cleaned)
    return tuple(items)


def infer_language(instruction: str, answer: str) -> str:
    haystack = f"{instruction}\n{answer}".lower()
    if "powershell" in haystack or re.search(r"\b(?:param\s*\(|get-childitem|where-object|foreach-object|write-output)\b", haystack):
        return "powershell"
    if "typescript" in haystack or re.search(r"\b(?:interface|type\s+\w+\s*=|unknown\b|promise<|readonly\s+\w+\[\])", haystack):
        return "typescript"
    if "javascript" in haystack or re.search(r"\b(?:const|let|function)\s+\w+\s*\([^)]*\)\s*\{", haystack):
        return "javascript"
    if "sql" in haystack or re.search(r"\b(?:select|insert into|update\s+\w+\s+set|create table|with\s+\w+\s+as)\b", haystack):
        return "sql"
    if re.search(r"\b(?:explain|describe the difference|why does)\b", instruction.lower()) and not re.search(r"\b(?:python|typescript|javascript|powershell|sql)\b", instruction.lower()):
        return "text"
    return "python"


def infer_category(instruction: str, bad_code: str, answer: str, language: str) -> str:
    text = f"{instruction}\n{answer}".lower()
    if bad_code:
        return "debugging"
    if re.search(r"\b(?:test|unittest|pester|assert)\b", text):
        return "tests"
    if language == "sql" or re.search(r"\b(?:database|query|table|transaction)\b", text):
        return "database"
    if re.search(r"\b(?:file|path|directory|json|csv|jsonl|utf-8)\b", text):
        return "files"
    if re.search(r"\b(?:secure|security|authorization|redact|secret|token|traversal|tenant)\b", text):
        return "security"
    if re.search(r"\b(?:efficient|complexity|linear time|avoid repeated|optimi[sz])\b", text):
        return "efficiency"
    if language == "text" or re.search(r"\b(?:explain|describe why|difference between)\b", instruction.lower()):
        return "explanation"
    if re.search(r"\b(?:validate|parse|required|reject|normalize|non-empty|must not)\b", text):
        return "validation"
    return "function"


def infer_difficulty(instruction: str, constraints: Sequence[str], answer: str, bad_code: str) -> str:
    score = 0
    score += 1 if len(constraints) >= 4 else 0
    score += 1 if len(lexical_tokens(answer)) >= 110 else 0
    score += 1 if bad_code else 0
    score += 1 if re.search(r"\b(?:async|transaction|rollback|authorization|concurrency|idempotent)\b", f"{instruction}\n{answer}", re.I) else 0
    return "intermediate" if score >= 2 else "basic"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def load_accepted_records(root: Path) -> tuple[list[AuditRecord], list[dict[str, Any]]]:
    incoming = root / "data" / "incoming"
    records: list[AuditRecord] = []
    integrity: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in sorted(incoming.glob("phase3_5b_batch*_accepted.jsonl")):
        rows = load_jsonl(path)
        rendered_rows: list[str] = []
        for row in rows:
            record_id = str(row.get("id", "")).strip()
            if not record_id:
                raise ValueError(f"{path}: accepted row has no id")
            if record_id in seen_ids:
                raise ValueError(f"duplicate accepted id: {record_id}")
            seen_ids.add(record_id)
            rendered = render_jsonl_row(row)
            rendered_rows.append(rendered)
            records.append(
                AuditRecord(
                    record_id=record_id,
                    source_kind="accepted_jsonl",
                    source_path=path.relative_to(root).as_posix(),
                    language=str(row.get("language", "unknown")).lower(),
                    category=str(row.get("category", "unknown")).lower(),
                    difficulty=str(row.get("difficulty", "unknown")).lower(),
                    instruction=str(row.get("instruction", "")).strip(),
                    constraints=tuple(str(item).strip() for item in row.get("constraints", []) if str(item).strip()),
                    answer=str(row.get("answer", "")).strip(),
                    bad_code=str(row.get("bad_code", "") or "").strip(),
                    reasoning=str(row.get("reasoning", "") or "").strip(),
                    edge_cases=tuple(str(item).strip() for item in row.get("edge_cases", []) if str(item).strip()),
                    quality_notes=tuple(str(item).strip() for item in row.get("quality_notes", []) if str(item).strip()),
                    rendered=rendered,
                )
            )
        batch_match = re.search(r"batch(\d+)_accepted", path.name)
        raw_name = f"phase3_5b_batch{batch_match.group(1)}_accepted.txt" if batch_match else ""
        raw_path = root / "data" / "raw" / raw_name
        expected = "\n\n".join(rendered_rows).strip()
        actual = raw_path.read_text(encoding="utf-8", errors="replace").strip() if raw_path.exists() else None
        integrity.append(
            {
                "jsonl": path.relative_to(root).as_posix(),
                "raw": raw_path.relative_to(root).as_posix() if raw_name else None,
                "rows": len(rows),
                "raw_exists": raw_path.exists(),
                "raw_matches_rendered_jsonl": actual == expected,
                "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.exists() else None,
            }
        )
    return records, integrity


def load_legacy_records(root: Path) -> tuple[list[AuditRecord], list[dict[str, Any]]]:
    raw_dir = root / "data" / "raw"
    records: list[AuditRecord] = []
    foundation_seed_files: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        if re.fullmatch(r"phase3_5b_batch\d+_accepted\.txt", path.name):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        starts = [match.start() for match in re.finditer(r"<instruction(?:\s[^>]*)?>", text, flags=re.I)]
        if not starts:
            foundation_seed_files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "characters": len(text),
                    "proxy_tokens": len(lexical_tokens(text)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            continue
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(text)
            chunk = text[start:end].strip()
            instruction = extract_tag(chunk, "instruction")
            constraints_text = extract_tag(chunk, "constraints")
            answer = extract_tag(chunk, "answer")
            bad_code = extract_tag(chunk, "bad_code")
            reasoning = extract_tag(chunk, "reasoning")
            language = infer_language(instruction, answer)
            category = infer_category(instruction, bad_code, answer, language)
            constraints = extract_bullets(constraints_text)
            difficulty = infer_difficulty(instruction, constraints, answer, bad_code)
            records.append(
                AuditRecord(
                    record_id=f"legacy:{path.stem}:{index + 1:03d}",
                    source_kind="legacy_raw",
                    source_path=path.relative_to(root).as_posix(),
                    language=language,
                    category=category,
                    difficulty=difficulty,
                    instruction=instruction,
                    constraints=constraints,
                    answer=answer,
                    bad_code=bad_code,
                    reasoning=reasoning,
                    edge_cases=(),
                    quality_notes=(),
                    rendered=chunk,
                )
            )
    return records, foundation_seed_files


def secondary_tags(record: AuditRecord) -> tuple[str, ...]:
    text = record.rendered.lower()
    tags = []
    for tag, patterns in SECONDARY_TAG_RULES.items():
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
            tags.append(tag)
    return tuple(tags)


def capability_tags(record: AuditRecord) -> tuple[str, ...]:
    """Map one record to the overlapping capability slots in the data plan."""
    tags = set(secondary_tags(record))
    primary = record.category
    capabilities: set[str] = set()

    if primary in {"function", "data"} or "data" in tags:
        capabilities.add("functions_and_data")
    if primary == "validation" or "validation" in tags:
        capabilities.add("validation")
    if primary == "debugging" or "debugging" in tags:
        capabilities.add("debugging")
    if primary == "tests" or "tests" in tags:
        capabilities.add("tests")
    if primary == "files" or tags.intersection({"files", "api"}):
        capabilities.add("files_apis_automation")
    if primary == "database" or record.language == "sql" or "database" in tags:
        capabilities.add("sql_database")
    if primary == "security" or "security" in tags:
        capabilities.add("security")
    if primary == "efficiency" or "efficiency" in tags:
        capabilities.add("efficiency")
    if primary == "explanation" or "explanation" in tags:
        capabilities.add("explanation")
    return tuple(sorted(capabilities))


def strip_comments_and_literals(text: str, language: str) -> str:
    if language == "python":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            pass
        else:
            class Normalizer(ast.NodeTransformer):
                def visit_Name(self, node: ast.Name) -> ast.AST:
                    return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)

                def visit_arg(self, node: ast.arg) -> ast.AST:
                    node.arg = "ARG"
                    return node

                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                    node.name = "FUNC"
                    return self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
                    node.name = "FUNC"
                    return self.generic_visit(node)

                def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                    node.name = "CLASS"
                    return self.generic_visit(node)

                def visit_Constant(self, node: ast.Constant) -> ast.AST:
                    if isinstance(node.value, str):
                        node.value = "STR"
                    elif isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
                        node.value = 0
                    return node

            normalized = Normalizer().visit(tree)
            ast.fix_missing_locations(normalized)
            return ast.dump(normalized, include_attributes=False)
    value = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    value = re.sub(r"(?m)//.*$|#.*$|--.*$", " ", value)
    value = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`", " STR ", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", " NUM ", value)
    keywords = {
        "if", "else", "for", "while", "return", "function", "async", "await", "class", "interface", "type",
        "const", "let", "var", "new", "true", "false", "null", "undefined", "select", "from", "where", "join",
        "insert", "into", "update", "delete", "create", "table", "begin", "commit", "rollback", "param", "foreach",
        "try", "catch", "finally", "throw", "raise", "def", "import", "with", "as", "and", "or", "not", "in",
    }
    tokens = lexical_tokens(value)
    normalized_tokens = [token.lower() if token.lower() in keywords else ("ID" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) else token) for token in tokens]
    return " ".join(normalized_tokens)


def duplicate_flags(records: Sequence[AuditRecord]) -> list[DuplicateFlag]:
    flags: list[DuplicateFlag] = []
    fields = ("instruction", "answer", "rendered")
    trigram_cache: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    normalized_cache: dict[tuple[str, str], str] = {}
    structural: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        signature = strip_comments_and_literals(record.answer, record.language)
        if record.language != "text" and len(lexical_tokens(record.answer)) >= 12 and signature.strip():
            structural[hashlib.sha256(signature.encode("utf-8")).hexdigest()].append(record)
        for field in fields:
            value = getattr(record, field)
            normalized_cache[(record.record_id, field)] = normalize_text(value)
            trigram_cache[(record.record_id, field)] = trigrams(value)

    exact_seen: set[tuple[str, str, str]] = set()
    for left, right in combinations(records, 2):
        for field in fields:
            left_norm = normalized_cache[(left.record_id, field)]
            right_norm = normalized_cache[(right.record_id, field)]
            if left_norm and left_norm == right_norm:
                key = tuple(sorted((left.record_id, right.record_id))) + (field,)
                if key not in exact_seen:
                    exact_seen.add(key)
                    flags.append(DuplicateFlag(left.record_id, right.record_id, field, 1.0, "exact normalized match", left.source_path, right.source_path))
                continue
            score = jaccard(trigram_cache[(left.record_id, field)], trigram_cache[(right.record_id, field)])
            threshold = {"instruction": 0.50, "answer": 0.68, "rendered": 0.56}[field]
            if score >= threshold:
                flags.append(DuplicateFlag(left.record_id, right.record_id, field, round(score, 6), "high trigram Jaccard", left.source_path, right.source_path))

    for group in structural.values():
        if len(group) < 2:
            continue
        for left, right in combinations(group, 2):
            if normalize_text(left.answer) == normalize_text(right.answer):
                continue
            flags.append(DuplicateFlag(left.record_id, right.record_id, "answer_structure", 1.0, "same normalized code structure", left.source_path, right.source_path))

    unique: dict[tuple[str, str, str, str], DuplicateFlag] = {}
    for flag in flags:
        left_id, right_id = sorted((flag.left_id, flag.right_id))
        unique[(left_id, right_id, flag.field, flag.reason)] = flag
    return sorted(unique.values(), key=lambda item: (-item.score, item.field, item.left_id, item.right_id))


def meaningful_eval_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("required_terms", "forbidden_terms"):
        for value in row.get(key, []) or []:
            term = str(value).strip()
            normalized = normalize_text(term)
            if not normalized or normalized in GENERIC_EVAL_TERMS:
                continue
            token_count = len(lexical_tokens(term))
            code_like = bool(re.search(r"[.()$:<>=\[\]{}'\"]", term))
            # Single ordinary words create hundreds of noisy flags. Keep
            # code-shaped terms, meaningful phrases, and unusually distinctive
            # identifiers only.
            if code_like and len(normalized) >= 6:
                terms.append(term)
            elif token_count >= 2 and len(normalized) >= 12:
                terms.append(term)
            elif token_count == 1 and len(normalized) >= 12:
                terms.append(term)
    return terms


def leakage_flags(records: Sequence[AuditRecord], eval_paths: Sequence[Path]) -> list[LeakageFlag]:
    flags: list[LeakageFlag] = []
    eval_rows: list[dict[str, Any]] = []
    for path in eval_paths:
        if path.exists():
            eval_rows.extend(load_jsonl(path))
    for record in records:
        record_full = canonical_text(record.rendered)
        record_instruction_tri = trigrams(record.instruction)
        record_answer_tri = trigrams(record.answer)
        for row in eval_rows:
            eval_id = str(row.get("id", "unknown"))
            prompt = str(row.get("prompt", "")).strip()
            prompt_norm = canonical_text(prompt)
            if prompt_norm and prompt_norm in record_full:
                flags.append(LeakageFlag(record.record_id, eval_id, "critical", "prompt", 1.0, "exact normalized eval prompt contained in training record"))
            prompt_score = jaccard(record_instruction_tri, trigrams(prompt))
            if prompt_score >= 0.42:
                flags.append(LeakageFlag(record.record_id, eval_id, "review", "prompt", round(prompt_score, 6), "high instruction/prompt trigram similarity"))
            test_code = str(row.get("test_code", "") or "").strip()
            test_norm = canonical_text(test_code)
            if len(test_norm) >= 30 and test_norm in record_full:
                flags.append(LeakageFlag(record.record_id, eval_id, "critical", "test_code", 1.0, "exact normalized eval test code contained in training record"))
            notes = str(row.get("expected_behavior_notes", "") or "").strip()
            note_score = max(jaccard(record_instruction_tri, trigrams(notes)), jaccard(record_answer_tri, trigrams(notes)))
            if note_score >= 0.55:
                flags.append(LeakageFlag(record.record_id, eval_id, "review", "expected_behavior_notes", round(note_score, 6), "high similarity to held-out behavior notes"))
            for term in meaningful_eval_terms(row):
                term_norm = canonical_text(term)
                if term_norm and term_norm in record_full:
                    flags.append(LeakageFlag(record.record_id, eval_id, "informational", "required_or_forbidden_term", 1.0, f"held-out distinctive term appears: {term}"))
            for line in test_code.splitlines():
                line_norm = canonical_text(line)
                if len(line_norm) >= 24 and line_norm in record_full:
                    flags.append(LeakageFlag(record.record_id, eval_id, "review", "test_fixture_line", 1.0, f"held-out test fixture line appears: {line.strip()[:100]}"))
    unique: dict[tuple[str, str, str, str], LeakageFlag] = {}
    for flag in flags:
        key = (flag.record_id, flag.eval_id, flag.field, flag.reason)
        unique[key] = flag
    severity_order = {"critical": 0, "review": 1, "informational": 2}
    return sorted(unique.values(), key=lambda item: (severity_order[item.severity], -item.score, item.eval_id, item.record_id))


class TokenCounter:
    def __init__(self, tokenizer_path: Path | None):
        self.path = tokenizer_path if tokenizer_path and tokenizer_path.exists() else None
        self.tokenizer = None
        self.mode = "proxy_lexical_tokens"
        if self.path is not None:
            if Tokenizer is None:
                raise SystemExit(
                    "A tokenizer file was selected, but the 'tokenizers' package is unavailable. "
                    "Install project requirements or omit --tokenizer to use the labeled lexical proxy."
                )
            try:
                self.tokenizer = Tokenizer.from_file(str(self.path))
            except Exception as exc:  # pragma: no cover - library error detail varies
                raise SystemExit(f"Could not load tokenizer {self.path}: {exc}") from exc
            self.mode = "project_tokenizer"

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text).ids)
        return len(lexical_tokens(text))

    def exact_supervised_count(self, record: AuditRecord) -> int | None:
        if self.tokenizer is None:
            return None
        answer_open_id = self.tokenizer.token_to_id("<answer>")
        answer_close_id = self.tokenizer.token_to_id("</answer>")
        eos_id = self.tokenizer.token_to_id("<eos>")
        if answer_open_id is None or answer_close_id is None:
            return None
        ids = self.tokenizer.encode(record.rendered).ids
        try:
            start = ids.index(answer_open_id)
            end = ids.index(answer_close_id, start + 1)
        except ValueError:
            return None
        count = end - start
        if eos_id is not None:
            count += 1
        return count


def record_length_rows(records: Sequence[AuditRecord], counter: TokenCounter) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        full_tokens = counter.count(record.rendered)
        answer_tokens = counter.count(record.answer)
        exact_supervised = counter.exact_supervised_count(record)
        supervised = exact_supervised if exact_supervised is not None else answer_tokens
        rows.append(
            {
                "record_id": record.record_id,
                "source_kind": record.source_kind,
                "source_path": record.source_path,
                "language": record.language,
                "category": record.category,
                "difficulty": record.difficulty,
                "instruction_tokens": counter.count(record.instruction),
                "constraints_tokens": counter.count("\n".join(record.constraints)),
                "bad_code_tokens": counter.count(record.bad_code),
                "reasoning_tokens": counter.count(record.reasoning),
                "answer_tokens": answer_tokens,
                "full_record_tokens": full_tokens,
                "supervised_tokens": supervised,
                "supervised_percentage": round((supervised / full_tokens * 100.0) if full_tokens else 0.0, 3),
                "characters": len(record.rendered),
                "lines": record.rendered.count("\n") + 1,
            }
        )
    return rows


def pattern_counts(records: Sequence[AuditRecord]) -> list[dict[str, Any]]:
    rows = []
    for name, pattern in PATTERNS.items():
        record_hits = []
        occurrences = 0
        regex = re.compile(pattern, flags=re.I | re.M)
        for record in records:
            matches = regex.findall(record.rendered)
            if matches:
                record_hits.append(record.record_id)
                occurrences += len(matches)
        rows.append({"pattern": name, "records": len(record_hits), "occurrences": occurrences, "record_ids": record_hits[:30]})
    return sorted(rows, key=lambda item: (-item["records"], item["pattern"]))


def matrix(records: Sequence[AuditRecord], row_field: str, col_field: str) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        result[str(getattr(record, row_field))][str(getattr(record, col_field))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def language_gap_analysis(counts: Counter[str], total: int) -> list[dict[str, Any]]:
    rows = []
    tolerance = max(2.0, total * 0.02)
    for language, share in LANGUAGE_TARGET_SHARES.items():
        actual = counts.get(language, 0)
        desired = total * share
        rows.append(
            {
                "language": language,
                "count": actual,
                "actual_share": round(actual / total if total else 0.0, 4),
                "target_share": share,
                "delta_examples_at_current_size": round(actual - desired, 2),
                "status": "under" if actual < desired - tolerance else ("over" if actual > desired + tolerance else "near_target"),
            }
        )
    return sorted(rows, key=lambda item: item["delta_examples_at_current_size"])


def capability_gap_analysis(records: Sequence[AuditRecord]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(capability_tags(record))
    rows = []
    for capability, target in CAPABILITY_TARGETS.items():
        actual = counts.get(capability, 0)
        rows.append(
            {
                "capability": capability,
                "count": actual,
                "target_for_400_reviewed_examples": target,
                "progress": round(actual / target if target else 0.0, 4),
                "remaining_to_target": max(0, target - actual),
            }
        )
    return sorted(rows, key=lambda item: (item["progress"], item["capability"]))


def curriculum_decision(
    total: int,
    duplicates: Sequence[DuplicateFlag],
    leakage: Sequence[LeakageFlag],
    length_rows: Sequence[dict[str, Any]],
    language_gaps: Sequence[dict[str, Any]],
    expected_context: int,
) -> dict[str, Any]:
    critical_leaks = sum(1 for item in leakage if item.severity == "critical")
    exact_duplicates = sum(1 for item in duplicates if item.reason == "exact normalized match")
    over_context = sum(1 for row in length_rows if row["full_record_tokens"] > expected_context)
    under_languages = [item["language"] for item in language_gaps if item["status"] == "under"]
    if critical_leaks or exact_duplicates:
        status = "structural_cleanup_required"
        reason = "Critical leakage or exact duplicate flags must be resolved before adding more data."
    elif total < 300:
        status = "targeted_gap_batch_recommended"
        reason = "The curriculum remains below the 300-example lower bound; add one measured gap batch, not another broad batch."
    elif over_context > max(5, int(total * 0.15)):
        status = "context_length_cleanup_required"
        reason = f"Too many records exceed the intended {expected_context}-token context and need shortening, splitting, or deferral."
    elif under_languages:
        status = "targeted_gap_batch_recommended"
        reason = "Language balance still has material deficits that should be corrected deliberately."
    else:
        status = "ready_to_freeze_instruction_corpus_v0_1"
        reason = "No blocking integrity issue was found and the curriculum is large enough for a freeze review."
    return {
        "status": status,
        "reason": reason,
        "critical_leakage_flags": critical_leaks,
        "exact_duplicate_flags": exact_duplicates,
        "expected_context": expected_context,
        "records_over_expected_context": over_context,
        "underrepresented_languages": under_languages,
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def human_review_queue(
    duplicates: Sequence[DuplicateFlag],
    leakage: Sequence[LeakageFlag],
    length_rows: Sequence[dict[str, Any]],
    expected_context: int,
    length_mode: str,
) -> list[dict[str, Any]]:
    """Build a deterministic queue that Leo can annotate without editing reports."""
    rows: list[dict[str, Any]] = []
    for item in duplicates:
        priority = "critical" if item.reason == "exact normalized match" else "review"
        rows.append(
            {
                "finding_id": f"duplicate:{item.left_id}:{item.right_id}:{item.field}:{item.reason}",
                "finding_type": "duplicate",
                "priority": priority,
                "record_id": item.left_id,
                "related_id": item.right_id,
                "field": item.field,
                "score": item.score,
                "reason": item.reason,
                "source": f"{item.left_source} | {item.right_source}",
                "reviewer_decision": "",
                "action": "",
                "notes": "",
            }
        )
    for item in leakage:
        if item.severity == "informational":
            continue
        rows.append(
            {
                "finding_id": f"leakage:{item.record_id}:{item.eval_id}:{item.field}:{item.reason}",
                "finding_type": "eval_leakage",
                "priority": item.severity,
                "record_id": item.record_id,
                "related_id": item.eval_id,
                "field": item.field,
                "score": item.score,
                "reason": item.reason,
                "source": "held-out evaluation suite",
                "reviewer_decision": "",
                "action": "",
                "notes": "",
            }
        )
    # Proxy counts are deliberately conservative and can overestimate the
    # project's eventual subword count. Only queue severe proxy outliers; the
    # full outlier CSV still contains every threshold hit. With a real project
    # tokenizer, queue every record that exceeds the requested context.
    context_queue_threshold = (
        expected_context
        if length_mode == "project_tokenizer"
        else int(math.ceil(expected_context * 1.25))
    )
    answer_queue_threshold = (
        max(96, expected_context // 2)
        if length_mode == "project_tokenizer"
        else max(160, int(math.ceil(expected_context * 0.75)))
    )
    for row in length_rows:
        reasons = []
        if row["full_record_tokens"] > context_queue_threshold:
            if length_mode == "project_tokenizer":
                reasons.append(f"full record exceeds {expected_context}-token context")
            else:
                reasons.append(
                    f"lexical proxy exceeds severe-outlier threshold {context_queue_threshold}; "
                    "rerun with the project tokenizer before rewriting"
                )
        if row["answer_tokens"] > answer_queue_threshold:
            if length_mode == "project_tokenizer":
                reasons.append("answer is unusually long")
            else:
                reasons.append(
                    f"answer lexical proxy exceeds severe-outlier threshold {answer_queue_threshold}; "
                    "rerun with the project tokenizer before rewriting"
                )
        if row["supervised_percentage"] < 10.0:
            reasons.append("less than 10% of record is supervised")
        if not reasons:
            continue
        rows.append(
            {
                "finding_id": f"length:{row['record_id']}",
                "finding_type": "context_or_supervision",
                "priority": "review",
                "record_id": row["record_id"],
                "related_id": "",
                "field": "full_record_tokens",
                "score": row["full_record_tokens"],
                "reason": "; ".join(reasons),
                "source": row["source_path"],
                "reviewer_decision": "",
                "action": "",
                "notes": "",
            }
        )
    priority_order = {"critical": 0, "review": 1}
    return sorted(
        rows,
        key=lambda item: (
            priority_order.get(str(item["priority"]), 9),
            str(item["finding_type"]),
            str(item["record_id"]),
            str(item["related_id"]),
            str(item["field"]),
        ),
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_markdown(summary: dict[str, Any]) -> str:
    counts = summary["record_counts"]
    decision = summary["decision"]
    lines = [
        "# NeomaV1 Phase 3.5B Corpus Audit",
        "",
        f"**Decision:** `{decision['status']}`",
        "",
        decision["reason"],
        "",
        "## Scope",
        "",
        f"- Total instruction records: **{counts['total_instruction_records']}**",
        f"- Accepted Phase 3.5B JSONL records: **{counts['accepted_phase3_5b_records']}**",
        f"- Legacy protocol records: **{counts['legacy_protocol_records']}**",
        f"- Non-instruction foundation seed files inventoried: **{counts['foundation_seed_files']}**",
        f"- Length metric: **{summary['length_analysis']['mode']}**",
        "",
        "## Languages",
        "",
        markdown_table(
            ["Language", "Count", "Actual share", "Target share", "Status"],
            [
                [row["language"], row["count"], f"{row['actual_share']:.1%}", f"{row['target_share']:.1%}", row["status"]]
                for row in summary["curriculum_gaps"]["language"]
            ],
        ),
        "",
        "## Primary categories",
        "",
        markdown_table(
            ["Category", "Count"],
            [[category, count] for category, count in summary["category_counts"].items()],
        ),
        "",
        "## Overlapping capability coverage",
        "",
        "Targets come from the 400-record plan and intentionally total 450 slots because one record may cover more than one capability.",
        "",
        markdown_table(
            ["Capability", "Covered records", "Target slots", "Progress"],
            [
                [row["capability"], row["count"], row["target_for_400_reviewed_examples"], f"{row['progress']:.1%}"]
                for row in summary["curriculum_gaps"]["capability"]
            ],
        ),
        "",
        "## Length and context fit",
        "",
        markdown_table(
            ["Threshold", "Fits", "Exceeds"],
            [
                [threshold, values["fits"], values["exceeds"]]
                for threshold, values in summary["length_analysis"]["context_fit"].items()
            ],
        ),
        "",
        "Full-record length statistics:",
        "",
        "```json",
        json.dumps(summary["length_analysis"]["full_record_tokens"], indent=2),
        "```",
        "",
        "## Supervision",
        "",
        f"- Records with zero supervised tokens: **{summary['supervision']['zero_supervised_records']}**",
        f"- Median supervised percentage: **{summary['supervision']['supervised_percentage']['median']:.2f}%**",
        f"- Mean supervised percentage: **{summary['supervision']['supervised_percentage']['mean']:.2f}%**",
        "",
        "## Duplicate and leakage review",
        "",
        f"- Duplicate flags: **{summary['duplicate_analysis']['total_flags']}**",
        f"- Exact duplicate flags: **{summary['duplicate_analysis']['exact_flags']}**",
        f"- Structural-similarity flags: **{summary['duplicate_analysis']['structural_flags']}**",
        f"- Critical leakage flags: **{summary['leakage_analysis']['critical']}**",
        f"- Review leakage flags: **{summary['leakage_analysis']['review']}**",
        f"- Informational term overlaps: **{summary['leakage_analysis']['informational']}**",
        "",
        "See the CSV reports before removing, merging, or rewriting any record. Flags require human review; they are not automatic deletion decisions.",
        "",
        "## Integrity",
        "",
        markdown_table(
            ["JSONL", "Rows", "Raw exists", "Raw matches"],
            [
                [item["jsonl"], item["rows"], item["raw_exists"], item["raw_matches_rendered_jsonl"]]
                for item in summary["integrity"]["accepted_batch_render_checks"]
            ],
        ),
        "",
        "## Recommended order",
        "",
        "1. Review every critical/review leakage flag.",
        "2. Review exact and high structural duplicate flags.",
        f"3. Inspect records over the intended {summary['length_analysis']['expected_context']}-token context.",
        "4. Confirm language-by-category gaps.",
        "5. Choose one targeted correction batch only from measured gaps.",
        "6. Rerun this audit after that batch, then decide whether to freeze Instruction Corpus v0.1.",
        "",
    ]
    return "\n".join(lines)


def build_gap_markdown(summary: dict[str, Any]) -> str:
    language_gaps = summary["curriculum_gaps"]["language"]
    capability_gaps = summary["curriculum_gaps"]["capability"]
    matrix_data = summary["language_category_matrix"]
    languages = sorted(matrix_data)
    categories = sorted({category for values in matrix_data.values() for category in values})
    lines = [
        "# NeomaV1 Curriculum Gap Analysis",
        "",
        "This report is diagnostic. It does not authorize generation or admission of another batch.",
        "",
        "## Language balance",
        "",
        markdown_table(
            ["Language", "Count", "Delta at current size", "Status"],
            [[row["language"], row["count"], row["delta_examples_at_current_size"], row["status"]] for row in language_gaps],
        ),
        "",
        "## Overlapping capability coverage",
        "",
        markdown_table(
            ["Capability", "Covered records", "Remaining target slots"],
            [[row["capability"], row["count"], row["remaining_to_target"]] for row in capability_gaps],
        ),
        "",
        "## Language × primary-category matrix",
        "",
        markdown_table(
            ["Language", *categories],
            [[language, *[matrix_data.get(language, {}).get(category, 0) for category in categories]] for language in languages],
        ),
        "",
        "## Decision rule",
        "",
        f"Current automated recommendation: **`{summary['decision']['status']}`**.",
        "",
        summary["decision"]["reason"],
        "",
        "The final gap batch topic must be selected after human review of duplicate flags, context outliers, leakage flags, and language-category coverage. Do not choose it from counts alone.",
        "",
    ]
    return "\n".join(lines)


def audit(root: Path, out_dir: Path, tokenizer_path: Path | None, expected_context: int) -> dict[str, Any]:
    accepted, integrity = load_accepted_records(root)
    legacy, foundation_seed_files = load_legacy_records(root)
    records = [*legacy, *accepted]
    records.sort(key=lambda item: (item.source_kind, item.source_path, item.record_id))
    if not records:
        raise SystemExit("No instruction records found")

    counter = TokenCounter(tokenizer_path)
    lengths = record_length_rows(records, counter)
    duplicates = duplicate_flags(records)
    eval_paths = [root / "data" / "eval" / "code_prompts.jsonl", root / "data" / "eval" / "phase3_5b_heldout_v1.jsonl"]
    leakage = leakage_flags(records, eval_paths)

    language_counts = Counter(record.language for record in records)
    category_counts = Counter(record.category for record in records)
    difficulty_counts = Counter(record.difficulty for record in records)
    source_counts = Counter(record.source_kind for record in records)
    tag_counts: Counter[str] = Counter()
    for record in records:
        tag_counts.update(secondary_tags(record))

    context_thresholds = sorted(set([128, 192, expected_context, 512]))
    context_fit = {
        str(threshold): {
            "fits": sum(1 for row in lengths if row["full_record_tokens"] <= threshold),
            "exceeds": sum(1 for row in lengths if row["full_record_tokens"] > threshold),
        }
        for threshold in context_thresholds
    }
    language_gaps = language_gap_analysis(language_counts, len(records))
    capability_gaps = capability_gap_analysis(records)
    capability_counts = {
        row["capability"]: row["count"]
        for row in capability_gaps
    }
    decision = curriculum_decision(
        len(records),
        duplicates,
        leakage,
        lengths,
        language_gaps,
        expected_context,
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "record_counts": {
            "total_instruction_records": len(records),
            "accepted_phase3_5b_records": source_counts.get("accepted_jsonl", 0),
            "legacy_protocol_records": source_counts.get("legacy_raw", 0),
            "foundation_seed_files": len(foundation_seed_files),
        },
        "language_counts": dict(sorted(language_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "capability_counts": dict(sorted(capability_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "secondary_tag_counts": dict(sorted(tag_counts.items())),
        "language_category_matrix": matrix(records, "language", "category"),
        "language_difficulty_matrix": matrix(records, "language", "difficulty"),
        "length_analysis": {
            "mode": counter.mode,
            "tokenizer_path": counter.path.relative_to(root).as_posix() if counter.path and counter.path.is_relative_to(root) else (str(counter.path) if counter.path else None),
            "full_record_tokens": stats([row["full_record_tokens"] for row in lengths]),
            "instruction_tokens": stats([row["instruction_tokens"] for row in lengths]),
            "answer_tokens": stats([row["answer_tokens"] for row in lengths]),
            "context_fit": context_fit,
            "expected_context": expected_context,
        },
        "supervision": {
            "supervised_tokens": stats([row["supervised_tokens"] for row in lengths]),
            "supervised_percentage": stats([int(round(row["supervised_percentage"] * 1000)) for row in lengths]),
            "supervised_percentage_scale": "stored stats are thousandths of a percentage point",
            "zero_supervised_records": sum(1 for row in lengths if row["supervised_tokens"] == 0),
        },
        "duplicate_analysis": {
            "total_flags": len(duplicates),
            "exact_flags": sum(1 for item in duplicates if item.reason == "exact normalized match"),
            "structural_flags": sum(1 for item in duplicates if item.field == "answer_structure"),
        },
        "leakage_analysis": {
            "critical": sum(1 for item in leakage if item.severity == "critical"),
            "review": sum(1 for item in leakage if item.severity == "review"),
            "informational": sum(1 for item in leakage if item.severity == "informational"),
        },
        "pattern_counts": pattern_counts(records),
        "curriculum_gaps": {"language": language_gaps, "capability": capability_gaps},
        "integrity": {
            "accepted_batch_render_checks": integrity,
            "all_accepted_raw_files_match": all(item["raw_exists"] and item["raw_matches_rendered_jsonl"] for item in integrity),
            "foundation_seed_files": foundation_seed_files,
        },
        "decision": decision,
    }

    # Human-readable percentage statistics use real percentages.
    percentage_values = [row["supervised_percentage"] for row in lengths]
    summary["supervision"]["supervised_percentage"] = stats([int(round(value * 1000)) for value in percentage_values])
    for key in ("min", "median", "mean", "p90", "p95", "max"):
        summary["supervision"]["supervised_percentage"][key] = round(float(summary["supervision"]["supervised_percentage"][key]) / 1000.0, 3)
    summary["supervision"].pop("supervised_percentage_scale", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase3_5b_corpus_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "phase3_5b_corpus_audit.md").write_text(build_markdown(summary), encoding="utf-8", newline="\n")
    (out_dir / "phase3_5b_curriculum_gaps.md").write_text(build_gap_markdown(summary), encoding="utf-8", newline="\n")

    write_csv(
        out_dir / "phase3_5b_near_duplicates.csv",
        ["left_id", "right_id", "field", "score", "reason", "left_source", "right_source"],
        [asdict(item) for item in duplicates],
    )
    write_csv(
        out_dir / "phase3_5b_eval_leakage_flags.csv",
        ["record_id", "eval_id", "severity", "field", "score", "reason"],
        [asdict(item) for item in leakage],
    )
    outlier_rows = [
        row
        for row in lengths
        if row["full_record_tokens"] > expected_context
        or row["answer_tokens"] > max(96, expected_context // 2)
        or row["supervised_percentage"] < 10.0
    ]
    write_csv(
        out_dir / "phase3_5b_length_outliers.csv",
        [
            "record_id", "source_kind", "source_path", "language", "category", "difficulty",
            "instruction_tokens", "constraints_tokens", "bad_code_tokens", "reasoning_tokens", "answer_tokens",
            "full_record_tokens", "supervised_tokens", "supervised_percentage", "characters", "lines",
        ],
        outlier_rows,
    )
    write_csv(
        out_dir / "phase3_5b_record_inventory.csv",
        [
            "record_id", "source_kind", "source_path", "language", "category", "difficulty",
            "instruction_tokens", "answer_tokens", "full_record_tokens", "supervised_tokens", "supervised_percentage",
        ],
        lengths,
    )
    write_csv(
        out_dir / "phase3_5b_human_review_queue.csv",
        [
            "finding_id", "finding_type", "priority", "record_id", "related_id", "field",
            "score", "reason", "source", "reviewer_decision", "action", "notes",
        ],
        human_review_queue(duplicates, leakage, lengths, expected_context, counter.mode),
    )
    return summary


def resolve_tokenizer(root: Path, requested: str | None) -> Path | None:
    if requested:
        path = Path(requested)
        path = path if path.is_absolute() else root / path
        if not path.exists():
            raise SystemExit(f"Tokenizer file does not exist: {path}")
        return path
    for candidate in (root / "data" / "tokenizer.json", root / "tokenizer.json"):
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the admitted NeomaV1 instruction corpus without modifying training data.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports/phase3_5b_current"))
    parser.add_argument("--tokenizer", help="Optional existing tokenizer JSON. This command never trains or rewrites a tokenizer.")
    parser.add_argument("--expected-context", type=int, default=256)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    if args.expected_context <= 0:
        raise SystemExit("--expected-context must be a positive integer")
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Repository root does not exist or is not a directory: {root}")
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    tokenizer_path = resolve_tokenizer(root, args.tokenizer)
    summary = audit(root, out_dir, tokenizer_path, args.expected_context)
    print(f"Instruction records audited: {summary['record_counts']['total_instruction_records']}")
    print(f"Accepted Phase 3.5B records: {summary['record_counts']['accepted_phase3_5b_records']}")
    print(f"Legacy protocol records: {summary['record_counts']['legacy_protocol_records']}")
    print(f"Length mode: {summary['length_analysis']['mode']}")
    print(f"Duplicate flags: {summary['duplicate_analysis']['total_flags']}")
    print(f"Critical leakage flags: {summary['leakage_analysis']['critical']}")
    print(f"Decision: {summary['decision']['status']}")
    print(f"Reports: {out_dir}")


if __name__ == "__main__":
    main()
