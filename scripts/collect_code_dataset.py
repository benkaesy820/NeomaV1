from __future__ import annotations

import argparse
import fnmatch
import re
from pathlib import Path


DEFAULT_EXTENSIONS = {
    ".bat",
    ".c",
    ".cc",
    ".cfg",
    ".cmd",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".lua",
    ".md",
    ".mjs",
    ".php",
    ".ps1",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".svelte",
    ".toml",
    ".ts",
    ".tsx",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".turbo",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "out",
    "target",
    "vendor",
}

SKIP_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "cargo.lock",
}

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"][^'\"]{16,}['\"]"),
    re.compile(r"\bghp_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
]


def parse_extensions(raw: str) -> set[str]:
    return {item if item.startswith(".") else f".{item}" for item in raw.split(",") if item.strip()}


def should_skip_path(path: Path, root: Path, max_bytes: int) -> str | None:
    relative_parts = path.relative_to(root).parts
    if any(part in SKIP_DIRS for part in relative_parts[:-1]):
        return "skipped directory"
    if path.name.lower() in SKIP_NAMES:
        return "lock/generated file"
    if path.stat().st_size > max_bytes:
        return "too large"
    return None


def looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:4096]


def likely_has_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def matches_ignore(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def iter_code_files(
    roots: list[Path],
    extensions: set[str],
    max_bytes: int,
    ignore: list[str],
) -> tuple[list[Path], dict[str, int]]:
    files: list[Path] = []
    stats = {
        "candidate": 0,
        "bad_extension": 0,
        "ignored": 0,
        "skipped": 0,
    }

    for root in roots:
        root = root.resolve()
        if not root.exists():
            raise SystemExit(f"Input path does not exist: {root}")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            stats["candidate"] += 1
            if path.suffix.lower() not in extensions:
                stats["bad_extension"] += 1
                continue
            if matches_ignore(path, root, ignore):
                stats["ignored"] += 1
                continue
            if should_skip_path(path, root, max_bytes) is not None:
                stats["skipped"] += 1
                continue
            files.append(path)
    return sorted(set(files)), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/raw/code_corpus.txt"))
    parser.add_argument("--extensions", type=str, default=",".join(sorted(DEFAULT_EXTENSIONS)))
    parser.add_argument("--max-file-kb", type=int, default=256)
    parser.add_argument("--ignore", action="append", default=[])
    args = parser.parse_args()

    extensions = parse_extensions(args.extensions)
    max_bytes = args.max_file_kb * 1024
    files, stats = iter_code_files(args.inputs, extensions, max_bytes, args.ignore)

    written = 0
    skipped_binary = 0
    skipped_secret = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as out:
        for path in files:
            data = path.read_bytes()
            if looks_binary(data):
                skipped_binary += 1
                continue
            text = data.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if likely_has_secret(text):
                skipped_secret += 1
                continue
            out.write(f"\n<file path=\"{path.as_posix()}\">\n")
            out.write(text)
            out.write("\n</file>\n")
            written += 1

    print(f"Saved: {args.out}")
    print(f"Files written: {written:,}")
    print(f"Candidates scanned: {stats['candidate']:,}")
    print(f"Skipped by extension: {stats['bad_extension']:,}")
    print(f"Skipped by ignore: {stats['ignored']:,}")
    print(f"Skipped by size/generated dirs: {stats['skipped']:,}")
    print(f"Skipped binary: {skipped_binary:,}")
    print(f"Skipped likely secrets: {skipped_secret:,}")


if __name__ == "__main__":
    main()
