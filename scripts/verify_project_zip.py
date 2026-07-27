#!/usr/bin/env python3
"""Verify the clean source-only project ZIP archive."""

from __future__ import annotations

import fnmatch
import re
import sys
import zipfile
from pathlib import Path


FORBIDDEN_FOLDERS = {
    ".git/",
    ".venv/",
    "venv/",
    "env/",
    "ENV/",
    "node_modules/",
    "build/",
    "dist/",
    ".gradle/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".ipynb_checkpoints/",
    "outputs/",
    "hybrid_outputs/",
    "web_outputs/",
    "checkpoints/",
    "models/",
}

FORBIDDEN_ARCHIVE_PATTERNS = {
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.7z",
    "*.rar",
}

FORBIDDEN_MODEL_PATTERNS = {
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.gguf",
    "*.ckpt",
    "*.h5",
    "*.keras",
}


def project_name(root: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", root.name).strip("_")
    return name or "Project"


def zip_path(root: Path) -> Path:
    return root / "dist" / f"{project_name(root)}_Project_Source.zip"


def has_forbidden_folder(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    for folder in FORBIDDEN_FOLDERS:
        if normalized.startswith(folder) or f"/{folder}" in normalized:
            return folder
    return None


def matches_any(name: str, patterns: set[str]) -> str | None:
    normalized = name.replace("\\", "/")
    base = Path(normalized).name
    for pattern in patterns:
        if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(base, pattern):
            return pattern
    return None


def verify(root: Path) -> int:
    archive_path = zip_path(root)
    errors: list[str] = []
    warnings: list[str] = []

    if not archive_path.exists():
        print(f"FAIL: ZIP not found: {archive_path}")
        return 1

    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            bad_zip = archive.testzip()
    except zipfile.BadZipFile:
        print(f"FAIL: Invalid ZIP file: {archive_path}")
        return 1

    if bad_zip:
        errors.append(f"corrupt file entry: {bad_zip}")

    name_set = set(names)
    for name in names:
        folder = has_forbidden_folder(name)
        if folder:
            errors.append(f"forbidden folder included: {folder} via {name}")
        archive_pattern = matches_any(name, FORBIDDEN_ARCHIVE_PATTERNS)
        if archive_pattern:
            errors.append(f"archive file included: {name} matches {archive_pattern}")
        model_pattern = matches_any(name, FORBIDDEN_MODEL_PATTERNS)
        if model_pattern:
            errors.append(f"large model/checkpoint included: {name} matches {model_pattern}")

    if "PROJECT_ZIP_MANIFEST.md" not in name_set:
        errors.append("PROJECT_ZIP_MANIFEST.md missing")

    if (root / "README.md").exists() and "README.md" not in name_set:
        errors.append("README.md exists in project but is missing from ZIP")

    for folder in ("docs", "src", "app", "backend", "frontend"):
        if (root / folder).exists() and not any(name.startswith(f"{folder}/") for name in names):
            errors.append(f"{folder}/ exists in project but is missing from ZIP")

    has_source = any(
        name.endswith((".py", ".ipynb", ".js", ".ts", ".tsx", ".jsx", ".html", ".css"))
        for name in names
    )
    if not has_source:
        errors.append("no source files detected in ZIP")

    if not any(name.endswith(".md") for name in names):
        warnings.append("no Markdown documentation found in ZIP")

    print(f"ZIP path: {archive_path}")
    print(f"File count: {len(names)}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("PASS: clean source ZIP verification succeeded")
    return 0


def main() -> int:
    return verify(Path.cwd().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
