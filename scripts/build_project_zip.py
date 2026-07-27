#!/usr/bin/env python3
"""Build a clean source-only project ZIP archive.

The script is intentionally conservative:
- tracked files are preferred when the project is a Git repository;
- local outputs, caches, environments, model checkpoints, and previous archives
  are excluded everywhere;
- a manifest is generated inside the archive so the package is self-describing.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


FORBIDDEN_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "ENV",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "node_modules",
    "dist",
    "build",
    "out",
    "target",
    ".gradle",
    ".idea",
    ".vscode",
    "coverage",
    "htmlcov",
    "outputs",
    "hybrid_outputs",
    "web_outputs",
    "checkpoints",
    "models",
    "reports/tmp",
    "laporan_render",
    "laporan_render_final",
    "laporan_reference_render",
    "makalah_render",
    "ref_pdf_render",
}

FORBIDDEN_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
    "local.properties",
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
}

FORBIDDEN_GLOBS = {
    ".github/workflows/*.log",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.log",
    "*.tmp",
    "*.temp",
    "*.bak",
    "*.swp",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.7z",
    "*.rar",
    "*.apk",
    "*.aab",
    "*.class",
    "*.jar",
    "*.war",
    "*.iml",
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

COMMON_INCLUDE_FILES = {
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "tsconfig.json",
    "docker-compose.yml",
    "Dockerfile",
    ".env.example",
    "gradlew",
    "gradlew.bat",
    "settings.gradle",
    "settings.gradle.kts",
    "build.gradle",
    "build.gradle.kts",
    "gradle.properties",
}

COMMON_INCLUDE_DIRS = {
    "app",
    "src",
    "backend",
    "frontend",
    "public",
    "assets",
    "docs",
    "scripts",
    "configs",
    "datasets",
    "tests",
    "test",
    "notebook",
    "notebooks",
    "templates",
    "static",
    "zones",
}

IMPORTANT_PROJECT_FILES = {
    "hybrid_realtime_anomaly_app.py",
    "web_app.py",
    "generate_anomaly_report.py",
    "CCTV_Anomaly_Detection_Realtime.ipynb",
    "Hybrid_CCTV_Anomaly_Detection_App.ipynb",
}


def run_git(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return result.stdout.strip()


def project_name(root: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", root.name).strip("_")
    return name or "Project"


def is_forbidden(path: Path) -> tuple[bool, str]:
    parts = path.parts
    posix = path.as_posix()
    for part in parts:
        if part in FORBIDDEN_DIRS:
            return True, f"directory:{part}"
    if path.name in FORBIDDEN_FILE_NAMES:
        return True, f"file:{path.name}"
    for pattern in FORBIDDEN_GLOBS:
        if fnmatch.fnmatch(posix, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True, f"pattern:{pattern}"
    return False, ""


def git_tracked_files(root: Path) -> set[Path]:
    # Use HEAD instead of the mutable index so unrelated staged files are not
    # accidentally archived. If HEAD is unavailable, fall back to the index.
    output = run_git(["ls-tree", "-r", "--name-only", "HEAD"], root)
    if not output:
        output = run_git(["ls-files"], root)
    if not output:
        return set()
    return {Path(line) for line in output.splitlines() if line.strip()}


def fallback_files(root: Path) -> set[Path]:
    files: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_file():
            files.add(path.relative_to(root))
    return files


def should_include(path: Path, tracked_mode: bool) -> bool:
    if is_forbidden(path)[0]:
        return False
    top = path.parts[0] if path.parts else path.name
    if tracked_mode:
        return True
    if path.as_posix() in COMMON_INCLUDE_FILES or path.as_posix() in IMPORTANT_PROJECT_FILES:
        return True
    if top in COMMON_INCLUDE_DIRS:
        return True
    if path.suffix == ".ipynb":
        return True
    return path.suffix in {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css", ".js"}


def detect_commands(root: Path) -> tuple[list[str], list[str], list[str]]:
    setup: list[str] = []
    run: list[str] = []
    test: list[str] = []
    if (root / "requirements.txt").exists():
        setup.append("pip install -r requirements.txt")
    if (root / "web_app.py").exists():
        run.append("python3 web_app.py")
    if (root / "hybrid_realtime_anomaly_app.py").exists():
        run.append("python3 hybrid_realtime_anomaly_app.py --source 0 --sensitivity high --no-human-tracking")
    if (root / "package.json").exists():
        setup.append("npm install")
        try:
            import json

            payload = json.loads((root / "package.json").read_text(encoding="utf-8"))
            scripts = payload.get("scripts", {})
            if "dev" in scripts:
                run.append("npm run dev")
            if "test" in scripts:
                test.append("npm test")
            if "build" in scripts:
                test.append("npm run build")
        except Exception:
            pass
    if (root / "gradlew").exists():
        test.extend(["./gradlew test", "./gradlew assembleDebug"])
    if any(root.glob("*.ipynb")):
        run.append("Open the included .ipynb notebook files in Jupyter or Colab.")
    test.append("python3 -m compileall .")
    if (root / "tests").exists() or (root / "test").exists():
        test.append("pytest -q")
    return setup, run, test


def write_manifest(
    staging: Path,
    root: Path,
    zip_name: str,
    included: list[Path],
    excluded_summary: dict[str, int],
) -> None:
    setup_commands, run_commands, test_commands = detect_commands(root)
    top_level = sorted({path.parts[0] for path in included if path.parts})
    branch = run_git(["branch", "--show-current"], root) or "unknown"
    commit = run_git(["rev-parse", "--short", "HEAD"], root) or "unknown"
    remote = run_git(["config", "--get", "remote.origin.url"], root) or "unknown"
    lines = [
        "# Project ZIP Manifest",
        "",
        f"Project name: {project_name(root)}",
        f"Generated timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"Git commit hash: {commit}",
        f"Git branch: {branch}",
        f"Repository remote URL: {remote}",
        f"ZIP file name: {zip_name}",
        "",
        "## Included Top-Level Folders/Files",
        "",
        *[f"- `{item}`" for item in top_level],
        "",
        "## Excluded Folders/Files Summary",
        "",
        *[f"- {key}: {value}" for key, value in sorted(excluded_summary.items())],
        "",
        "## How To Set Up",
        "",
        *([f"- `{cmd}`" for cmd in setup_commands] or ["- No setup command was inferred from project files."]),
        "",
        "## How To Run Project",
        "",
        *([f"- `{cmd}`" for cmd in run_commands] or ["- No run command was inferred from project files."]),
        "",
        "## How To Test Project",
        "",
        *[f"- `{cmd}`" for cmd in test_commands],
        "",
        "## Known Local Setup Requirements",
        "",
        "- Python 3 is required for Python scripts.",
        "- Webcam, CCTV footage, or RTSP source is required to run detection workflows.",
        "- Large model checkpoints are intentionally excluded; place optional model files locally when needed.",
        "- Runtime outputs are intentionally excluded and will be regenerated when the application runs.",
        "",
    ]
    (staging / "PROJECT_ZIP_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def copy_file(root: Path, staging: Path, rel_path: Path) -> None:
    source = root / rel_path
    target = staging / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path.cwd().resolve()
    name = project_name(root)
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    zip_path = dist / f"{name}_Project_Source.zip"
    tracked = git_tracked_files(root)
    tracked_mode = bool(tracked)
    candidates = tracked if tracked_mode else fallback_files(root)

    # Include known runtime files even when they are ignored/untracked in a new project.
    for rel in IMPORTANT_PROJECT_FILES | COMMON_INCLUDE_FILES:
        if (root / rel).is_file():
            candidates.add(Path(rel))
    for folder in COMMON_INCLUDE_DIRS:
        folder_path = root / folder
        if folder_path.is_dir():
            for path in folder_path.rglob("*"):
                if path.is_file():
                    candidates.add(path.relative_to(root))

    included: list[Path] = []
    excluded_summary: dict[str, int] = {}
    for rel_path in sorted(candidates, key=lambda p: p.as_posix()):
        forbidden, reason = is_forbidden(rel_path)
        if forbidden or not should_include(rel_path, tracked_mode):
            reason = reason or "not-allowlisted"
            excluded_summary[reason] = excluded_summary.get(reason, 0) + 1
            continue
        if (root / rel_path).is_file():
            included.append(rel_path)

    if not included:
        print("No files matched packaging rules.", file=sys.stderr)
        return 1

    if zip_path.exists():
        zip_path.unlink()

    staging_parent = dist / ".zip_staging"
    staging_parent.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=staging_parent))
    try:
        for rel_path in included:
            copy_file(root, staging, rel_path)

        manifest_rel = Path("PROJECT_ZIP_MANIFEST.md")
        write_manifest(staging, root, zip_path.name, included, excluded_summary)
        all_files = sorted(
            [path for path in staging.rglob("*") if path.is_file()],
            key=lambda p: p.relative_to(staging).as_posix(),
        )
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in all_files:
                archive.write(path, path.relative_to(staging).as_posix())
        included_count = len(all_files)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if staging_parent.exists() and not any(staging_parent.iterdir()):
            staging_parent.rmdir()

    zip_size = zip_path.stat().st_size
    digest = sha256(zip_path)
    print(f"Project name: {name}")
    print(f"Output ZIP path: {zip_path}")
    print(f"ZIP size: {zip_size} bytes")
    print(f"Included file count: {included_count}")
    print("Excluded pattern summary:")
    for reason, count in sorted(excluded_summary.items()):
        print(f"  - {reason}: {count}")
    print(f"SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
