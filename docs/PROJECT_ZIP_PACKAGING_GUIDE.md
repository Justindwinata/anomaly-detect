# Project ZIP Packaging Guide

This guide explains how to create a clean source-only ZIP archive for this
project.

## Purpose

The ZIP archive is intended for project backup, submission, portfolio review, or
sharing the source code with another developer. It should contain the files
needed to reopen, inspect, run, test, and continue development of the CCTV
Anomaly Detection project.

The ZIP is not a runtime output bundle. It does not include generated reports,
large model checkpoints, Git history, local cache, installed dependencies, or
previous archive files.

## What Is Included

The packaging script includes useful project source files and documentation,
including:

- `README.md`
- `requirements.txt`
- `.gitignore`
- `hybrid_realtime_anomaly_app.py`
- `web_app.py`
- `generate_anomaly_report.py`
- project notebooks tracked by Git
- `templates/`
- `static/`
- `zones/`
- `docs/`
- `scripts/`
- `PROJECT_ZIP_MANIFEST.md` generated inside the ZIP

The archive is built from Git-tracked project files plus allowlisted project
source/documentation folders. This keeps local untracked reports, renders, and
temporary files out of the archive.

## What Is Excluded

The ZIP excludes files and folders that are local, generated, heavy, or not
needed for source review:

- `.git/` and Git history
- virtual environments such as `.venv/`, `venv/`, `env/`
- Python cache such as `__pycache__/`, `.pytest_cache/`
- Node/Gradle dependency and build folders such as `node_modules/`, `build/`,
  `dist/`, `.gradle/`
- runtime outputs such as `outputs/`, `hybrid_outputs/`, `web_outputs/`
- generated reports and temporary render folders
- previous archives such as `*.zip`, `*.tar`, `*.rar`
- large AI/model artifacts such as `models/`, `*.pt`, `*.pth`, `*.onnx`,
  `*.safetensors`, `*.bin`, `*.ckpt`
- OS junk such as `.DS_Store` and `Thumbs.db`
- local secret files such as `.env`

These exclusions keep the archive portable, small, and safe to submit.

## Generate The ZIP

Run this command from the project root:

```bash
python3 scripts/build_project_zip.py
```

The generated ZIP is saved at:

```text
dist/CCTV_ANOMALY_DETECT_Project_Source.zip
```

The script prints:

- project name
- ZIP path
- ZIP size
- included file count
- excluded pattern summary
- SHA256 hash

## Verify The ZIP

Run:

```bash
python3 scripts/verify_project_zip.py
```

The verifier opens the ZIP and checks that it does not contain forbidden
folders, previous archive files, large model/checkpoint files, or missing
required source/documentation indicators.

Expected result:

```text
PASS: clean source ZIP verification succeeded
```

## How To Run After Extracting

After extracting the ZIP on another machine:

```bash
pip install -r requirements.txt
python3 web_app.py
```

Open:

```text
http://127.0.0.1:8090
```

For command-line detection:

```bash
python3 hybrid_realtime_anomaly_app.py --source 0 --sensitivity high --no-human-tracking
```

To test a CCTV video, place the video file locally under `data/` after
extracting the ZIP, then use its path as `--source`.

## How To Test Before Packaging

Recommended validation commands:

```bash
git diff --check
git status --short --branch
python3 -m compileall .
python3 scripts/build_project_zip.py
python3 scripts/verify_project_zip.py
```

Run `pytest -q` only when a `tests/` or `test/` folder exists.

## Why Generated ZIP Files Are Not Committed

Generated ZIP files are local release artifacts. They are intentionally ignored
with:

```text
dist/*.zip
```

Commit the scripts and documentation, not the generated ZIP. Regenerate the ZIP
whenever a fresh archive is needed.
