from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename


WORKSPACE = Path(__file__).resolve().parent
DETECTOR_SCRIPT = WORKSPACE / "hybrid_realtime_anomaly_app.py"
WEB_OUTPUT_DIR = WORKSPACE / "web_outputs"
UPLOAD_DIR = WEB_OUTPUT_DIR / "uploads"
JOB_DIR = WEB_OUTPUT_DIR / "jobs"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def ensure_web_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)


def job_path(job_id: str):
    return JOB_DIR / job_id


def metadata_path(job_id: str):
    return job_path(job_id) / "job.json"


def read_job(job_id: str):
    path = metadata_path(job_id)
    if not path.exists():
        abort(404)
    return json.loads(path.read_text(encoding="utf-8"))


def write_job(job: dict):
    path = metadata_path(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")


def list_jobs():
    jobs = []
    if not JOB_DIR.exists():
        return jobs
    for path in JOB_DIR.glob("*/job.json"):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)


def latest_report_url(job: dict):
    output_dir = Path(job["output_dir"])
    report_dir = output_dir / "anomaly_reports" / "html"
    reports = sorted(report_dir.glob("*.html"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not reports:
        return None
    relative = reports[0].relative_to(output_dir)
    return url_for("job_output_file", job_id=job["id"], filename=str(relative))


def enhanced_log_url(job: dict):
    output_dir = Path(job["output_dir"])
    for name in ("anomaly_evidence_log_enhanced.csv", "anomaly_evidence_log.csv"):
        path = output_dir / name
        if path.exists():
            return url_for("job_output_file", job_id=job["id"], filename=name)
    return None


def build_detector_command(source: str, output_dir: Path, form_data):
    sensitivity = form_data.get("sensitivity", "high")
    threshold = form_data.get("threshold", "").strip()
    max_frames = form_data.get("max_frames", "").strip()

    command = [
        sys.executable,
        str(DETECTOR_SCRIPT),
        "--source",
        source,
        "--sensitivity",
        sensitivity,
        "--output-dir",
        str(output_dir),
        "--no-window",
    ]
    if form_data.get("save_video") != "on":
        command.append("--no-video")
    if form_data.get("human_tracking") != "on":
        command.append("--no-human-tracking")
    if threshold:
        command.extend(["--threshold", threshold])
    if max_frames:
        command.extend(["--max-frames", max_frames])
    return command


def run_detection_job(job_id: str, command: list[str]):
    job = read_job(job_id)
    job["status"] = "running"
    job["started_at"] = now_iso()
    job["command"] = command
    write_job(job)

    try:
        result = subprocess.run(
            command,
            cwd=str(WORKSPACE),
            text=True,
            capture_output=True,
            check=False,
        )
        job = read_job(job_id)
        job["finished_at"] = now_iso()
        job["returncode"] = result.returncode
        job["stdout"] = result.stdout[-12000:]
        job["stderr"] = result.stderr[-12000:]
        job["status"] = "finished" if result.returncode == 0 else "failed"
        write_job(job)
    except Exception as exc:
        job = read_job(job_id)
        job["finished_at"] = now_iso()
        job["status"] = "failed"
        job["stderr"] = str(exc)
        write_job(job)


def create_job(name: str, source: str, job_type: str, form_data):
    ensure_web_dirs()
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_dir = job_path(job_id) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_detector_command(source, output_dir, form_data)
    job = {
        "id": job_id,
        "name": name,
        "type": job_type,
        "source": source,
        "status": "queued",
        "created_at": now_iso(),
        "output_dir": str(output_dir),
        "stdout": "",
        "stderr": "",
        "command": command,
    }
    write_job(job)

    thread = threading.Thread(target=run_detection_job, args=(job_id, command), daemon=True)
    thread.start()
    return job_id


@app.route("/")
def index():
    ensure_web_dirs()
    return render_template("index.html", jobs=list_jobs())


@app.post("/upload")
def upload_video():
    file = request.files.get("video")
    if not file or not file.filename:
        return redirect(url_for("index"))

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return redirect(url_for("index"))

    upload_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    filename = f"{upload_id}_{secure_filename(file.filename)}"
    upload_path = UPLOAD_DIR / filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(upload_path)

    job_id = create_job(file.filename, str(upload_path), "upload", request.form)
    return redirect(url_for("job_detail", job_id=job_id))


@app.post("/realtime")
def start_realtime():
    source = request.form.get("source", "0").strip() or "0"
    name = f"Realtime source {source}"
    job_id = create_job(name, source, "realtime", request.form)
    return redirect(url_for("job_detail", job_id=job_id))


@app.route("/jobs/<job_id>")
def job_detail(job_id: str):
    job = read_job(job_id)
    return render_template(
        "job.html",
        job=job,
        report_url=latest_report_url(job),
        log_url=enhanced_log_url(job),
    )


@app.route("/outputs/<job_id>/<path:filename>")
def job_output_file(job_id: str, filename: str):
    job = read_job(job_id)
    output_dir = Path(job["output_dir"])
    return send_from_directory(output_dir, filename)


if __name__ == "__main__":
    ensure_web_dirs()
    port = int(os.environ.get("CCTV_WEB_PORT", "8090"))
    debug = os.environ.get("CCTV_WEB_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
