from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from hybrid_realtime_anomaly_app import (
    HybridAnomalyDetector,
    RuntimeConfig,
    append_evidence_log,
    append_log,
    apply_sensitivity_preset,
    build_evidence_record,
    draw_overlay,
    ensure_outputs,
    resize_frame,
    timestamp,
    write_explanation_json,
    write_html_report,
    write_metadata,
)


WORKSPACE = Path(__file__).resolve().parent
DETECTOR_SCRIPT = WORKSPACE / "hybrid_realtime_anomaly_app.py"
WEB_OUTPUT_DIR = WORKSPACE / "web_outputs"
UPLOAD_DIR = WEB_OUTPUT_DIR / "uploads"
JOB_DIR = WEB_OUTPUT_DIR / "jobs"
LIVE_DIR = WEB_OUTPUT_DIR / "live"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
LIVE_SESSIONS: dict[str, dict] = {}


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def ensure_web_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)


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


def sensitivity_config(form_data):
    class Args:
        threshold = float(form_data.get("threshold") or 0.78)
        sensitivity = form_data.get("sensitivity", "high")
        model_threshold_scale = 1.0
        min_anomaly_frames = None
        normal_reset_frames = None

    return apply_sensitivity_preset(Args)


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


def create_live_session(form_data):
    ensure_web_dirs()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_dir = LIVE_DIR / session_id / "outputs"
    sensitivity = sensitivity_config(form_data)
    config = RuntimeConfig(
        source="browser-webcam",
        combined_threshold=sensitivity["threshold"],
        model_threshold_scale=sensitivity["model_threshold_scale"],
        min_anomaly_frames=sensitivity["min_anomaly_frames"],
        normal_reset_frames=sensitivity["normal_reset_frames"],
        use_human_tracking=form_data.get("human_tracking") == "on",
        save_video=False,
        show_window=False,
        output_dir=output_dir,
    )
    anomaly_dir, html_report_dir, json_report_dir, video_dir, log_path, evidence_log_path = ensure_outputs(config)
    metadata_path = write_metadata(config)
    report_path = html_report_dir / f"live_report_{timestamp()}.html"
    write_html_report(report_path, html_report_dir, [])

    LIVE_SESSIONS[session_id] = {
        "id": session_id,
        "created_at": now_iso(),
        "status": "running",
        "config": config,
        "detector": HybridAnomalyDetector(config),
        "anomaly_dir": anomaly_dir,
        "html_report_dir": html_report_dir,
        "json_report_dir": json_report_dir,
        "log_path": log_path,
        "evidence_log_path": evidence_log_path,
        "metadata_path": metadata_path,
        "report_path": report_path,
        "records": [],
        "frames": 0,
        "anomalies": 0,
        "saved": 0,
        "last_result": None,
    }
    return session_id


def decode_data_url_image(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Frame webcam tidak bisa dibaca.")
    return frame


def live_report_url(session: dict):
    relative = session["report_path"].relative_to(session["config"].output_dir)
    return url_for("live_output_file", session_id=session["id"], filename=str(relative))


@app.route("/")
def index():
    ensure_web_dirs()
    return render_template("index.html", jobs=list_jobs())


@app.post("/live")
def start_live_webcam():
    session_id = create_live_session(request.form)
    delay = request.form.get("frame_delay_ms", "700")
    return redirect(url_for("live_detail", session_id=session_id, delay=delay))


@app.route("/live/<session_id>")
def live_detail(session_id: str):
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        abort(404)
    return render_template("live.html", session=session, report_url=live_report_url(session))


@app.post("/api/live/<session_id>/frame")
def live_frame(session_id: str):
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        abort(404)
    if session["status"] != "running":
        return jsonify({"status": session["status"], "message": "Session sudah berhenti."}), 409

    payload = request.get_json(force=True)
    frame = decode_data_url_image(payload.get("image", ""))
    config = session["config"]
    detector = session["detector"]
    frame = resize_frame(frame, config)
    metrics = detector.score(frame)
    annotated = draw_overlay(frame, metrics, fps=0.0)

    session["frames"] += 1
    saved_image_url = None
    if metrics["is_anomaly"]:
        session["anomalies"] += 1
        image_path = session["anomaly_dir"] / f"live_anomaly_frame_{metrics['frame_index']:06d}_{timestamp()}.jpg"
        cv2.imwrite(str(image_path), annotated)
        append_log(session["log_path"], metrics, image_path)
        evidence_record = build_evidence_record(config, detector, metrics, image_path)
        append_evidence_log(session["evidence_log_path"], evidence_record)
        write_explanation_json(session["json_report_dir"], evidence_record)
        session["records"].append(evidence_record)
        write_html_report(session["report_path"], session["html_report_dir"], session["records"])
        session["saved"] += 1
        saved_image_url = url_for(
            "live_output_file",
            session_id=session_id,
            filename=str(image_path.relative_to(config.output_dir)),
        )

    result = {
        "status": "ok",
        "label": "ANOMALY" if metrics["is_anomaly"] else "NORMAL",
        "frame_index": metrics["frame_index"],
        "combined_score": round(metrics["combined_score"], 6),
        "threshold": config.combined_threshold,
        "motion_area_ratio": round(metrics["motion_area_ratio"], 6),
        "flow_mean": round(metrics["flow_mean"], 6),
        "abrupt_score": round(metrics.get("abrupt_score", 0.0), 6),
        "sudden_motion_score": round(metrics.get("sudden_motion_score", 0.0), 6),
        "scene_change_score": round(metrics.get("scene_change_score", 0.0), 6),
        "streak": metrics["raw_anomaly_streak"],
        "reason": metrics.get("reason", "normal"),
        "frames": session["frames"],
        "anomalies": session["anomalies"],
        "saved": session["saved"],
        "report_url": live_report_url(session),
        "saved_image_url": saved_image_url,
    }
    session["last_result"] = result
    return jsonify(result)


@app.post("/api/live/<session_id>/stop")
def stop_live(session_id: str):
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        abort(404)
    session["status"] = "stopped"
    return jsonify({"status": "stopped", "report_url": live_report_url(session)})


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


@app.route("/live_outputs/<session_id>/<path:filename>")
def live_output_file(session_id: str, filename: str):
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        abort(404)
    return send_from_directory(session["config"].output_dir, filename)


if __name__ == "__main__":
    ensure_web_dirs()
    port = int(os.environ.get("CCTV_WEB_PORT", "8090"))
    debug = os.environ.get("CCTV_WEB_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
