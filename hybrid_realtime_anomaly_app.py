"""
Hybrid real-time CCTV anomaly detection app.

This combines:
- the trained UCSD Ped2 autoencoder checkpoint from cctv-video-anomaly-detection-main;
- real-time webcam/video/RTSP processing;
- motion and optical-flow scoring for practical CCTV sensitivity;
- anomaly frame photo export.

Output classes stay simple:
- NORMAL
- ANOMALY

Examples that are intentionally included as ANOMALY:
- vandalism / perusakan fasilitas
- abusive or violent behavior / perkelahian
- theft / pencurian
- restricted-area entry
- abandoned or unusual objects
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = WORKSPACE / "models" / "trained_model.pth"
DEFAULT_OUTPUT_DIR = WORKSPACE / "hybrid_outputs"

ANOMALY_CATEGORY = {
    "label": "ANOMALY",
    "included_events": [
        "vandalism / perusakan fasilitas",
        "abusive or violent behavior / perkelahian",
        "theft / pencurian",
        "restricted-area entry / masuk area terlarang",
        "abandoned or unusual object / benda tertinggal atau asing",
        "sudden abnormal movement / gerakan tidak wajar",
    ],
}


@dataclass
class RuntimeConfig:
    source: str | int = 0
    frame_width: int = 640
    frame_height: int = 360
    warmup_frames: int = 45
    combined_threshold: float = 0.78
    motion_area_threshold: float = 0.045
    flow_threshold: float = 2.8
    model_threshold_scale: float = 1.0
    min_anomaly_frames: int = 8
    normal_reset_frames: int = 5
    use_human_tracking: bool = True
    human_detect_every: int = 5
    human_min_confidence: float = 0.0
    track_max_distance: float = 90.0
    track_max_missing: int = 20
    zone_file: Path | None = None
    zone_loiter_frames: int = 75
    zone_min_ratio: float = 0.15
    save_stride: int = 1
    max_frames: int | None = None
    show_window: bool = True
    save_video: bool = True
    model_path: Path = DEFAULT_MODEL_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    calibration_file: Path | None = None
    calibrated_model_threshold: float | None = None


def default_zone(width: int, height: int):
    """Default alert zone: full frame. User can replace it with a JSON polygon."""
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.int32)


def load_zone_polygon(config: RuntimeConfig):
    if config.zone_file is None:
        return default_zone(config.frame_width, config.frame_height)
    if not config.zone_file.exists():
        raise FileNotFoundError(f"File zona tidak ditemukan: {config.zone_file}")

    payload = json.loads(config.zone_file.read_text(encoding="utf-8"))
    points = payload.get("points", payload)
    if len(points) < 3:
        raise ValueError("Zone polygon minimal harus berisi 3 titik.")
    return np.array(points, dtype=np.int32)


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def box_zone_ratio(box, polygon):
    x, y, w, h = box
    box_area = max(w * h, 1)
    mask = np.zeros((max(y + h + 2, int(polygon[:, 1].max()) + 2), max(x + w + 2, int(polygon[:, 0].max()) + 2)), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, mask.shape[1]), min(y + h, mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    overlap = np.count_nonzero(mask[y1:y2, x1:x2])
    return float(overlap / box_area)


class HumanTracker:
    """Lightweight human detector + centroid tracker inspired by PASS-CCTV tracking logic."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.tracks: dict[int, dict[str, Any]] = {}
        self.next_id = 1
        self.last_detections = []

    def detect_people(self, frame):
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        detections = []
        for rect, weight in zip(rects, weights):
            if float(weight) < self.config.human_min_confidence:
                continue
            x, y, w, h = [int(v) for v in rect]
            if w * h < 900:
                continue
            detections.append((x, y, w, h))
        return self._nms(detections)

    def _nms(self, boxes, threshold=0.35):
        if not boxes:
            return []
        boxes_np = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=np.float32)
        scores = np.array([w * h for _, _, w, h in boxes], dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_np.tolist(),
            scores=scores.tolist(),
            score_threshold=0,
            nms_threshold=threshold,
        )
        if len(indices) == 0:
            return []
        return [boxes[int(i)] for i in np.array(indices).flatten()]

    def update(self, frame, frame_index, zone_polygon):
        if frame_index % max(self.config.human_detect_every, 1) == 0 or not self.last_detections:
            detections = self.detect_people(frame)
            self.last_detections = detections
        else:
            detections = self.last_detections

        assigned_tracks = set()
        assigned_detections = set()

        centers = [(x + w / 2, y + h / 2) for x, y, w, h in detections]
        for det_idx, center in enumerate(centers):
            best_track_id = None
            best_dist = self.config.track_max_distance
            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                tx, ty = track["center"]
                dist = float(np.hypot(center[0] - tx, center[1] - ty))
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id

            if best_track_id is not None:
                self._update_track(best_track_id, detections[det_idx], center, frame_index, zone_polygon)
                assigned_tracks.add(best_track_id)
                assigned_detections.add(det_idx)

        for det_idx, box in enumerate(detections):
            if det_idx in assigned_detections:
                continue
            center = centers[det_idx]
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "id": track_id,
                "box": box,
                "center": center,
                "first_seen": frame_index,
                "last_seen": frame_index,
                "missing": 0,
                "history": [center],
                "zone_frames": 0,
                "zone_ratio": box_zone_ratio(box, zone_polygon),
            }

        for track_id, track in list(self.tracks.items()):
            if track_id not in assigned_tracks and track["last_seen"] != frame_index:
                track["missing"] += 1
                if track["missing"] > self.config.track_max_missing:
                    del self.tracks[track_id]

        return list(self.tracks.values())

    def _update_track(self, track_id, box, center, frame_index, zone_polygon):
        track = self.tracks[track_id]
        track["box"] = box
        track["center"] = center
        track["last_seen"] = frame_index
        track["missing"] = 0
        track["history"].append(center)
        track["history"] = track["history"][-60:]
        ratio = box_zone_ratio(box, zone_polygon)
        track["zone_ratio"] = ratio
        if ratio >= self.config.zone_min_ratio or point_in_polygon(center, zone_polygon):
            track["zone_frames"] += 1
        else:
            track["zone_frames"] = 0

    @staticmethod
    def trajectory_area(track):
        pts = np.array(track.get("history", []), dtype=np.float32)
        if len(pts) < 3:
            return 0.0
        return float(abs(cv2.contourArea(pts.reshape((-1, 1, 2)))))


class ConvolutionalAutoencoder(nn.Module):
    """Autoencoder architecture compatible with the existing trained_model.pth."""

    def __init__(self, input_channels: int = 1, latent_dim: int = 256):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        self.flatten = nn.Flatten()
        self.encode_fc = nn.Linear(4 * 4 * 256, latent_dim)
        self.decode_fc = nn.Linear(latent_dim, 4 * 4 * 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        latent = self.encode_fc(self.flatten(encoded))
        decoded = self.decode_fc(latent).view(batch_size, 256, 4, 4)
        return self.decoder(decoded)


class HybridAnomalyDetector:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.device = self._select_device()
        self.model = None
        self.model_threshold = None
        self.prev_gray = None
        self.frame_count = 0
        self.saved_anomaly_count = 0
        self.raw_anomaly_streak = 0
        self.normal_streak = 0
        self.active_anomaly = False
        self.zone_polygon = load_zone_polygon(config)
        self.human_tracker = HumanTracker(config) if config.use_human_tracking else None

        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=32,
            detectShadows=True,
        )
        self._load_model()

    def _select_device(self):
        if not TORCH_AVAILABLE:
            return None
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        if not TORCH_AVAILABLE:
            print("PyTorch tidak tersedia. Aplikasi memakai motion + optical flow saja.")
            return

        if not self.config.model_path.exists():
            print(f"Model tidak ditemukan: {self.config.model_path}")
            print("Aplikasi memakai motion + optical flow saja.")
            return

        checkpoint = torch.load(self.config.model_path, map_location=self.device, weights_only=False)
        self.model = ConvolutionalAutoencoder(input_channels=1, latent_dim=256).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        base_threshold = self.config.calibrated_model_threshold
        if base_threshold is None:
            base_threshold = checkpoint.get("threshold") or 0.005069
        self.model_threshold = float(base_threshold) * self.config.model_threshold_scale
        print("Model autoencoder loaded:", self.config.model_path)
        print("Device:", self.device)
        print("Model threshold:", f"{self.model_threshold:.6f}")

    def _preprocess_model_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def _autoencoder_score(self, frame):
        if self.model is None or self.model_threshold is None:
            return None, 0.0
        tensor = self._preprocess_model_frame(frame)
        with torch.no_grad():
            reconstruction = self.model(tensor)
            error = torch.mean((tensor - reconstruction) ** 2).item()
        score = min(error / max(self.model_threshold, 1e-8), 2.0)
        return float(error), float(score)

    def score(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg_mask = self.bg.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        motion_area_ratio = float(np.count_nonzero(fg_mask) / fg_mask.size)
        motion_score = min(motion_area_ratio / self.config.motion_area_threshold, 2.0)

        flow_mean = 0.0
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mean = float(np.mean(mag))
        self.prev_gray = gray
        flow_score = min(flow_mean / self.config.flow_threshold, 2.0)

        reconstruction_error, ae_score = self._autoencoder_score(frame)
        tracks = self.human_tracker.update(frame, self.frame_count, self.zone_polygon) if self.human_tracker else []
        zone_tracks = [track for track in tracks if track.get("zone_frames", 0) >= self.config.zone_loiter_frames]
        human_tracking_score = 1.0 if zone_tracks else 0.0
        reasons = []

        if self.frame_count <= self.config.warmup_frames:
            combined_score = 0.0
            raw_is_anomaly = False
        elif self.model is not None:
            combined_score = 0.50 * ae_score + 0.25 * motion_score + 0.10 * flow_score + 0.15 * human_tracking_score
            raw_is_anomaly = combined_score >= self.config.combined_threshold
        else:
            combined_score = 0.55 * motion_score + 0.25 * flow_score + 0.20 * human_tracking_score
            raw_is_anomaly = combined_score >= self.config.combined_threshold

        if zone_tracks:
            raw_is_anomaly = True
            ids = ",".join(str(track["id"]) for track in zone_tracks[:5])
            reasons.append(f"person stayed in alert zone: track_id={ids}")
        if raw_is_anomaly and not reasons:
            reasons.append("persistent abnormal visual/motion pattern")

        if raw_is_anomaly:
            self.raw_anomaly_streak += 1
            self.normal_streak = 0
        else:
            self.normal_streak += 1
            if self.normal_streak >= self.config.normal_reset_frames:
                self.raw_anomaly_streak = 0
                self.active_anomaly = False

        if self.raw_anomaly_streak >= self.config.min_anomaly_frames:
            self.active_anomaly = True

        is_anomaly = self.active_anomaly

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 650:
                continue
            boxes.append(cv2.boundingRect(contour))

        return {
            "frame_index": self.frame_count,
            "is_anomaly": bool(is_anomaly),
            "raw_is_anomaly": bool(raw_is_anomaly),
            "raw_anomaly_streak": int(self.raw_anomaly_streak),
            "warmup": self.frame_count <= self.config.warmup_frames,
            "combined_score": float(combined_score),
            "motion_area_ratio": motion_area_ratio,
            "motion_score": float(motion_score),
            "flow_mean": flow_mean,
            "flow_score": float(flow_score),
            "reconstruction_error": reconstruction_error,
            "autoencoder_score": float(ae_score),
            "boxes": boxes,
            "tracks": tracks,
            "zone_polygon": self.zone_polygon,
            "zone_tracks": zone_tracks,
            "reason": "; ".join(reasons) if reasons else "normal",
        }


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resize_frame(frame, config: RuntimeConfig):
    return cv2.resize(frame, (config.frame_width, config.frame_height), interpolation=cv2.INTER_AREA)


def draw_overlay(frame, metrics, fps):
    output = frame.copy()
    if metrics["warmup"]:
        label = "WARMING UP"
        color = (0, 190, 255)
    elif metrics["is_anomaly"]:
        label = "ANOMALY"
        color = (0, 0, 255)
    else:
        label = "NORMAL"
        color = (0, 180, 0)

    for x, y, w, h in metrics["boxes"]:
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    zone_polygon = metrics.get("zone_polygon")
    if zone_polygon is not None:
        cv2.polylines(output, [zone_polygon], isClosed=True, color=(255, 180, 0), thickness=2)

    for track in metrics.get("tracks", []):
        x, y, w, h = track["box"]
        track_color = (0, 0, 255) if track.get("zone_frames", 0) >= 1 else (255, 200, 0)
        cv2.rectangle(output, (x, y), (x + w, y + h), track_color, 2)
        cv2.putText(
            output,
            f"ID {track['id']} z={track.get('zone_frames', 0)}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            track_color,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (0, 0), (output.shape[1], 112), (20, 20, 20), -1)
    cv2.putText(output, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(
        output,
        f"score={metrics['combined_score']:.2f} streak={metrics['raw_anomaly_streak']} motion={metrics['motion_area_ratio']:.3f} flow={metrics['flow_mean']:.2f} fps={fps:.1f}",
        (18, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    ae_error = metrics["reconstruction_error"]
    ae_text = "ae=not-loaded" if ae_error is None else f"ae_error={ae_error:.6f} ae_score={metrics['autoencoder_score']:.2f}"
    cv2.putText(
        output,
        ae_text,
        (18, 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Kategori anomali: vandalism, abusive/violence, theft, restricted activity",
        (18, output.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if metrics.get("is_anomaly"):
        reason = metrics.get("reason", "anomaly")
        cv2.putText(
            output,
            reason[:90],
            (18, output.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def ensure_outputs(config: RuntimeConfig):
    anomaly_dir = config.output_dir / "anomaly_frames"
    video_dir = config.output_dir / "videos"
    log_path = config.output_dir / "anomaly_log_v2.csv"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "frame_index",
                    "label",
                    "combined_score",
                    "motion_area_ratio",
                    "flow_mean",
                    "reconstruction_error",
                    "human_count",
                    "zone_track_ids",
                    "reason",
                    "image_path",
                    "included_anomaly_examples",
                ]
            )
    return anomaly_dir, video_dir, log_path


def append_log(log_path: Path, metrics: dict[str, Any], image_path: Path):
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                metrics["frame_index"],
                "ANOMALY",
                f"{metrics['combined_score']:.6f}",
                f"{metrics['motion_area_ratio']:.6f}",
                f"{metrics['flow_mean']:.6f}",
                "" if metrics["reconstruction_error"] is None else f"{metrics['reconstruction_error']:.8f}",
                len(metrics.get("tracks", [])),
                "|".join(str(track["id"]) for track in metrics.get("zone_tracks", [])),
                metrics.get("reason", ""),
                str(image_path),
                " | ".join(ANOMALY_CATEGORY["included_events"]),
            ]
        )


def write_metadata(config: RuntimeConfig):
    metadata_path = config.output_dir / "anomaly_definition.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_labels": ["NORMAL", "ANOMALY"],
        "anomaly_category": ANOMALY_CATEGORY,
        "model_path": str(config.model_path),
        "methodology": [
            "autoencoder reconstruction error",
            "motion area and optical flow",
            "human detection and centroid tracking",
            "alert-zone duration rule",
            "persistent anomaly streak filter",
        ],
        "notes": "Vandalism, abusive/violence, and theft are treated as one general ANOMALY category.",
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def apply_calibration(config: RuntimeConfig):
    if config.calibration_file is None:
        return
    if not config.calibration_file.exists():
        raise FileNotFoundError(f"File kalibrasi tidak ditemukan: {config.calibration_file}")

    payload = json.loads(config.calibration_file.read_text(encoding="utf-8"))
    config.calibrated_model_threshold = payload.get("model_threshold", config.calibrated_model_threshold)
    config.motion_area_threshold = payload.get("motion_area_threshold", config.motion_area_threshold)
    config.flow_threshold = payload.get("flow_threshold", config.flow_threshold)
    config.combined_threshold = payload.get("combined_threshold", config.combined_threshold)
    config.warmup_frames = payload.get("warmup_frames", config.warmup_frames)

    print("Kalibrasi dipakai:", config.calibration_file)
    print("  model_threshold:", config.calibrated_model_threshold)
    print("  motion_area_threshold:", config.motion_area_threshold)
    print("  flow_threshold:", config.flow_threshold)
    print("  combined_threshold:", config.combined_threshold)


def percentile(values, pct, default):
    if not values:
        return default
    return float(np.percentile(np.array(values, dtype=np.float32), pct))


def calibrate_normal(config: RuntimeConfig, frames: int = 300, percentile_value: float = 99.5, safety_scale: float = 1.25):
    """Capture normal camera/video frames and create camera-specific thresholds."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka source untuk kalibrasi: {config.source}")

    detector = HybridAnomalyDetector(config)
    ae_errors = []
    motion_ratios = []
    flow_means = []
    combined_scores = []
    frame_index = 0

    print("Mode kalibrasi normal dimulai.")
    print("Pastikan kamera hanya melihat kondisi NORMAL. Jangan ada gerakan mencurigakan dulu.")
    print("Tekan 'q' untuk berhenti lebih cepat.")

    try:
        while frame_index < frames:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            frame = resize_frame(frame, config)
            metrics = detector.score(frame)

            if frame_index > config.warmup_frames:
                if metrics["reconstruction_error"] is not None:
                    ae_errors.append(metrics["reconstruction_error"])
                motion_ratios.append(metrics["motion_area_ratio"])
                flow_means.append(metrics["flow_mean"])
                combined_scores.append(metrics["combined_score"])

            annotated = draw_overlay(frame, metrics, fps=0.0)
            cv2.putText(
                annotated,
                f"CALIBRATING NORMAL {frame_index}/{frames}",
                (18, 134),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            if config.show_window:
                cv2.imshow("Calibrating Normal Baseline", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if config.show_window:
            cv2.destroyAllWindows()

    if not motion_ratios:
        raise RuntimeError("Kalibrasi gagal: frame normal terlalu sedikit setelah warmup.")

    model_threshold = None
    if ae_errors:
        model_threshold = percentile(ae_errors, percentile_value, 0.005069) * safety_scale

    motion_area_threshold = max(
        percentile(motion_ratios, percentile_value, config.motion_area_threshold) * safety_scale,
        0.01,
    )
    flow_threshold = max(
        percentile(flow_means, percentile_value, config.flow_threshold) * safety_scale,
        0.35,
    )
    combined_threshold = max(
        percentile(combined_scores, percentile_value, config.combined_threshold) * safety_scale,
        0.95,
    )

    calibration = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(config.source),
        "normal_frames_used": len(motion_ratios),
        "percentile": percentile_value,
        "safety_scale": safety_scale,
        "model_threshold": model_threshold,
        "motion_area_threshold": motion_area_threshold,
        "flow_threshold": flow_threshold,
        "combined_threshold": combined_threshold,
        "warmup_frames": config.warmup_frames,
        "normal_stats": {
            "ae_error_mean": None if not ae_errors else float(np.mean(ae_errors)),
            "ae_error_p99": None if not ae_errors else percentile(ae_errors, 99, 0.0),
            "motion_area_mean": float(np.mean(motion_ratios)),
            "motion_area_p99": percentile(motion_ratios, 99, 0.0),
            "flow_mean": float(np.mean(flow_means)),
            "flow_p99": percentile(flow_means, 99, 0.0),
            "combined_mean": float(np.mean(combined_scores)),
            "combined_p99": percentile(combined_scores, 99, 0.0),
        },
        "notes": "Gunakan file ini dengan --calibration-file agar webcam diam tidak sering dianggap anomali.",
    }

    out_path = config.output_dir / "normal_calibration.json"
    out_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Kalibrasi selesai.")
    print("File kalibrasi:", out_path)
    print(json.dumps(calibration, indent=2, ensure_ascii=False))
    return out_path


def run(config: RuntimeConfig):
    apply_calibration(config)
    anomaly_dir, video_dir, log_path = ensure_outputs(config)
    metadata_path = write_metadata(config)

    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka source: {config.source}")

    detector = HybridAnomalyDetector(config)
    writer = None
    if config.save_video:
        video_path = video_dir / f"hybrid_result_{timestamp()}.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (config.frame_width, config.frame_height),
        )
        print("Output video:", video_path)

    print("Output foto anomali:", anomaly_dir)
    print("Log CSV:", log_path)
    print("Definisi anomali:", metadata_path)
    print("Konfigurasi deteksi:")
    print("  combined_threshold:", config.combined_threshold)
    print("  model_threshold_scale:", config.model_threshold_scale)
    print("  min_anomaly_frames:", config.min_anomaly_frames)
    print("  normal_reset_frames:", config.normal_reset_frames)
    print("  human_tracking:", config.use_human_tracking)
    print("  zone_loiter_frames:", config.zone_loiter_frames)
    print("  zone_file:", config.zone_file)
    print("Tekan 'q' pada window video untuk berhenti.")

    frame_index = 0
    anomaly_frame_counter = 0
    saved_photo_counter = 0
    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream selesai atau frame tidak terbaca.")
                break

            frame_index += 1
            frame = resize_frame(frame, config)
            metrics = detector.score(frame)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            annotated = draw_overlay(frame, metrics, fps)

            if metrics["is_anomaly"]:
                anomaly_frame_counter += 1
                if anomaly_frame_counter % max(config.save_stride, 1) == 0:
                    image_path = anomaly_dir / f"anomaly_frame_{metrics['frame_index']:06d}_{timestamp()}.jpg"
                    cv2.imwrite(str(image_path), annotated)
                    append_log(log_path, metrics, image_path)
                    saved_photo_counter += 1

            if writer is not None:
                writer.write(annotated)

            if config.show_window:
                cv2.imshow("Hybrid Real-Time CCTV Anomaly Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if config.max_frames is not None and frame_index >= config.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if config.show_window:
            cv2.destroyAllWindows()

    print("Selesai.")
    print("Total frame diproses:", frame_index)
    print("Total frame terdeteksi anomali:", anomaly_frame_counter)
    print("Total foto anomali tersimpan:", saved_photo_counter)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Hybrid real-time CCTV anomaly detection")
    parser.add_argument("--source", default="0", help="0 untuk webcam, path video, atau URL RTSP")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path checkpoint trained_model.pth")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Folder output foto/log/video")
    parser.add_argument("--threshold", type=float, default=0.78, help="Ambang keputusan gabungan")
    parser.add_argument(
        "--sensitivity",
        choices=["very_low", "low", "medium", "high"],
        default="medium",
        help="very_low paling ketat, low seimbang, high lebih sensitif",
    )
    parser.add_argument("--model-threshold-scale", type=float, default=1.0, help="Skala threshold autoencoder")
    parser.add_argument("--calibration-file", default=None, help="Path normal_calibration.json hasil --calibrate")
    parser.add_argument("--calibrate", action="store_true", help="Rekam kondisi normal dan buat threshold khusus kamera")
    parser.add_argument("--calibrate-frames", type=int, default=300, help="Jumlah frame normal untuk kalibrasi")
    parser.add_argument("--calibrate-percentile", type=float, default=99.5, help="Percentile threshold normal")
    parser.add_argument("--safety-scale", type=float, default=1.25, help="Pengali agar threshold tidak terlalu sensitif")
    parser.add_argument("--save-stride", type=int, default=1, help="Simpan setiap N frame anomali")
    parser.add_argument("--min-anomaly-frames", type=int, default=None, help="Jumlah frame mencurigakan berturut-turut sebelum ANOMALY aktif")
    parser.add_argument("--normal-reset-frames", type=int, default=None, help="Jumlah frame normal untuk reset status ANOMALY")
    parser.add_argument("--no-human-tracking", action="store_true", help="Matikan deteksi/tracking manusia")
    parser.add_argument("--human-detect-every", type=int, default=5, help="Deteksi manusia setiap N frame")
    parser.add_argument("--zone-file", default=None, help="JSON polygon zona alert, contoh: {\"points\": [[x,y], ...]}")
    parser.add_argument("--zone-loiter-frames", type=int, default=75, help="Frame orang berada di zona sebelum dianggap anomali")
    parser.add_argument("--zone-min-ratio", type=float, default=0.15, help="Minimal overlap bbox manusia dengan zona")
    parser.add_argument("--max-frames", type=int, default=None, help="Batas frame untuk diproses")
    parser.add_argument("--no-window", action="store_true", help="Jangan tampilkan window OpenCV")
    parser.add_argument("--no-video", action="store_true", help="Jangan simpan video hasil")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser


def apply_sensitivity_preset(args):
    presets = {
        "very_low": {
            "threshold": max(args.threshold, 1.35),
            "model_threshold_scale": max(args.model_threshold_scale, 1.8),
            "min_anomaly_frames": 14 if args.min_anomaly_frames is None else args.min_anomaly_frames,
            "normal_reset_frames": 10 if args.normal_reset_frames is None else args.normal_reset_frames,
        },
        "low": {
            "threshold": max(args.threshold, 1.05),
            "model_threshold_scale": max(args.model_threshold_scale, 1.35),
            "min_anomaly_frames": 8 if args.min_anomaly_frames is None else args.min_anomaly_frames,
            "normal_reset_frames": 8 if args.normal_reset_frames is None else args.normal_reset_frames,
        },
        "medium": {
            "threshold": args.threshold,
            "model_threshold_scale": args.model_threshold_scale,
            "min_anomaly_frames": 8 if args.min_anomaly_frames is None else args.min_anomaly_frames,
            "normal_reset_frames": 5 if args.normal_reset_frames is None else args.normal_reset_frames,
        },
        "high": {
            "threshold": min(args.threshold, 0.10),
            "model_threshold_scale": min(args.model_threshold_scale, 0.85),
            "min_anomaly_frames": 2 if args.min_anomaly_frames is None else args.min_anomaly_frames,
            "normal_reset_frames": 3 if args.normal_reset_frames is None else args.normal_reset_frames,
        },
    }
    return presets[args.sensitivity]


def main():
    args = build_arg_parser().parse_args()
    sensitivity = apply_sensitivity_preset(args)
    config = RuntimeConfig(
        source=parse_source(args.source),
        frame_width=args.width,
        frame_height=args.height,
        combined_threshold=sensitivity["threshold"],
        model_threshold_scale=sensitivity["model_threshold_scale"],
        min_anomaly_frames=sensitivity["min_anomaly_frames"],
        normal_reset_frames=sensitivity["normal_reset_frames"],
        use_human_tracking=not args.no_human_tracking,
        human_detect_every=args.human_detect_every,
        zone_file=None if args.zone_file is None else Path(args.zone_file),
        zone_loiter_frames=args.zone_loiter_frames,
        zone_min_ratio=args.zone_min_ratio,
        save_stride=args.save_stride,
        max_frames=args.max_frames,
        show_window=not args.no_window,
        save_video=not args.no_video,
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        calibration_file=None if args.calibration_file is None else Path(args.calibration_file),
    )
    if args.calibrate:
        calibrate_normal(
            config,
            frames=args.calibrate_frames,
            percentile_value=args.calibrate_percentile,
            safety_scale=args.safety_scale,
        )
    else:
        run(config)


if __name__ == "__main__":
    main()
