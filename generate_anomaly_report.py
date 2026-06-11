"""
Generate an HTML anomaly report from anomaly_evidence_log.csv.

Use this when you already have detection outputs and want a readable report
containing each anomaly photo plus its explanation.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
from datetime import datetime
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = WORKSPACE / "hybrid_outputs"
DEFAULT_CSV_PATH = DEFAULT_OUTPUT_DIR / "anomaly_evidence_log.csv"
DEFAULT_ENHANCED_CSV_PATH = DEFAULT_OUTPUT_DIR / "anomaly_evidence_log_enhanced.csv"
DEFAULT_ABRUPT_CSV_PATH = DEFAULT_OUTPUT_DIR / "anomaly_evidence_log_abrupt.csv"
DEFAULT_REPORT_DIR = DEFAULT_OUTPUT_DIR / "anomaly_reports" / "html"


def read_records(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def default_csv_path():
    if DEFAULT_ABRUPT_CSV_PATH.exists():
        return DEFAULT_ABRUPT_CSV_PATH
    if DEFAULT_ENHANCED_CSV_PATH.exists():
        return DEFAULT_ENHANCED_CSV_PATH
    return DEFAULT_CSV_PATH


def row_value(record: dict[str, str], key: str, default: str = "-"):
    value = record.get(key, "")
    return value if value not in ("", None) else default


def write_report(records: list[dict[str, str]], report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cards = []

    for record in records:
        image_path = row_value(record, "image_path", "")
        image_src = os.path.relpath(image_path, start=report_path.parent) if image_path else ""
        details = [
            ("Confidence", row_value(record, "confidence_level")),
            ("Severity", row_value(record, "severity")),
            ("Possible event", row_value(record, "possible_event")),
            ("Duration", f"{row_value(record, 'anomaly_duration_seconds')} sec / {row_value(record, 'anomaly_duration_frames')} frames"),
            ("Combined score", f"{row_value(record, 'combined_score')} / threshold {row_value(record, 'combined_threshold')}"),
            ("Score ratio", f"{row_value(record, 'score_vs_threshold')}x threshold"),
            ("Motion", f"area {row_value(record, 'motion_area_ratio')}, score {row_value(record, 'motion_score')}, {row_value(record, 'motion_vs_threshold')}x threshold"),
            ("Optical flow", f"mean {row_value(record, 'flow_mean')}, score {row_value(record, 'flow_score')}, {row_value(record, 'flow_vs_threshold')}x threshold"),
            ("Sudden motion", f"score {row_value(record, 'sudden_motion_score')}, delta {row_value(record, 'motion_delta')}, baseline {row_value(record, 'motion_baseline')}"),
            ("Flow spike", f"score {row_value(record, 'flow_spike_score')}, delta {row_value(record, 'flow_delta')}, baseline {row_value(record, 'flow_baseline')}"),
            ("Scene change", f"score {row_value(record, 'scene_change_score')}, frame diff {row_value(record, 'frame_diff_mean')}"),
            ("Abrupt score", row_value(record, "abrupt_score")),
            ("Autoencoder", row_value(record, "reconstruction_error", "model tidak tersedia")),
            ("Human/zone", f"human_count={row_value(record, 'human_count')}, zone_track_ids={row_value(record, 'zone_track_ids')}"),
            ("Motion boxes", row_value(record, "detected_motion_boxes")),
            ("Largest motion box", row_value(record, "largest_motion_box")),
            ("Dominant region", row_value(record, "dominant_motion_region")),
        ]
        detail_html = "\n".join(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>" for label, value in details
        )
        cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(image_src)}" alt="Frame anomali {html.escape(row_value(record, 'frame_index'))}">
              <section class="content">
                <div class="meta">Frame {html.escape(row_value(record, 'frame_index'))} - {html.escape(row_value(record, 'timestamp'))}</div>
                <h2>{html.escape(row_value(record, 'label', 'ANOMALY'))}</h2>
                <p class="reason">{html.escape(row_value(record, 'reason'))}</p>
                <p class="explanation">{html.escape(row_value(record, 'natural_language_explanation'))}</p>
                <p>{html.escape(row_value(record, 'evidence_summary'))}</p>
                <table>{detail_html}</table>
              </section>
            </article>
            """
        )

    body = "\n".join(cards) if cards else "<p>Belum ada data anomali.</p>"
    document = f"""<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Laporan Deteksi Anomali CCTV</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7fb; color: #17202a; }}
    header {{ padding: 24px 32px; background: #152238; color: white; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    main {{ padding: 24px; display: grid; gap: 18px; }}
    .card {{ display: grid; grid-template-columns: minmax(260px, 480px) 1fr; gap: 18px; background: white; border: 1px solid #dde3ee; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; height: 100%; min-height: 260px; object-fit: cover; background: #101820; }}
    .content {{ padding: 18px; }}
    .meta {{ color: #637083; font-size: 13px; margin-bottom: 6px; }}
    h2 {{ margin: 0 0 10px; font-size: 22px; color: #b42318; }}
    p {{ line-height: 1.5; }}
    .reason {{ font-weight: 700; }}
    .explanation {{ background: #f1f5fb; border-left: 4px solid #2f6fed; padding: 10px 12px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 14px; }}
    th, td {{ border-top: 1px solid #e5eaf1; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ width: 160px; color: #465466; }}
    @media (max-width: 780px) {{ .card {{ grid-template-columns: 1fr; }} header, main {{ padding: 16px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Laporan Deteksi Anomali CCTV</h1>
    <div>Total foto anomali: {len(records)}</div>
  </header>
  <main>
    {body}
  </main>
</body>
</html>
"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from anomaly evidence CSV")
    parser.add_argument("--csv", default=str(default_csv_path()), help="Path anomaly_evidence_log.csv")
    parser.add_argument("--output", default=None, help="Path output HTML")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output = Path(args.output) if args.output else DEFAULT_REPORT_DIR / f"anomaly_report_from_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    records = read_records(csv_path)
    report_path = write_report(records, output)
    print("Laporan HTML dibuat:", report_path)


if __name__ == "__main__":
    main()
