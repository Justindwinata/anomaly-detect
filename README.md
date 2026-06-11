# CCTV Anomaly Detection

Program deteksi anomali CCTV untuk tugas Computer Vision. Output utama adalah label `NORMAL` atau `ANOMALY`, dengan contoh kejadian anomali seperti pencurian, vandalism, abusive/violence, aktivitas mencurigakan, dan objek tidak wajar.

## Isi Utama

- `hybrid_realtime_anomaly_app.py` - aplikasi utama untuk webcam, video CCTV, atau RTSP.
- `generate_anomaly_report.py` - membuat laporan HTML berisi foto anomali dan penjelasannya.
- `Hybrid_CCTV_Anomaly_Detection_App.ipynb` - notebook ringkas untuk menjalankan aplikasi hybrid.
- `CCTV_Anomaly_Detection_Realtime.ipynb` - notebook eksperimen awal/end-to-end.
- `zones/alert_zone_example.json` - contoh polygon zona pengawasan.
- `.gitignore` - mengabaikan output, video lokal, cache, model besar, dan file sementara.

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Dengan Webcam

```bash
python3 hybrid_realtime_anomaly_app.py --source 0 --sensitivity high --no-human-tracking
```

Tekan `q` pada window video untuk berhenti.

## Menjalankan Dengan Video CCTV

Letakkan video lokal di folder `data/`, misalnya `data/cctv.mp4`, lalu jalankan:

```bash
python3 hybrid_realtime_anomaly_app.py --source data/cctv.mp4 --sensitivity high --no-human-tracking
```

Jika tidak ingin membuka window:

```bash
python3 hybrid_realtime_anomaly_app.py --source data/cctv.mp4 --sensitivity high --no-human-tracking --no-window
```

## Output

Hasil runtime otomatis disimpan secara lokal dan tidak ikut masuk GitHub:

- `hybrid_outputs/anomaly_frames/`
- `hybrid_outputs/anomaly_reports/html/`
- `hybrid_outputs/anomaly_reports/json/`
- `hybrid_outputs/anomaly_log_v2.csv`
- `hybrid_outputs/anomaly_evidence_log.csv`
- `hybrid_outputs/videos/`
- `outputs/`

`anomaly_evidence_log.csv` adalah log detail untuk laporan. Isinya mencakup nama/path foto anomali, link foto, alasan deteksi, score gabungan, threshold, motion score, optical-flow score, autoencoder score jika model tersedia, jumlah track manusia, dan contoh kategori anomali seperti pencurian, vandalism, serta abusive/violence.

Jika folder output masih memiliki CSV lama dengan format kolom lama, program akan menulis log detail baru ke `anomaly_evidence_log_enhanced.csv` agar data lama tidak rusak.

Selain CSV, program utama juga otomatis membuat laporan HTML di `hybrid_outputs/anomaly_reports/html/`. Laporan ini berisi kartu per foto anomali: gambar, alasan, confidence level, severity, kemungkinan kejadian, durasi anomali, region gerakan dominan, skor deteksi, dan penjelasan otomatis berbentuk kalimat. File JSON detail per foto disimpan terpisah di `hybrid_outputs/anomaly_reports/json/`.

Jika ingin membuat ulang laporan HTML dari CSV yang sudah ada:

```bash
python3 generate_anomaly_report.py
```

## Catatan Model

Jika tersedia model autoencoder PyTorch, letakkan di:

```text
models/trained_model.pth
```

Jika model tidak tersedia, program tetap berjalan dengan mode motion dan optical flow.
