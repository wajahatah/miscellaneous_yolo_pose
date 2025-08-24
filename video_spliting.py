import os
import json
import shutil
import pandas as pd
from datetime import datetime

# === CONFIGURATION ===
d = 10
json_path = f"C:/Users/LAMBDA THETA/Downloads/{d}_08_2025_qiyas_multicam.alerts2.json"
csv_path = f"C:/Users/LAMBDA THETA/Downloads/2025-08-{d}_class_results_QIYAS_SYSTEM_2.csv"
video_folder = f"C:/Users/LAMBDA THETA/Downloads/aug_{d}-2"
output_base = f"F:/Wajahat/qiyas_analysis/aug_{d}-2"

# === Load and clean CSV ===
df = pd.read_csv(csv_path,)# sep="\t", encoding="latin1" if not csv_path.endswith(".csv") else "utf-8")

# df.columns = df.columns.str.strip()
# df.fillna("", inplace=True)

# Parse timestamp in CSV to datetime (format: 5/8/2025 8:25)
df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M", errors="coerce")
df["cam_id"] = df["cam_id"].astype(str).str.strip()
df["class_name"] = df["class_name"].astype(str).str.strip().str.title()
df["desk"] = pd.to_numeric(df["desk"], errors="coerce").fillna(-1).astype(int)

# === Load JSON ===
with open(json_path, 'r') as f:
    data = json.load(f)

if isinstance(data, dict):
    data = [data]

# === Process each JSON entry ===
for entry in data:
    file_name = entry.get("file_name", "").strip()
    cam_id = entry.get("cam_id", "").strip()
    alert_ID = entry.get("alert_ID", "").strip()

    alerts = entry.get("alerts", [])
    if not alerts or "title" not in alerts[0]:
        title = "Others"
        desk = -1
    else:
        title = alerts[0]["title"].strip().title() or "Others"
        desk = alerts[0].get("desk", -1)

    # Convert JSON alert_ID to datetime (ignore seconds)
    try:
        alert_dt = datetime.strptime(alert_ID, "%Y-%m-%d %H:%M:%S")
        alert_dt_minute = alert_dt.replace(second=0)
    except:
        alert_dt_minute = None

    # Match on cam_id, desk, title, and timestamp up to minute
    matched_row = df[
        (df["cam_id"] == cam_id) &
        (df["desk"] == desk) &
        (df["class_name"] == title) &
        (df["timestamp_dt"] == alert_dt_minute)
    ]

    # Determine label
    if not matched_row.empty:
        row = matched_row.iloc[0]
        if int(row["TP"]) == 1:
            label = "TP"
        elif int(row["FP"]) == 1:
            label = "FP"
        else:
            label = "Unlabeled"
    else:
        label = "Unmatched"

    # Copy/move video
    src_video = os.path.join(video_folder, f"{file_name}.mp4")
    dest_folder = os.path.join(output_base, title, label)
    os.makedirs(dest_folder, exist_ok=True)
    dest_video = os.path.join(dest_folder, f"{file_name}.mp4")

    if os.path.exists(src_video):
        shutil.copy2(src_video, dest_video)
        print(f"Copied: {src_video} --> {dest_folder}")
    else:
        print(f"[Missing] {src_video}")
