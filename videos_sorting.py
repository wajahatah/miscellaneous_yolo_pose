"""Sort videos using alerts json file in different folders based on the titles"""

import os
import json
import shutil

# === CONFIGURATION ===
json_path = "C:/Users/LAMBDA THETA/Downloads/27_july_qiyas_multicam.alerts.json"          # Path to your JSON file
video_folder = "C:/Users/LAMBDA THETA/Downloads/july_27"          # Folder with video files
output_base = "C:/Users/LAMBDA THETA/Downloads/july_27/fp"         # Output folder to store sorted videos

# === STEP 1: Load the JSON ===
with open(json_path, 'r') as f:
    data = json.load(f)

# If your JSON is a list of entries
if isinstance(data, dict):
    data = [data]  # wrap single object as list if needed

# === STEP 2: Process each entry ===
for entry in data:
    file_name = entry.get("file_name", "")
    alerts = entry.get("alerts", [])

    # Handle empty/missing alerts
    if not alerts or "title" not in alerts[0]:
        title = "Others"
    else:
        title = alerts[0]["title"].strip() or "Others"

    # Construct source and destination paths
    src_video = os.path.join(video_folder, f"{file_name}.mp4")
    dest_folder = os.path.join(output_base, title)
    os.makedirs(dest_folder, exist_ok=True)
    dest_video = os.path.join(dest_folder, f"{file_name}.mp4")

    # Copy if source video exists
    if os.path.exists(src_video):
        shutil.copy2(src_video, dest_video)
        # print(f"Copied {src_video} -> {dest_video}")
    else:
        print(f"[Missing] {src_video}")
