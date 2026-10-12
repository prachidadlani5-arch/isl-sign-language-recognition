import os
import pandas as pd
from yt_dlp import YoutubeDL

# -----------------------------
# Load matched videos
# -----------------------------
df = pd.read_csv("dataset/metadata/matched_videos.csv")

# -----------------------------
# Create output folder
# -----------------------------
output_folder = "dataset/videos"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# yt-dlp settings
# -----------------------------
ydl_opts = {
    "format": "mp4/best",
    "outtmpl": os.path.join(output_folder, "%(title)s.%(ext)s"),
    "quiet": True,
    "noplaylist": True,
}

# -----------------------------
# Download
# -----------------------------
total = len(df)

for i, row in df.iterrows():

    title = row["word"]
    url = row["url"]

    # Safe filename
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-").strip()

    # Skip if already downloaded
    already = False
    for ext in [".mp4", ".webm", ".mkv"]:
        if os.path.exists(os.path.join(output_folder, safe_title + ext)):
            already = True
            break

    if already:
        print(f"[{i+1}/{total}] Skipped: {title}")
        continue

    print(f"[{i+1}/{total}] Downloading: {title}")

    try:
        ydl_opts["outtmpl"] = os.path.join(output_folder, safe_title + ".%(ext)s")

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        print("✓ Done")

    except Exception as e:
        print(f"✗ Failed: {title}")
        print(e)

print("\n=================================")
print("Dataset download completed!")
print("=================================")