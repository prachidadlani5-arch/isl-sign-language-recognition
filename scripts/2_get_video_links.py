import yt_dlp
import pandas as pd

CHANNEL_URL = "https://www.youtube.com/@isldictionary/videos"

ydl_opts = {
    "quiet": True,
    "extract_flat": True,
    "skip_download": True,
}

videos = []

print("Fetching videos from YouTube...")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(CHANNEL_URL, download=False)

    for video in info["entries"]:
        videos.append({
            "title": video["title"],
            "video_id": video["id"],
            "url": f"https://www.youtube.com/watch?v={video['id']}"
        })

df = pd.DataFrame(videos)

df.to_csv("dataset/metadata/video_links.csv", index=False)

print(f"\nTotal videos found: {len(df)}")
print("Saved to dataset/metadata/video_links.csv")