from pathlib import Path
import cv2

# -----------------------------
# Paths
# -----------------------------
VIDEOS_DIR = Path("dataset/videos")
FRAMES_DIR = Path("dataset/frames")

FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Save every nth frame
# -----------------------------
FRAME_SKIP = 2        # Save every 2nd frame

video_files = sorted(VIDEOS_DIR.glob("*.mp4"))

print(f"\nFound {len(video_files)} videos.\n")

for idx, video_path in enumerate(video_files, start=1):

    word = video_path.stem

    save_folder = FRAMES_DIR / word
    save_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_id = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % FRAME_SKIP == 0:

            filename = save_folder / f"frame_{saved:04d}.jpg"

            cv2.imwrite(str(filename), frame)

            saved += 1

        frame_id += 1

    cap.release()

    print(f"[{idx}/{len(video_files)}] {word:<35} Saved {saved} frames")

print("\n✅ Frame extraction completed!")