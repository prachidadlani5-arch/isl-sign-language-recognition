from pathlib import Path
import pandas as pd

FRAMES_DIR = Path("dataset/frames")
LANDMARKS_FILE = Path("dataset/landmarks.csv")

print("=" * 60)
print("SCRIPT 11 - INSPECT VIDEO / LANDMARK STRUCTURE")
print("=" * 60)

# ------------------------------------------------------------
# 1. Check landmark CSV
# ------------------------------------------------------------

print("\nLoading landmarks.csv...")

df = pd.read_csv(LANDMARKS_FILE)

print("✓ landmarks.csv loaded")

print("\nLandmarks:")
print("Rows       :", len(df))
print("Columns    :", len(df))
print("Words      :", df["label"].nunique())

# ------------------------------------------------------------
# 2. Inspect frame folders
# ------------------------------------------------------------

print("\nInspecting dataset/frames...")

if not FRAMES_DIR.exists():
    print("ERROR: dataset/frames does not exist.")
    raise SystemExit

folders = sorted([
    folder for folder in FRAMES_DIR.iterdir()
    if folder.is_dir()
])

print("Word folders:", len(folders))

# ------------------------------------------------------------
# 3. Count images
# ------------------------------------------------------------

total_images = 0
folder_counts = []

for folder in folders:

    images = list(folder.glob("*.jpg"))

    count = len(images)

    total_images += count

    folder_counts.append(
        (folder.name, count)
    )

print("Total JPG frames:", total_images)

# ------------------------------------------------------------
# 4. Compare frames vs landmarks
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print("Frames on disk       :", total_images)
print("Landmark CSV rows    :", len(df))

if total_images == len(df):

    print("✓ Counts match exactly")

else:

    print("⚠ Counts DO NOT match")

    print(
        "Difference:",
        abs(total_images - len(df))
    )

# ------------------------------------------------------------
# 5. Show sample folders
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE WORD FOLDERS")
print("=" * 60)

for word, count in folder_counts[:20]:

    print(f"{word:<30} {count} frames")

# ------------------------------------------------------------
# 6. Show largest folders
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LARGEST WORD FOLDERS")
print("=" * 60)

largest = sorted(
    folder_counts,
    key=lambda x: x[1],
    reverse=True
)[:20]

for word, count in largest:

    print(f"{word:<30} {count} frames")

# ------------------------------------------------------------
# 7. Inspect actual filenames
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE FILENAMES")
print("=" * 60)

shown = 0

for folder in folders:

    images = sorted(folder.glob("*.jpg"))

    if not images:
        continue

    print("\nWORD:", folder.name)

    for image in images[:10]:

        print(" ", image.name)

    shown += 1

    if shown >= 5:
        break

# ------------------------------------------------------------
# 8. Check label counts in landmarks.csv
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LANDMARK SAMPLE COUNTS")
print("=" * 60)

counts = df["label"].value_counts()

print("\nSmallest:")
print(counts.nsmallest(10))

print("\nLargest:")
print(counts.nlargest(10))

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)