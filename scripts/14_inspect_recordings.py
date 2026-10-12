import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SCRIPT 14 - INSPECT RECORDINGS AND TEMPORAL WINDOWS
# ============================================================

LANDMARK_FILE = Path("dataset/landmarks.csv")
FRAMES_DIR = Path("dataset/frames")

WINDOW_SIZE = 64
STRIDE = 16

# ============================================================
# START
# ============================================================

print("=" * 60)
print("SCRIPT 14 - INSPECT RECORDINGS AND TEMPORAL WINDOWS")
print("=" * 60)

print("\nWindow size :", WINDOW_SIZE)
print("Window stride:", STRIDE)

# ============================================================
# 1. LOAD LANDMARK CSV
# ============================================================

print("\nLoading landmarks.csv...")

df = pd.read_csv(LANDMARK_FILE)

print("✓ landmarks.csv loaded")

print("\nRows        :", len(df))
print("Words       :", df["label"].nunique())
print("Columns     :", len(df.columns))

# ============================================================
# 2. EXACT LANDMARK FEATURES
# ============================================================

feature_columns = [
    f"f{i}"
    for i in range(126)
]

missing = [
    col
    for col in feature_columns
    if col not in df.columns
]

if missing:

    print("\n❌ Missing features:")
    print(missing)

    raise ValueError(
        "Landmark features are missing."
    )

print(
    "Landmark features:",
    len(feature_columns)
)

print("✓ f0 to f125 present")

# ============================================================
# 3. CHECK FRAME DIRECTORY
# ============================================================

if not FRAMES_DIR.exists():

    raise FileNotFoundError(
        f"Frames directory not found: "
        f"{FRAMES_DIR}"
    )

print("\nFrames directory:")
print(FRAMES_DIR)

# ============================================================
# 4. GET WORD FOLDERS
# ============================================================

word_folders = sorted([
    folder
    for folder in FRAMES_DIR.iterdir()
    if folder.is_dir()
])

print(
    "\nWord folders:",
    len(word_folders)
)

if len(word_folders) != 2667:

    raise ValueError(
        f"Expected 2667 folders, "
        f"found {len(word_folders)}"
    )

print("✓ All 2667 folders found")

# ============================================================
# 5. ANALYZE EACH WORD
# ============================================================

print("\nAnalyzing frame counts...")

results = []

for index, folder in enumerate(
    word_folders
):

    word = folder.name

    frames = sorted(
        folder.glob("*.jpg")
    )

    frame_count = len(frames)

    # Number of possible temporal windows
    if frame_count < WINDOW_SIZE:

        windows = 1

    else:

        windows = (
            (frame_count - WINDOW_SIZE)
            // STRIDE
        ) + 1

    results.append({

        "label": word,

        "frames": frame_count,

        "windows": windows,

        "has_enough_frames":
            frame_count >= WINDOW_SIZE

    })

    if (index + 1) % 250 == 0:

        print(
            f"Analyzed "
            f"{index + 1}/2667"
        )

# ============================================================
# 6. DATAFRAME
# ============================================================

stats = pd.DataFrame(
    results
)

print("\n✓ Folder analysis complete")

# ============================================================
# 7. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("FRAME STATISTICS")
print("=" * 60)

print(
    "\nMinimum frames:",
    stats["frames"].min()
)

print(
    "Maximum frames:",
    stats["frames"].max()
)

print(
    "Average frames:",
    round(
        stats["frames"].mean(),
        2
    )
)

print(
    "Median frames:",
    stats["frames"].median()
)

# ============================================================
# 8. WINDOWS STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("WINDOW STATISTICS")
print("=" * 60)

print(
    "\nWindow size:",
    WINDOW_SIZE
)

print(
    "Stride:",
    STRIDE
)

print(
    "Minimum windows:",
    stats["windows"].min()
)

print(
    "Maximum windows:",
    stats["windows"].max()
)

print(
    "Average windows:",
    round(
        stats["windows"].mean(),
        2
    )
)

print(
    "Total possible windows:",
    stats["windows"].sum()
)

# ============================================================
# 9. SHORT SEQUENCES
# ============================================================

short = stats[
    stats["frames"] < WINDOW_SIZE
].sort_values(
    "frames"
)

print("\n" + "=" * 60)
print("SEQUENCES SHORTER THAN 64 FRAMES")
print("=" * 60)

print(
    "\nNumber of words:",
    len(short)
)

if len(short) > 0:

    print(
        "\nShortest words:"
    )

    print(
        short[
            ["label", "frames"]
        ].head(20).to_string(
            index=False
        )
    )

# ============================================================
# 10. WORDS WITH MANY WINDOWS
# ============================================================

print("\n" + "=" * 60)
print("WORDS WITH MOST TEMPORAL WINDOWS")
print("=" * 60)

print(
    stats[
        ["label", "frames", "windows"]
    ]
    .sort_values(
        "windows",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)

# ============================================================
# 11. WINDOW DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("WINDOW DISTRIBUTION")
print("=" * 60)

bins = [

    ("1 window",
     stats["windows"] == 1),

    ("2-5 windows",
     stats["windows"].between(2, 5)),

    ("6-10 windows",
     stats["windows"].between(6, 10)),

    ("11-20 windows",
     stats["windows"].between(11, 20)),

    ("21-50 windows",
     stats["windows"].between(21, 50)),

    ("51-100 windows",
     stats["windows"].between(51, 100)),

    ("101+ windows",
     stats["windows"] >= 101)

]

for name, condition in bins:

    count = condition.sum()

    print(
        f"{name:<15}: {count} words"
    )

# ============================================================
# 12. TOTAL POSSIBLE TRAINING WINDOWS
# ============================================================

total_windows = int(
    stats["windows"].sum()
)

print("\n" + "=" * 60)
print("TOTAL TEMPORAL DATA")
print("=" * 60)

print(
    "\nOriginal word sequences:",
    len(stats)
)

print(
    "Possible 64-frame windows:",
    total_windows
)

print(
    "Expansion factor:",
    round(
        total_windows /
        len(stats),
        2
    )
)

# ============================================================
# 13. IMPORTANT DATASET WARNING
# ============================================================

print("\n" + "=" * 60)
print("IMPORTANT DATASET CHECK")
print("=" * 60)

print(
    """
We need to distinguish between:

1. ORIGINAL RECORDINGS
2. TEMPORAL WINDOWS

Multiple windows extracted from the SAME
recording are NOT independent recordings.

Therefore:

✓ Windows can increase training data.

BUT:

❌ We must NOT randomly put windows from
   the same recording into both train and test.

Otherwise the test accuracy can become
artificially high because the model sees
nearly identical motion during training.
"""
)

# ============================================================
# 14. CHECK FRAME NAMING
# ============================================================

print("\n" + "=" * 60)
print("FRAME NAMING CHECK")
print("=" * 60)

bad_examples = []

for folder in word_folders[:50]:

    frames = sorted(
        folder.glob("*.jpg")
    )

    for expected, frame in enumerate(
        frames
    ):

        expected_name = (
            f"frame_{expected:04d}.jpg"
        )

        if frame.name != expected_name:

            bad_examples.append({

                "word": folder.name,

                "expected":
                    expected_name,

                "found":
                    frame.name

            })

            break

if bad_examples:

    print(
        "⚠ Frame naming problems found:"
    )

    for item in bad_examples[:10]:

        print(item)

else:

    print(
        "✓ Frame naming looks correct"
    )

# ============================================================
# 15. SAVE INSPECTION REPORT
# ============================================================

output_file = Path(
    "dataset/recording_inspection.csv"
)

stats.to_csv(
    output_file,
    index=False
)

print("\n✓ Inspection report saved:")

print(output_file)

# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SCRIPT 14 COMPLETE")
print("=" * 60)

print(
    "\nWords:",
    len(stats)
)

print(
    "Total frames:",
    stats["frames"].sum()
)

print(
    "Possible 64-frame windows:",
    total_windows
)

print(
    "Words shorter than 64 frames:",
    len(short)
)

print(
    "\nNo files were modified."
)

print(
    "No landmarks were regenerated."
)

print(
    "No model was trained."
)

print(
    "\n✓ Inspection complete."
)

print(
    "\nSend me this COMPLETE output before "
    "we create Script 15."
)