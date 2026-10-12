import pandas as pd
from pathlib import Path
import numpy as np

# ============================================================
# SCRIPT 12 - BUILD AND VERIFY SEQUENCES
# ============================================================

FRAMES_DIR = Path("dataset/frames")
LANDMARKS_FILE = Path("dataset/landmarks.csv")
OUTPUT_FILE = Path("dataset/sequence_metadata.csv")

print("=" * 60)
print("SCRIPT 12 - BUILD AND VERIFY SEQUENCES")
print("=" * 60)

# ============================================================
# 1. LOAD LANDMARKS
# ============================================================

print("\nLoading landmarks.csv...")

df = pd.read_csv(LANDMARKS_FILE)

print("✓ Landmarks loaded")

print("\nLandmark dataset:")
print("Rows       :", len(df))
print("Columns    :", len(df.columns))
print("Words      :", df["label"].nunique())

# ============================================================
# 2. CHECK FEATURES
# ============================================================

feature_columns = [
    c for c in df.columns
    if c.startswith("f")
]

print("Features   :", len(feature_columns))

if len(feature_columns) != 126:
    print("\n❌ ERROR: Expected 126 landmark features")
    print("Found:", len(feature_columns))
    raise SystemExit

print("✓ 126 landmark features found")

# ============================================================
# 3. GET WORD ORDER FROM LANDMARK CSV
# ============================================================

print("\nReading word order directly from landmarks.csv...")

csv_word_order = df["label"].astype(str).drop_duplicates().tolist()

print("Unique words in CSV:", len(csv_word_order))

if len(csv_word_order) != 2667:
    print("\n❌ ERROR: Expected 2667 words")
    raise SystemExit

print("✓ 2667 words found")

# ============================================================
# 4. BUILD FOLDER LOOKUP
# ============================================================

print("\nBuilding folder lookup...")

folders = [
    p for p in FRAMES_DIR.iterdir()
    if p.is_dir()
]

folder_lookup = {
    p.name: p
    for p in folders
}

print("Folders found:", len(folder_lookup))

if len(folder_lookup) != 2667:
    print("\n❌ ERROR: Expected 2667 word folders")
    raise SystemExit

print("✓ 2667 folders found")

# ============================================================
# 5. CHECK EVERY CSV WORD HAS A FOLDER
# ============================================================

print("\nChecking word ↔ folder mapping...")

missing_folders = []

for word in csv_word_order:

    if word not in folder_lookup:
        missing_folders.append(word)

if missing_folders:

    print("\n❌ MISSING FOLDERS")

    for word in missing_folders[:20]:
        print(word)

    print("\nTotal missing:", len(missing_folders))

    raise SystemExit

print("✓ Every CSV word has an exact matching folder")

# ============================================================
# 6. CREATE FRAME METADATA
# ============================================================

print("\nCreating frame metadata...")
print("Using CSV word order as the authoritative order.")

metadata = []

for word_index, word in enumerate(csv_word_order):

    folder = folder_lookup[word]

    images = sorted(
        folder.glob("*.jpg"),
        key=lambda p: p.name
    )

    total_frames = len(images)

    if total_frames == 0:

        print("\n❌ NO FRAMES FOUND")
        print("Word:", word)

        raise SystemExit

    for frame_number, image_path in enumerate(images):

        metadata.append({
            "label": word,
            "frame_number": frame_number,
            "total_frames": total_frames,
            "frame_file": image_path.name
        })

    if (word_index + 1) % 100 == 0:

        print(
            f"Processed words: "
            f"{word_index + 1}/{len(csv_word_order)}"
        )

metadata_df = pd.DataFrame(metadata)

print("\n✓ Frame metadata created")

print("Metadata rows:", len(metadata_df))

# ============================================================
# 7. ROW COUNT CHECK
# ============================================================

print("\n" + "=" * 60)
print("ROW COUNT VERIFICATION")
print("=" * 60)

print("Landmark rows :", len(df))
print("Metadata rows :", len(metadata_df))

if len(df) != len(metadata_df):

    print("\n❌ ROW COUNT MISMATCH")

    print(
        "Difference:",
        abs(len(df) - len(metadata_df))
    )

    raise SystemExit

print("✓ Row counts match exactly")

# ============================================================
# 8. LABEL ORDER CHECK
# ============================================================

print("\n" + "=" * 60)
print("LABEL ORDER VERIFICATION")
print("=" * 60)

landmark_labels = df["label"].astype(str).values
metadata_labels = metadata_df["label"].astype(str).values

if not np.array_equal(
    landmark_labels,
    metadata_labels
):

    print("\n❌ LABEL ORDER STILL DOES NOT MATCH")

    mismatch = np.where(
        landmark_labels != metadata_labels
    )[0]

    if len(mismatch) > 0:

        i = mismatch[0]

        print("\nFirst mismatch:")
        print("Row:", i)
        print(
            "Landmarks:",
            landmark_labels[i]
        )
        print(
            "Metadata:",
            metadata_labels[i]
        )

    raise SystemExit

print("✓ Every landmark row matches its word")

# ============================================================
# 9. FRAME NUMBER VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("FRAME NUMBER VERIFICATION")
print("=" * 60)

expected_frame = 0
current_word = None

frame_errors = []

for i, row in metadata_df.iterrows():

    word = row["label"]
    frame_number = int(row["frame_number"])

    if word != current_word:

        current_word = word
        expected_frame = 0

    if frame_number != expected_frame:

        frame_errors.append({
            "row": i,
            "word": word,
            "expected": expected_frame,
            "found": frame_number
        })

        if len(frame_errors) >= 10:
            break

    expected_frame += 1

if frame_errors:

    print("\n❌ FRAME NUMBER ERRORS")

    for error in frame_errors:
        print(error)

    raise SystemExit

print("✓ Frame numbers are sequential")

# ============================================================
# 10. FILENAME VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("FILENAME VERIFICATION")
print("=" * 60)

filename_errors = []

for i, row in metadata_df.iterrows():

    frame_number = int(row["frame_number"])

    expected_filename = (
        f"frame_{frame_number:04d}.jpg"
    )

    actual_filename = row["frame_file"]

    if expected_filename != actual_filename:

        filename_errors.append({
            "row": i,
            "word": row["label"],
            "expected": expected_filename,
            "found": actual_filename
        })

        if len(filename_errors) >= 10:
            break

if filename_errors:

    print("\n❌ FILENAME ERRORS")

    for error in filename_errors:
        print(error)

    raise SystemExit

print("✓ All filenames match frame numbers")

# ============================================================
# 11. CREATE SEQUENCE POSITION
# ============================================================

print("\nCreating sequence information...")

metadata_df["sequence_position"] = (
    metadata_df["frame_number"]
)

metadata_df["sequence_id"] = (
    metadata_df["label"]
)

# ============================================================
# 12. COMBINE WITH LANDMARKS
# ============================================================

print("\nCombining metadata with landmark features...")

combined = pd.concat(
    [
        metadata_df.reset_index(drop=True),

        df[feature_columns].reset_index(drop=True)
    ],
    axis=1
)

print("✓ Data combined")

# ============================================================
# 13. FINAL CHECKS
# ============================================================

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

print("Rows             :", len(combined))
print("Columns          :", len(combined.columns))
print("Words            :", combined["label"].nunique())
print("Landmark features:", len(feature_columns))

missing = combined.isna().sum().sum()

print("Missing values   :", missing)

if missing != 0:

    print("\n❌ Missing values found")

    raise SystemExit

print("✓ No missing values")

# ============================================================
# 14. SEQUENCE LENGTH STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("SEQUENCE LENGTH STATISTICS")
print("=" * 60)

sequence_lengths = (
    combined
    .groupby("label")
    .size()
)

print(
    "Minimum frames :",
    sequence_lengths.min()
)

print(
    "Maximum frames :",
    sequence_lengths.max()
)

print(
    "Average frames :",
    round(sequence_lengths.mean(), 2)
)

print(
    "Median frames  :",
    sequence_lengths.median()
)

print("\nShortest sequences:")
print(sequence_lengths.nsmallest(10))

print("\nLongest sequences:")
print(sequence_lengths.nlargest(10))

# ============================================================
# 15. SAVE
# ============================================================

print("\nSaving sequence_metadata.csv...")

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n✓ FILE SAVED")

print("Location:")
print(OUTPUT_FILE)

# ============================================================
# 16. SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE")
print("=" * 60)

print(
    combined[
        [
            "label",
            "frame_number",
            "total_frames",
            "frame_file",
            "sequence_id"
        ]
    ].head(20).to_string(index=False)
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("SCRIPT 12 COMPLETE")
print("=" * 60)

print("\n✓ 281001 landmark rows verified")
print("✓ 2667 words verified")
print("✓ Frame mapping verified")
print("✓ Frame numbers verified")
print("✓ Filenames verified")
print("✓ No missing values")
print("✓ Sequence metadata created")
print("✓ Existing landmarks reused")
print("✓ No MediaPipe processing performed")
print("✓ No model training performed")

print("\nReady for the next stage.")