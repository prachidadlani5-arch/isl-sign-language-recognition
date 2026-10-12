import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SCRIPT 15 - BUILD BALANCED TEMPORAL DATASET
# ============================================================

LANDMARK_FILE = Path("dataset/landmarks.csv")
OUTPUT_DIR = Path("dataset/sequences")

WINDOW_SIZE = 64
MAX_WINDOWS_PER_CLASS = 5

# For sequences shorter than 64 frames:
# interpolate them to exactly 64 frames.
HANDLE_SHORT_SEQUENCES = True

# ============================================================
# START
# ============================================================

print("=" * 60)
print("SCRIPT 15 - BUILD BALANCED TEMPORAL DATASET")
print("=" * 60)

print("\nWindow size:", WINDOW_SIZE)
print("Maximum windows per class:", MAX_WINDOWS_PER_CLASS)

# ============================================================
# 1. LOAD LANDMARK DATA
# ============================================================

print("\nLoading landmarks.csv...")

df = pd.read_csv(LANDMARK_FILE)

print("✓ Dataset loaded")

print("\nRows       :", len(df))
print("Words      :", df["label"].nunique())
print("Columns    :", len(df.columns))

# ============================================================
# 2. VERIFY LANDMARK FEATURES
# ============================================================

FEATURE_COLUMNS = [
    f"f{i}" for i in range(126)
]

missing = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing landmark features: {missing}"
    )

print("\nFeatures:", len(FEATURE_COLUMNS))
print("✓ Exactly f0 to f125 found")

# ============================================================
# 3. VERIFY DATA
# ============================================================

values = df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32
)

if np.isnan(values).any():

    raise ValueError(
        "NaN values found in landmark data."
    )

if np.isinf(values).any():

    raise ValueError(
        "Infinite values found in landmark data."
    )

print("✓ No NaN values")
print("✓ No infinite values")

# ============================================================
# 4. GET WORD ORDER
# ============================================================

words = df["label"].drop_duplicates().tolist()

print(
    "\nTotal classes:",
    len(words)
)

if len(words) != 2667:

    raise ValueError(
        f"Expected 2667 classes, "
        f"found {len(words)}"
    )

# ============================================================
# 5. LABEL ENCODING
# ============================================================

label_to_id = {
    word: idx
    for idx, word in enumerate(words)
}

id_to_label = {
    idx: word
    for word, idx in label_to_id.items()
}

# ============================================================
# 6. OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 7. TEMPORAL RESAMPLING FUNCTION
# ============================================================

def resize_sequence(
    sequence,
    target_length=64
):

    """
    Resize a sequence to target_length
    using linear interpolation.

    Input:
        (frames, features)

    Output:
        (target_length, features)
    """

    old_length = sequence.shape[0]

    if old_length == target_length:

        return sequence.astype(
            np.float32
        )

    old_positions = np.linspace(
        0,
        1,
        old_length
    )

    new_positions = np.linspace(
        0,
        1,
        target_length
    )

    output = np.empty(
        (
            target_length,
            sequence.shape[1]
        ),
        dtype=np.float32
    )

    for feature in range(
        sequence.shape[1]
    ):

        output[:, feature] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature]
        )

    return output


# ============================================================
# 8. CREATE WINDOWS
# ============================================================

all_sequences = []
all_labels = []

sequence_records = []

print("\nCreating temporal sequences...")

for class_index, word in enumerate(words):

    word_df = df[
        df["label"] == word
    ]

    sequence = word_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    frame_count = len(sequence)

    # --------------------------------------------------------
    # SHORT SEQUENCE
    # --------------------------------------------------------

    if frame_count < WINDOW_SIZE:

        if not HANDLE_SHORT_SEQUENCES:

            print(
                f"Skipping {word}: "
                f"{frame_count} frames"
            )

            continue

        resized = resize_sequence(
            sequence,
            WINDOW_SIZE
        )

        all_sequences.append(
            resized
        )

        all_labels.append(
            class_index
        )

        sequence_records.append({

            "label": word,

            "label_id": class_index,

            "window_id": 0,

            "start_frame": 0,

            "end_frame":
                frame_count - 1,

            "original_frames":
                frame_count,

            "window_frames":
                WINDOW_SIZE,

            "resampled":
                True

        })

    # --------------------------------------------------------
    # NORMAL SEQUENCE
    # --------------------------------------------------------

    else:

        possible_windows = (
            frame_count - WINDOW_SIZE
        ) // 16 + 1

        # ----------------------------------------------------
        # EVENLY SELECT MAX_WINDOWS_PER_CLASS
        # ----------------------------------------------------

        if possible_windows <= MAX_WINDOWS_PER_CLASS:

            starts = np.arange(
                0,
                frame_count - WINDOW_SIZE + 1,
                16
            )

        else:

            max_start = (
                frame_count - WINDOW_SIZE
            )

            starts = np.linspace(
                0,
                max_start,
                MAX_WINDOWS_PER_CLASS,
                dtype=int
            )

        # Remove duplicates caused by rounding

        starts = np.unique(starts)

        for window_id, start in enumerate(
            starts
        ):

            end = start + WINDOW_SIZE

            window = sequence[
                start:end
            ]

            if len(window) != WINDOW_SIZE:

                continue

            all_sequences.append(
                window
            )

            all_labels.append(
                class_index
            )

            sequence_records.append({

                "label": word,

                "label_id": class_index,

                "window_id": window_id,

                "start_frame": int(start),

                "end_frame":
                    int(end - 1),

                "original_frames":
                    frame_count,

                "window_frames":
                    WINDOW_SIZE,

                "resampled":
                    False

            })

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (class_index + 1) % 100 == 0:

        print(
            f"Processed "
            f"{class_index + 1}/"
            f"{len(words)}"
        )

# ============================================================
# 9. CONVERT TO NUMPY
# ============================================================

X = np.asarray(
    all_sequences,
    dtype=np.float32
)

y = np.asarray(
    all_labels,
    dtype=np.int32
)

metadata = pd.DataFrame(
    sequence_records
)

# ============================================================
# 10. BASIC SHAPE CHECK
# ============================================================

print("\n" + "=" * 60)
print("SEQUENCE DATASET")
print("=" * 60)

print(
    "\nX shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "Metadata rows:",
    len(metadata)
)

if X.ndim != 3:

    raise ValueError(
        f"Expected 3D X, got {X.ndim}D"
    )

if X.shape[1] != WINDOW_SIZE:

    raise ValueError(
        "Incorrect sequence length."
    )

if X.shape[2] != 126:

    raise ValueError(
        "Incorrect feature count."
    )

if len(X) != len(y):

    raise ValueError(
        "X and y sample counts differ."
    )

if len(X) != len(metadata):

    raise ValueError(
        "X and metadata counts differ."
    )

print(
    "✓ Correct sequence shape"
)

# ============================================================
# 11. VALUE CHECK
# ============================================================

print("\nChecking sequence values...")

nan_count = np.isnan(X).sum()

inf_count = np.isinf(X).sum()

print(
    "NaN values:",
    nan_count
)

print(
    "Infinite values:",
    inf_count
)

if nan_count != 0:

    raise ValueError(
        "NaN values detected."
    )

if inf_count != 0:

    raise ValueError(
        "Infinite values detected."
    )

print(
    "✓ Sequence values valid"
)

# ============================================================
# 12. CLASS DISTRIBUTION
# ============================================================

class_counts = (
    pd.Series(y)
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)

print(
    "\nMinimum samples/class:",
    class_counts.min()
)

print(
    "Maximum samples/class:",
    class_counts.max()
)

print(
    "Average samples/class:",
    round(
        class_counts.mean(),
        2
    )
)

print(
    "Total sequences:",
    len(X)
)

# ============================================================
# 13. VERIFY EVERY CLASS EXISTS
# ============================================================

unique_labels = np.unique(y)

print(
    "\nClasses represented:",
    len(unique_labels)
)

if len(unique_labels) != 2667:

    missing_ids = sorted(
        set(range(2667))
        - set(unique_labels)
    )

    missing_words = [
        id_to_label[i]
        for i in missing_ids
    ]

    print(
        "\n❌ Missing classes:"
    )

    print(
        missing_words[:50]
    )

    raise ValueError(
        "Not every class has a sequence."
    )

print(
    "✓ All 2667 classes represented"
)

# ============================================================
# 14. SHORT-SEQUENCE CHECK
# ============================================================

short_metadata = metadata[
    metadata["resampled"] == True
]

print("\n" + "=" * 60)
print("SHORT SEQUENCES")
print("=" * 60)

print(
    "\nShort recordings resampled:",
    len(short_metadata)
)

if len(short_metadata) > 0:

    print(
        "\nExamples:"
    )

    print(
        short_metadata[
            [
                "label",
                "original_frames"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

# ============================================================
# 15. SAVE DATASET
# ============================================================

print("\n" + "=" * 60)
print("SAVING DATASET")
print("=" * 60)

X_file = (
    OUTPUT_DIR /
    "X_temporal.npy"
)

y_file = (
    OUTPUT_DIR /
    "y_temporal.npy"
)

metadata_file = (
    OUTPUT_DIR /
    "temporal_metadata.csv"
)

label_map_file = (
    OUTPUT_DIR /
    "temporal_label_map.csv"
)

np.save(
    X_file,
    X
)

np.save(
    y_file,
    y
)

metadata.to_csv(
    metadata_file,
    index=False
)

label_map = pd.DataFrame({

    "label_id":
        list(id_to_label.keys()),

    "label":
        list(id_to_label.values())

})

label_map.to_csv(
    label_map_file,
    index=False
)

print(
    "\n✓ X saved:"
)

print(X_file)

print(
    "\n✓ y saved:"
)

print(y_file)

print(
    "\n✓ Metadata saved:"
)

print(metadata_file)

print(
    "\n✓ Label map saved:"
)

print(label_map_file)

# ============================================================
# 16. RELOAD VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("RELOAD VERIFICATION")
print("=" * 60)

X_check = np.load(
    X_file
)

y_check = np.load(
    y_file
)

metadata_check = pd.read_csv(
    metadata_file
)

print(
    "\nReloaded X:",
    X_check.shape
)

print(
    "Reloaded y:",
    y_check.shape
)

print(
    "Reloaded metadata:",
    len(metadata_check)
)

if not np.array_equal(
    X,
    X_check
):

    raise ValueError(
        "Saved X does not match original X."
    )

if not np.array_equal(
    y,
    y_check
):

    raise ValueError(
        "Saved y does not match original y."
    )

if len(metadata_check) != len(X):

    raise ValueError(
        "Metadata count mismatch."
    )

print(
    "✓ Saved data verified"
)

# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SCRIPT 15 COMPLETE")
print("=" * 60)

print(
    "\nOriginal recordings:",
    2667
)

print(
    "Temporal sequences:",
    len(X)
)

print(
    "Frames/sequence:",
    WINDOW_SIZE
)

print(
    "Features/frame:",
    126
)

print(
    "Classes:",
    len(unique_labels)
)

print(
    "Short sequences resampled:",
    len(short_metadata)
)

print(
    "\n✓ Dataset successfully created"
)

print(
    "\nIMPORTANT:"
)

print(
    "These windows come from the same "
    "original recordings."
)

print(
    "DO NOT perform a random window-level "
    "train/test split."
)

print(
    "\nScript 15 finished successfully."
)

print(
    "No model was trained."
)