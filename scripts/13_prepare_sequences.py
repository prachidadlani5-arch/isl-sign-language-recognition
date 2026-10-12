import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# SCRIPT 13 - PREPARE TEMPORAL SEQUENCES
# ============================================================

INPUT_FILE = Path("dataset/sequence_metadata.csv")
OUTPUT_DIR = Path("dataset/sequences")

SEQUENCE_LENGTH = 64
RANDOM_STATE = 42

# ============================================================
# START
# ============================================================

print("=" * 60)
print("SCRIPT 13 - PREPARE TEMPORAL SEQUENCES")
print("=" * 60)

print("\nSequence length:", SEQUENCE_LENGTH)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading sequence_metadata.csv...")

df = pd.read_csv(INPUT_FILE)

print("✓ Dataset loaded")

print("\nRows       :", len(df))
print("Columns    :", len(df))
print("Words      :", df["label"].nunique())

# ============================================================
# 2. GET EXACT LANDMARK FEATURES
# ============================================================

# IMPORTANT:
# Do NOT use startswith("f")
# because columns such as frame_file also start with "f".

feature_columns = [
    f"f{i}"
    for i in range(126)
]

# Check that every expected landmark column exists.

missing_features = [
    col
    for col in feature_columns
    if col not in df.columns
]

if missing_features:

    print("\n❌ Missing landmark columns:")

    for col in missing_features:
        print(col)

    raise ValueError(
        "Some landmark features are missing."
    )

print("\nFeatures   :", len(feature_columns))
print("✓ Exactly f0 to f125 selected")

# ============================================================
# 3. VERIFY NO EXTRA LANDMARKS
# ============================================================

print("\nChecking landmark columns...")

unexpected_landmarks = []

for col in df.columns:

    if (
        col.startswith("f")
        and col not in feature_columns
        and col not in [
            "frame_file"
        ]
    ):
        unexpected_landmarks.append(col)

if unexpected_landmarks:

    print(
        "⚠ Additional f-prefixed columns found:"
    )

    print(
        unexpected_landmarks
    )

else:

    print(
        "✓ No unexpected landmark columns"
    )

# ============================================================
# 4. VERIFY DATASET
# ============================================================

if len(df) != 281001:

    raise ValueError(
        f"Expected 281001 rows, found {len(df)}"
    )

if df["label"].nunique() != 2667:

    raise ValueError(
        f"Expected 2667 words, "
        f"found {df['label'].nunique()}"
    )

print("\n✓ 281001 rows verified")
print("✓ 2667 words verified")

# ============================================================
# 5. CHECK LANDMARK VALUES
# ============================================================

print("\nChecking landmark values...")

landmark_values = df[
    feature_columns
].values

nan_count = np.isnan(
    landmark_values
).sum()

inf_count = np.isinf(
    landmark_values
).sum()

print("NaN values      :", nan_count)
print("Infinite values:", inf_count)

if nan_count != 0:

    raise ValueError(
        "NaN values found in landmarks."
    )

if inf_count != 0:

    raise ValueError(
        "Infinite values found in landmarks."
    )

print("✓ Landmark values are valid")

# ============================================================
# 6. LABEL ENCODING
# ============================================================

print("\nCreating label encoding...")

words = sorted(
    df["label"]
    .astype(str)
    .unique()
    .tolist()
)

label_to_id = {
    word: index
    for index, word in enumerate(words)
}

id_to_label = {
    index: word
    for word, index in label_to_id.items()
}

df["label_id"] = (
    df["label"]
    .astype(str)
    .map(label_to_id)
)

print(
    "Total classes:",
    len(words)
)

print("✓ Labels encoded")

# ============================================================
# 7. GROUP FRAMES BY WORD
# ============================================================

print("\nGrouping frames by word...")

groups = []

for label, group in df.groupby(
    "label",
    sort=False
):

    group = group.sort_values(
        "frame_number"
    )

    groups.append(
        (
            str(label),
            group
        )
    )

print(
    "Sequences found:",
    len(groups)
)

if len(groups) != 2667:

    raise ValueError(
        "Expected 2667 sequences."
    )

# ============================================================
# 8. FIXED SEQUENCE FUNCTION
# ============================================================

def make_fixed_sequence(
    values,
    length=64
):
    """
    Convert variable-length landmark data
    into exactly 64 frames.

    Short sequences:
        Sample repeatedly across the
        complete sequence.

    Long sequences:
        Uniformly sample across the
        complete sequence.

    No random cropping is used.
    """

    num_frames = len(values)

    if num_frames == 0:

        raise ValueError(
            "Empty sequence encountered."
        )

    # --------------------------------------------------------
    # EXACT LENGTH
    # --------------------------------------------------------

    if num_frames == length:

        return values.astype(
            np.float32
        )

    # --------------------------------------------------------
    # ANY OTHER LENGTH
    # --------------------------------------------------------

    indices = np.linspace(
        0,
        num_frames - 1,
        length
    )

    indices = np.round(
        indices
    ).astype(int)

    sequence = values[
        indices
    ]

    return sequence.astype(
        np.float32
    )


# ============================================================
# 9. CREATE SEQUENCES
# ============================================================

print("\nCreating fixed-length sequences...")

X = []
y = []
sequence_names = []

for index, (label, group) in enumerate(
    groups
):

    values = group[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    sequence = make_fixed_sequence(
        values,
        SEQUENCE_LENGTH
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        126
    ):

        raise ValueError(
            f"Invalid sequence shape "
            f"for {label}: "
            f"{sequence.shape}"
        )

    X.append(sequence)

    y.append(
        label_to_id[label]
    )

    sequence_names.append(label)

    if (index + 1) % 100 == 0:

        print(
            f"Processed: "
            f"{index + 1}/2667"
        )

# ============================================================
# 10. CONVERT TO NUMPY
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int32
)

print("\n✓ Sequences created")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# 11. SHAPE CHECK
# ============================================================

print("\n" + "=" * 60)
print("SEQUENCE SHAPE VERIFICATION")
print("=" * 60)

print("Samples :", X.shape[0])
print("Frames  :", X.shape[1])
print("Features:", X.shape[2])
print("Classes :", len(np.unique(y)))

if X.shape != (
    2667,
    64,
    126
):

    raise ValueError(
        f"Unexpected X shape: {X.shape}"
    )

if y.shape != (2667,):

    raise ValueError(
        f"Unexpected y shape: {y.shape}"
    )

print("✓ X shape = (2667, 64, 126)")
print("✓ y shape = (2667,)")

# ============================================================
# 12. CHECK SEQUENCES FOR NaN / INF
# ============================================================

print("\nChecking sequence values...")

print(
    "NaN values:",
    np.isnan(X).sum()
)

print(
    "Infinite values:",
    np.isinf(X).sum()
)

if np.isnan(X).sum() != 0:

    raise ValueError(
        "NaN values found."
    )

if np.isinf(X).sum() != 0:

    raise ValueError(
        "Infinite values found."
    )

print("✓ All sequence values valid")

# ============================================================
# 13. IMPORTANT DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("IMPORTANT DATASET STRUCTURE")
print("=" * 60)

print(
    "Number of original sequences:",
    len(X)
)

print(
    "Number of classes:",
    len(words)
)

print(
    "Sequences per class:",
    "1"
)

print(
    "\nEach word folder currently represents "
    "one original sequence."
)

print(
    "Therefore, we will NOT create fake "
    "train/test copies of the same sequence."
)

# ============================================================
# 14. CREATE CLASS-PRESERVING SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

print(
    "\nBecause there is currently exactly "
    "one sequence per class:"
)

print(
    "✓ We keep every class available "
    "for training."
)

print(
    "⚠ A conventional 80/10/10 split "
    "would remove classes from training."
)

print(
    "⚠ That would be invalid for a "
    "2667-class vocabulary."
)

# ------------------------------------------------------------
# For now:
#
# ALL sequences go into training.
#
# We will create validation/test samples
# later from additional recordings/windows
# if available.
# ------------------------------------------------------------

X_train = X
y_train = y

X_val = np.empty(
    (0, SEQUENCE_LENGTH, 126),
    dtype=np.float32
)

y_val = np.empty(
    (0,),
    dtype=np.int32
)

X_test = np.empty(
    (0, SEQUENCE_LENGTH, 126),
    dtype=np.float32
)

y_test = np.empty(
    (0,),
    dtype=np.int32
)

print("\nTraining samples  :", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples   :", len(X_test))

print(
    "\n✓ All 2667 classes preserved "
    "in training data"
)

# ============================================================
# 15. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 16. SAVE TRAINING DATA
# ============================================================

print("\nSaving training data...")

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train
)

# ============================================================
# 17. SAVE EMPTY VALIDATION / TEST FILES
# ============================================================

np.save(
    OUTPUT_DIR / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_DIR / "y_val.npy",
    y_val
)

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test
)

# ============================================================
# 18. SAVE LABEL MAP
# ============================================================

label_map = pd.DataFrame({
    "label_id": range(
        len(words)
    ),
    "label": words
})

label_map.to_csv(
    OUTPUT_DIR / "label_map.csv",
    index=False
)

# ============================================================
# 19. SAVE SEQUENCE INDEX
# ============================================================

sequence_index = pd.DataFrame({
    "sequence_id": range(
        len(sequence_names)
    ),
    "label_id": [
        label_to_id[word]
        for word in sequence_names
    ],
    "label": sequence_names
})

sequence_index.to_csv(
    OUTPUT_DIR / "sequence_index.csv",
    index=False
)

# ============================================================
# 20. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

# Reload training data

check_X = np.load(
    OUTPUT_DIR / "X_train.npy"
)

check_y = np.load(
    OUTPUT_DIR / "y_train.npy"
)

print(
    "Saved X_train:",
    check_X.shape
)

print(
    "Saved y_train:",
    check_y.shape
)

if check_X.shape != (
    2667,
    64,
    126
):

    raise ValueError(
        "Saved X_train has wrong shape."
    )

if check_y.shape != (
    2667,
):

    raise ValueError(
        "Saved y_train has wrong shape."
    )

if len(
    np.unique(check_y)
) != 2667:

    raise ValueError(
        "Not all 2667 classes are present."
    )

print(
    "✓ All 2667 classes present"
)

print(
    "✓ 64 frames per sequence"
)

print(
    "✓ 126 features per frame"
)

print(
    "✓ Training dataset verified"
)

# ============================================================
# 21. OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SCRIPT 13 COMPLETE")
print("=" * 60)

print("\nCreated:")

print(
    "dataset/sequences/X_train.npy"
)

print(
    "dataset/sequences/y_train.npy"
)

print(
    "dataset/sequences/X_val.npy"
)

print(
    "dataset/sequences/y_val.npy"
)

print(
    "dataset/sequences/X_test.npy"
)

print(
    "dataset/sequences/y_test.npy"
)

print(
    "dataset/sequences/label_map.csv"
)

print(
    "dataset/sequences/sequence_index.csv"
)

print("\nFinal training shape:")
print(
    "X_train =",
    X_train.shape
)

print(
    "y_train =",
    y_train.shape
)

print("\n✓ Script 13 finished successfully.")
print(
    "\nIMPORTANT: Do NOT train the final model yet."
)
print(
    "We need proper validation/test sequences "
    "before claiming an accuracy."
)