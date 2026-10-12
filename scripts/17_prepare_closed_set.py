import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SCRIPT 17 - PREPARE CLOSED-SET TEMPORAL DATASET
# ============================================================

INPUT_DIR = Path("dataset/sequences")

X_FILE = INPUT_DIR / "X_temporal.npy"
Y_FILE = INPUT_DIR / "y_temporal.npy"
META_FILE = INPUT_DIR / "temporal_metadata.csv"
LABEL_FILE = INPUT_DIR / "temporal_label_map.csv"

RANDOM_STATE = 42

# ============================================================
# DATASET SETTINGS
# ============================================================

# Number of temporal windows used for training per class.
# We already have at most 5 windows/class.
#
# We keep ALL available windows for training.
#
# Validation/test windows are selected from the SAME
# recording, but they are NOT used during training.
#
# IMPORTANT:
# This is NOT an independent-recording benchmark because
# the dataset contains only one original recording per word.

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# START
# ============================================================

print("=" * 60)
print("SCRIPT 17 - PREPARE CLOSED-SET TEMPORAL DATASET")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading temporal dataset...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

metadata = pd.read_csv(META_FILE)
label_map = pd.read_csv(LABEL_FILE)

print("✓ X loaded")
print("✓ y loaded")
print("✓ Metadata loaded")
print("✓ Label map loaded")

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("Metadata rows:", len(metadata))

# ============================================================
# 2. BASIC VERIFICATION
# ============================================================

if X.ndim != 3:
    raise ValueError(
        f"Expected 3D X, found {X.ndim}D"
    )

if X.shape[1] != 64:
    raise ValueError(
        f"Expected 64 frames, found {X.shape[1]}"
    )

if X.shape[2] != 126:
    raise ValueError(
        f"Expected 126 features, found {X.shape[2]}"
    )

if len(X) != len(y):
    raise ValueError(
        "X and y sample counts do not match."
    )

if len(X) != len(metadata):
    raise ValueError(
        "X and metadata sample counts do not match."
    )

print("\n✓ Dataset dimensions verified")

# ============================================================
# 3. VERIFY VALUES
# ============================================================

print("\nChecking landmark values...")

nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()

print("NaN values      :", nan_count)
print("Infinite values :", inf_count)

if nan_count != 0:
    raise ValueError("NaN values found.")

if inf_count != 0:
    raise ValueError("Infinite values found.")

print("✓ All values valid")

# ============================================================
# 4. VERIFY CLASSES
# ============================================================

unique_classes = np.unique(y)

print("\nTotal classes:", len(unique_classes))

if len(unique_classes) != 2667:
    raise ValueError(
        f"Expected 2667 classes, found "
        f"{len(unique_classes)}"
    )

print("✓ All 2667 classes present")

# ============================================================
# 5. IMPORTANT DATASET FACT
# ============================================================

print("\n" + "=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)

print("""
This dataset contains:

2667 vocabulary classes
but only ONE original recording per class.

Each recording has been converted into
multiple temporal windows.

Therefore:

✓ All 2667 classes will be represented in training.
✓ Windows from a class remain associated with that class.
✓ Validation/test windows are NOT independent recordings.

Because there is only one recording per word,
this is a closed-set temporal evaluation rather
than an independent-recording benchmark.
""")

# ============================================================
# 6. CREATE INDEX LISTS
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

train_indices = []
val_indices = []
test_indices = []

# ============================================================
# 7. SPLIT WINDOWS WITHIN EACH CLASS
# ============================================================

print("\nCreating closed-set split...")

for class_id in unique_classes:

    class_indices = np.where(
        y == class_id
    )[0]

    # --------------------------------------------------------
    # SHUFFLE WINDOWS OF THIS CLASS
    # --------------------------------------------------------

    class_indices = class_indices.copy()

    rng.shuffle(class_indices)

    n = len(class_indices)

    # --------------------------------------------------------
    # SPECIAL CASE: ONLY ONE WINDOW
    # --------------------------------------------------------

    if n == 1:

        # There is no way to create independent
        # validation/test samples.

        train_indices.extend(
            class_indices
        )

        continue

    # --------------------------------------------------------
    # SPECIAL CASE: TWO WINDOWS
    # --------------------------------------------------------

    elif n == 2:

        train_indices.append(
            class_indices[0]
        )

        # Keep second sample for validation.
        val_indices.append(
            class_indices[1]
        )

        continue

    # --------------------------------------------------------
    # THREE WINDOWS
    # --------------------------------------------------------

    elif n == 3:

        train_indices.append(
            class_indices[0]
        )

        val_indices.append(
            class_indices[1]
        )

        test_indices.append(
            class_indices[2]
        )

        continue

    # --------------------------------------------------------
    # FOUR OR MORE WINDOWS
    # --------------------------------------------------------

    else:

        train_count = max(
            1,
            int(
                round(
                    n * TRAIN_RATIO
                )
            )
        )

        val_count = max(
            1,
            int(
                round(
                    n * VAL_RATIO
                )
            )
        )

        # Make sure at least one test sample remains.

        if (
            train_count
            + val_count
            >= n
        ):

            val_count = 1

            train_count = n - 2

        test_count = (
            n
            - train_count
            - val_count
        )

        # Safety

        if test_count < 1:

            test_count = 1

            train_count = max(
                1,
                n - val_count - 1
            )

        # ----------------------------------------------------
        # ASSIGN
        # ----------------------------------------------------

        train_indices.extend(
            class_indices[
                :train_count
            ]
        )

        val_indices.extend(
            class_indices[
                train_count:
                train_count + val_count
            ]
        )

        test_indices.extend(
            class_indices[
                train_count + val_count:
            ]
        )

# ============================================================
# 8. CONVERT TO NUMPY
# ============================================================

train_indices = np.array(
    train_indices,
    dtype=np.int64
)

val_indices = np.array(
    val_indices,
    dtype=np.int64
)

test_indices = np.array(
    test_indices,
    dtype=np.int64
)

# ============================================================
# 9. SORT INDICES
# ============================================================

# Sorting gives deterministic file ordering.

train_indices.sort()
val_indices.sort()
test_indices.sort()

# ============================================================
# 10. BUILD DATASETS
# ============================================================

X_train = X[
    train_indices
]

y_train = y[
    train_indices
]

X_val = X[
    val_indices
]

y_val = y[
    val_indices
]

X_test = X[
    test_indices
]

y_test = y[
    test_indices
]

# ============================================================
# 11. PRINT SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("CLOSED-SET SPLIT")
print("=" * 60)

print(
    "\nTraining samples  :",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Testing samples   :",
    len(X_test)
)

print(
    "\nTraining classes  :",
    len(np.unique(y_train))
)

print(
    "Validation classes:",
    len(np.unique(y_val))
)

print(
    "Testing classes   :",
    len(np.unique(y_test))
)

# ============================================================
# 12. VERIFY ALL CLASSES IN TRAINING
# ============================================================

train_classes = set(
    np.unique(y_train)
)

all_classes = set(
    unique_classes
)

missing_training_classes = (
    all_classes
    - train_classes
)

print(
    "\nMissing training classes:",
    len(
        missing_training_classes
    )
)

if missing_training_classes:

    print(
        "❌ Missing classes:"
    )

    print(
        sorted(
            missing_training_classes
        )[:50]
    )

    raise ValueError(
        "Some vocabulary classes are "
        "missing from training."
    )

print(
    "✓ All 2667 classes present "
    "in training"
)

# ============================================================
# 13. VERIFY CLASS DISTRIBUTION
# ============================================================

train_counts = (
    pd.Series(y_train)
    .value_counts()
)

val_counts = (
    pd.Series(y_val)
    .value_counts()
)

test_counts = (
    pd.Series(y_test)
    .value_counts()
)

print("\n" + "=" * 60)
print("SAMPLE DISTRIBUTION")
print("=" * 60)

print(
    "\nTraining:"
)

print(
    "Minimum:",
    train_counts.min()
)

print(
    "Maximum:",
    train_counts.max()
)

print(
    "Average:",
    round(
        train_counts.mean(),
        2
    )
)

print(
    "\nValidation:"
)

print(
    "Minimum:",
    val_counts.min()
    if len(val_counts)
    else 0
)

print(
    "Maximum:",
    val_counts.max()
    if len(val_counts)
    else 0
)

print(
    "\nTesting:"
)

print(
    "Minimum:",
    test_counts.min()
    if len(test_counts)
    else 0
)

print(
    "Maximum:",
    test_counts.max()
    if len(test_counts)
    else 0
)

# ============================================================
# 14. VERIFY NO SAMPLE INDEX OVERLAP
# ============================================================

print("\nChecking sample overlap...")

train_set = set(
    train_indices
)

val_set = set(
    val_indices
)

test_set = set(
    test_indices
)

if train_set & val_set:
    raise ValueError(
        "Train/validation overlap detected."
    )

if train_set & test_set:
    raise ValueError(
        "Train/test overlap detected."
    )

if val_set & test_set:
    raise ValueError(
        "Validation/test overlap detected."
    )

print(
    "✓ No sample index overlap"
)

# ============================================================
# 15. CHECK TOTAL SAMPLES
# ============================================================

total_split_samples = (
    len(X_train)
    + len(X_val)
    + len(X_test)
)

print(
    "\nTotal split samples:",
    total_split_samples
)

print(
    "Original samples   :",
    len(X)
)

if total_split_samples != len(X):
    raise ValueError(
        "Some samples were lost during split."
    )

print(
    "✓ All samples accounted for"
)

# ============================================================
# 16. SAVE DATASETS
# ============================================================

print("\n" + "=" * 60)
print("SAVING CLOSED-SET DATA")
print("=" * 60)

np.save(
    INPUT_DIR /
    "X_closed_train.npy",
    X_train
)

np.save(
    INPUT_DIR /
    "y_closed_train.npy",
    y_train
)

np.save(
    INPUT_DIR /
    "X_closed_val.npy",
    X_val
)

np.save(
    INPUT_DIR /
    "y_closed_val.npy",
    y_val
)

np.save(
    INPUT_DIR /
    "X_closed_test.npy",
    X_test
)

np.save(
    INPUT_DIR /
    "y_closed_test.npy",
    y_test
)

# ============================================================
# 17. SAVE METADATA
# ============================================================

metadata.iloc[
    train_indices
].to_csv(
    INPUT_DIR /
    "closed_train_metadata.csv",
    index=False
)

metadata.iloc[
    val_indices
].to_csv(
    INPUT_DIR /
    "closed_val_metadata.csv",
    index=False
)

metadata.iloc[
    test_indices
].to_csv(
    INPUT_DIR /
    "closed_test_metadata.csv",
    index=False
)

# ============================================================
# 18. SAVE SPLIT SUMMARY
# ============================================================

split_summary = pd.DataFrame({

    "split": [
        "train",
        "validation",
        "test"
    ],

    "samples": [
        len(X_train),
        len(X_val),
        len(X_test)
    ],

    "classes": [
        len(
            np.unique(y_train)
        ),
        len(
            np.unique(y_val)
        ),
        len(
            np.unique(y_test)
        )
    ]

})

split_summary.to_csv(
    INPUT_DIR /
    "closed_set_split_summary.csv",
    index=False
)

# ============================================================
# 19. RELOAD VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("RELOAD VERIFICATION")
print("=" * 60)

X_train_check = np.load(
    INPUT_DIR /
    "X_closed_train.npy"
)

y_train_check = np.load(
    INPUT_DIR /
    "y_closed_train.npy"
)

X_val_check = np.load(
    INPUT_DIR /
    "X_closed_val.npy"
)

y_val_check = np.load(
    INPUT_DIR /
    "y_closed_val.npy"
)

X_test_check = np.load(
    INPUT_DIR /
    "X_closed_test.npy"
)

y_test_check = np.load(
    INPUT_DIR /
    "y_closed_test.npy"
)

if not np.array_equal(
    X_train,
    X_train_check
):
    raise ValueError(
        "Training X reload mismatch."
    )

if not np.array_equal(
    y_train,
    y_train_check
):
    raise ValueError(
        "Training y reload mismatch."
    )

if not np.array_equal(
    X_val,
    X_val_check
):
    raise ValueError(
        "Validation X reload mismatch."
    )

if not np.array_equal(
    y_val,
    y_val_check
):
    raise ValueError(
        "Validation y reload mismatch."
    )

if not np.array_equal(
    X_test,
    X_test_check
):
    raise ValueError(
        "Testing X reload mismatch."
    )

if not np.array_equal(
    y_test,
    y_test_check
):
    raise ValueError(
        "Testing y reload mismatch."
    )

print(
    "✓ All saved files verified"
)

# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SCRIPT 17 COMPLETE")
print("=" * 60)

print(
    "\nTraining X:",
    X_train.shape
)

print(
    "Training y:",
    y_train.shape
)

print(
    "\nValidation X:",
    X_val.shape
)

print(
    "Validation y:",
    y_val.shape
)

print(
    "\nTesting X:",
    X_test.shape
)

print(
    "Testing y:",
    y_test.shape
)

print(
    "\nClasses:",
    len(
        np.unique(y_train)
    )
)

print(
    "\n✓ All 2667 classes available "
    "for training"
)

print(
    "✓ No samples duplicated across splits"
)

print(
    "✓ No data files overwritten"
)

print(
    "\nREADY FOR MODEL TRAINING."
)

print(
    "\nIMPORTANT:"
)

print(
    "The validation/test windows originate "
    "from the same single recording per word."
)

print(
    "Therefore the resulting accuracy must "
    "not be presented as independent-recording "
    "or signer-independent accuracy."
)

print(
    "\nScript 17 finished successfully."
)