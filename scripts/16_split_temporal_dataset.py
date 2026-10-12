import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SCRIPT 16 - PROPER TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

INPUT_DIR = Path("dataset/sequences")

X_FILE = INPUT_DIR / "X_temporal.npy"
Y_FILE = INPUT_DIR / "y_temporal.npy"
META_FILE = INPUT_DIR / "temporal_metadata.csv"
LABEL_FILE = INPUT_DIR / "temporal_label_map.csv"

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# START
# ============================================================

print("=" * 60)
print("SCRIPT 16 - PROPER TEMPORAL DATASET SPLIT")
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
# 2. VERIFY
# ============================================================

if len(X) != len(y):
    raise ValueError("X and y sizes do not match.")

if len(X) != len(metadata):
    raise ValueError("X and metadata sizes do not match.")

if X.ndim != 3:
    raise ValueError("X must be 3-dimensional.")

if X.shape[1] != 64:
    raise ValueError("Expected 64 frames.")

if X.shape[2] != 126:
    raise ValueError("Expected 126 features.")

print("\n✓ Dataset dimensions verified")

# ============================================================
# 3. IMPORTANT:
#    IDENTIFY ORIGINAL RECORDINGS
# ============================================================

print("\nIdentifying original recordings...")

# Each word corresponds to one original recording.
# Therefore ALL windows belonging to the same word
# must remain in the same split.

classes = metadata["label"].unique()

print("Original recordings/classes:", len(classes))

if len(classes) != 2667:
    raise ValueError(
        f"Expected 2667 classes, found {len(classes)}"
    )

# ============================================================
# 4. SHUFFLE ORIGINAL RECORDINGS
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

shuffled_classes = np.array(
    classes,
    dtype=object
)

rng.shuffle(shuffled_classes)

# ============================================================
# 5. CALCULATE SPLIT
# ============================================================

n_classes = len(shuffled_classes)

train_count = int(
    n_classes * TRAIN_RATIO
)

val_count = int(
    n_classes * VAL_RATIO
)

test_count = (
    n_classes
    - train_count
    - val_count
)

train_classes = shuffled_classes[
    :train_count
]

val_classes = shuffled_classes[
    train_count:
    train_count + val_count
]

test_classes = shuffled_classes[
    train_count + val_count:
]

print("\n" + "=" * 60)
print("CLASS SPLIT")
print("=" * 60)

print("\nTraining classes  :", len(train_classes))
print("Validation classes:", len(val_classes))
print("Testing classes   :", len(test_classes))

# ============================================================
# 6. CHECK CLASS DISJOINTNESS
# ============================================================

train_set = set(train_classes)
val_set = set(val_classes)
test_set = set(test_classes)

if train_set & val_set:
    raise ValueError(
        "Training and validation classes overlap."
    )

if train_set & test_set:
    raise ValueError(
        "Training and testing classes overlap."
    )

if val_set & test_set:
    raise ValueError(
        "Validation and testing classes overlap."
    )

print("\n✓ Train / validation / test classes are disjoint")

# ============================================================
# 7. CREATE SAMPLE INDICES
# ============================================================

train_indices = []
val_indices = []
test_indices = []

for i, label in enumerate(
    metadata["label"]
):

    if label in train_set:

        train_indices.append(i)

    elif label in val_set:

        val_indices.append(i)

    elif label in test_set:

        test_indices.append(i)

    else:

        raise ValueError(
            f"Unknown label: {label}"
        )

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
# 8. CREATE SPLITS
# ============================================================

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

X_test = X[test_indices]
y_test = y[test_indices]

# ============================================================
# 9. DISPLAY SAMPLE COUNTS
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL SAMPLE SPLIT")
print("=" * 60)

print("\nTraining samples  :", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples   :", len(X_test))

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
# 10. IMPORTANT WARNING
# ============================================================

print("\n" + "=" * 60)
print("IMPORTANT EVALUATION NOTE")
print("=" * 60)

print("""
This split keeps every original recording together.

Therefore:
✓ No overlapping windows cross train/test.
✓ No recording appears in multiple splits.
✓ Test accuracy will not be inflated by temporal leakage.

However, because there is only ONE original recording
for each vocabulary word, the validation and test sets
contain classes that the model has NEVER seen during training.

That means ordinary classification accuracy on these
splits is NOT a valid closed-set 2,667-class recognition
metric.

The split is useful for evaluating generalization,
but a proper closed-set benchmark requires multiple
independent recordings per word.
""")

# ============================================================
# 11. SAVE SPLITS
# ============================================================

print("\n" + "=" * 60)
print("SAVING DATA")
print("=" * 60)

np.save(
    INPUT_DIR / "X_train_temporal.npy",
    X_train
)

np.save(
    INPUT_DIR / "y_train_temporal.npy",
    y_train
)

np.save(
    INPUT_DIR / "X_val_temporal.npy",
    X_val
)

np.save(
    INPUT_DIR / "y_val_temporal.npy",
    y_val
)

np.save(
    INPUT_DIR / "X_test_temporal.npy",
    X_test
)

np.save(
    INPUT_DIR / "y_test_temporal.npy",
    y_test
)

# ============================================================
# 12. SAVE METADATA SPLITS
# ============================================================

metadata.iloc[
    train_indices
].to_csv(
    INPUT_DIR / "train_temporal_metadata.csv",
    index=False
)

metadata.iloc[
    val_indices
].to_csv(
    INPUT_DIR / "val_temporal_metadata.csv",
    index=False
)

metadata.iloc[
    test_indices
].to_csv(
    INPUT_DIR / "test_temporal_metadata.csv",
    index=False
)

# ============================================================
# 13. SAVE CLASS SPLIT INFORMATION
# ============================================================

train_class_df = pd.DataFrame({
    "label": sorted(train_set),
    "split": "train"
})

val_class_df = pd.DataFrame({
    "label": sorted(val_set),
    "split": "validation"
})

test_class_df = pd.DataFrame({
    "label": sorted(test_set),
    "split": "test"
})

class_split = pd.concat(
    [
        train_class_df,
        val_class_df,
        test_class_df
    ],
    ignore_index=True
)

class_split.to_csv(
    INPUT_DIR / "class_split.csv",
    index=False
)

# ============================================================
# 14. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

total_samples = (
    len(X_train)
    + len(X_val)
    + len(X_test)
)

print(
    "\nTotal samples:",
    total_samples
)

print(
    "Original samples:",
    len(X)
)

if total_samples != len(X):

    raise ValueError(
        "Split sample counts do not add up."
    )

print(
    "✓ All samples accounted for"
)

# ------------------------------------------------------------
# Shape checks
# ------------------------------------------------------------

for name, data in [
    ("X_train", X_train),
    ("X_val", X_val),
    ("X_test", X_test)
]:

    if data.ndim != 3:
        raise ValueError(
            f"{name} has incorrect dimensions."
        )

    if data.shape[1] != 64:
        raise ValueError(
            f"{name} does not have 64 frames."
        )

    if data.shape[2] != 126:
        raise ValueError(
            f"{name} does not have 126 features."
        )

print("✓ All sequence shapes valid")

# ============================================================
# 15. RELOAD CHECK
# ============================================================

print("\nReloading saved files...")

X_train_check = np.load(
    INPUT_DIR / "X_train_temporal.npy"
)

X_val_check = np.load(
    INPUT_DIR / "X_val_temporal.npy"
)

X_test_check = np.load(
    INPUT_DIR / "X_test_temporal.npy"
)

if not np.array_equal(
    X_train,
    X_train_check
):
    raise ValueError(
        "Saved training data mismatch."
    )

if not np.array_equal(
    X_val,
    X_val_check
):
    raise ValueError(
        "Saved validation data mismatch."
    )

if not np.array_equal(
    X_test,
    X_test_check
):
    raise ValueError(
        "Saved testing data mismatch."
    )

print(
    "✓ Saved files verified"
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("SCRIPT 16 COMPLETE")
print("=" * 60)

print("\nCreated:")

print(
    "dataset/sequences/X_train_temporal.npy"
)

print(
    "dataset/sequences/y_train_temporal.npy"
)

print(
    "dataset/sequences/X_val_temporal.npy"
)

print(
    "dataset/sequences/y_val_temporal.npy"
)

print(
    "dataset/sequences/X_test_temporal.npy"
)

print(
    "dataset/sequences/y_test_temporal.npy"
)

print(
    "dataset/sequences/class_split.csv"
)

print("\n✓ No temporal leakage")
print("✓ No model trained")
print("✓ Data verified")
print("\nREADY FOR MODEL DEVELOPMENT")