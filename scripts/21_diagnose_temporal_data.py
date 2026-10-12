import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# SCRIPT 21 - TEMPORAL DATA DIAGNOSTIC
# ============================================================

print("=" * 70)
print("SCRIPT 21 - TEMPORAL DATA DIAGNOSTIC")
print("=" * 70)


BASE = Path("dataset/sequences")


# ============================================================
# 1. FILES
# ============================================================

files = [
    "X_closed_train.npy",
    "y_closed_train.npy",
    "X_closed_val.npy",
    "y_closed_val.npy",
    "X_closed_test.npy",
    "y_closed_test.npy",
    "X_train_temporal.npy",
    "y_train_temporal.npy",
    "X_val_temporal.npy",
    "y_val_temporal.npy",
    "X_test_temporal.npy",
    "y_test_temporal.npy",
    "X_temporal.npy",
    "y_temporal.npy",
]


# ============================================================
# 2. CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FILES")
print("=" * 70)

for filename in files:

    path = BASE / filename

    if path.exists():
        print(f"✓ {filename}")
    else:
        print(f"✗ MISSING: {filename}")


# ============================================================
# 3. LOAD ARRAYS
# ============================================================

print("\n" + "=" * 70)
print("LOADING ARRAYS")
print("=" * 70)


arrays = {}


for filename in files:

    path = BASE / filename

    if not path.exists():
        continue

    try:

        arrays[filename] = np.load(
            path,
            mmap_mode="r"
        )

        print(
            f"{filename:25s} "
            f"shape={arrays[filename].shape} "
            f"dtype={arrays[filename].dtype}"
        )

    except Exception as e:

        print(
            f"ERROR loading {filename}: {e}"
        )


# ============================================================
# 4. CLOSED TEMPORAL DATA
# ============================================================

print("\n" + "=" * 70)
print("CLOSED TEMPORAL DATA")
print("=" * 70)


closed_files = [
    "X_closed_train.npy",
    "X_closed_val.npy",
    "X_closed_test.npy",
    "y_closed_train.npy",
    "y_closed_val.npy",
    "y_closed_test.npy",
]


for filename in closed_files:

    if filename not in arrays:
        continue

    arr = arrays[filename]

    print(
        f"{filename:25s} "
        f"shape={arr.shape} "
        f"dtype={arr.dtype}"
    )


# ============================================================
# 5. VERIFY MODEL INPUT SHAPE
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING MODEL INPUT SHAPE")
print("=" * 70)


X_train = arrays.get(
    "X_closed_train.npy"
)

X_val = arrays.get(
    "X_closed_val.npy"
)

X_test = arrays.get(
    "X_closed_test.npy"
)


for name, X in [
    ("TRAIN", X_train),
    ("VALIDATION", X_val),
    ("TEST", X_test),
]:

    if X is None:
        continue

    print(
        f"{name}: {X.shape}"
    )

    if len(X.shape) == 3:

        print(
            f"  Frames   : {X.shape[1]}"
        )

        print(
            f"  Features : {X.shape[2]}"
        )

        if X.shape[1] == 64 and X.shape[2] == 126:

            print(
                "  ✓ Correct: (64, 126)"
            )

        else:

            print(
                "  ⚠️ Unexpected shape!"
            )


# ============================================================
# 6. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)


y_train = arrays.get(
    "y_closed_train.npy"
)

y_val = arrays.get(
    "y_closed_val.npy"
)

y_test = arrays.get(
    "y_closed_test.npy"
)


def analyze_labels(
    name,
    y
):

    if y is None:
        return

    y_values = np.asarray(y)

    print(
        f"\n{name}"
    )

    print(
        "Shape:",
        y_values.shape
    )

    unique, counts = np.unique(
        y_values,
        return_counts=True
    )

    print(
        "Number of classes:",
        len(unique)
    )

    print(
        "Minimum samples/class:",
        counts.min()
    )

    print(
        "Maximum samples/class:",
        counts.max()
    )

    print(
        "Average samples/class:",
        round(
            counts.mean(),
            2
        )
    )

    print(
        "Median samples/class:",
        round(
            np.median(counts),
            2
        )
    )

    zero_classes = (
        2667 -
        len(unique)
    )

    print(
        "Classes absent:",
        zero_classes
    )

    print(
        "Most frequent class:",
        counts.max()
    )

    print(
        "Least frequent class:",
        counts.min()
    )


analyze_labels(
    "TRAIN",
    y_train
)

analyze_labels(
    "VALIDATION",
    y_val
)

analyze_labels(
    "TEST",
    y_test
)


# ============================================================
# 7. FEATURE VALUES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALUE ANALYSIS")
print("=" * 70)


if X_train is not None:

    # Sample to avoid loading everything
    sample_count = min(
        1000,
        X_train.shape[0]
    )

    sample = np.asarray(
        X_train[
            :sample_count
        ]
    )

    print(
        "Sample shape:",
        sample.shape
    )

    print(
        "Minimum:",
        float(
            np.min(sample)
        )
    )

    print(
        "Maximum:",
        float(
            np.max(sample)
        )
    )

    print(
        "Mean:",
        float(
            np.mean(sample)
        )
    )

    print(
        "Standard deviation:",
        float(
            np.std(sample)
        )
    )

    print(
        "NaN count:",
        int(
            np.isnan(sample).sum()
        )
    )

    print(
        "Inf count:",
        int(
            np.isinf(sample).sum()
        )
    )


# ============================================================
# 8. LABEL MAP
# ============================================================

print("\n" + "=" * 70)
print("LABEL MAP")
print("=" * 70)


label_map = BASE / "temporal_label_map.csv"


if label_map.exists():

    df = pd.read_csv(
        label_map
    )

    print(
        "Columns:",
        list(df.columns)
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "\nFirst 10 labels:"
    )

    print(
        df.head(10).to_string(
            index=False
        )
    )

else:

    print(
        "Label map not found."
    )


# ============================================================
# 9. METADATA
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL METADATA")
print("=" * 70)


metadata_files = [
    "temporal_metadata.csv",
    "train_temporal_metadata.csv",
    "val_temporal_metadata.csv",
    "test_temporal_metadata.csv",
]


for filename in metadata_files:

    path = BASE / filename

    if not path.exists():

        print(
            f"✗ {filename}"
        )

        continue

    try:

        df = pd.read_csv(
            path
        )

        print(
            f"\n{filename}"
        )

        print(
            "Rows:",
            len(df)
        )

        print(
            "Columns:",
            list(df.columns)
        )

        print(
            df.head(3).to_string(
                index=False
            )
        )

    except Exception as e:

        print(
            f"Error reading {filename}: {e}"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)