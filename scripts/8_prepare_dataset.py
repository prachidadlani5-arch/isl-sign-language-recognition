import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "dataset/landmarks.csv"

TRAIN_FILE = "dataset/train.csv"
TEST_FILE = "dataset/test.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("SCRIPT 8 - PREPARE TRAINING DATA")
print("=" * 60)

print("\nLoading landmarks.csv...")

df = pd.read_csv(INPUT_FILE)

print("✓ Dataset loaded")

print("Total samples :", len(df))
print("Total words   :", df["label"].nunique())

# ============================================================
# SEPARATE FEATURES AND LABEL
# ============================================================

feature_columns = [
    column for column in df.columns
    if column.startswith("f")
]

X = df[feature_columns]
y = df["label"]

print("\nFeatures :", len(feature_columns))
print("Labels   :", y.nunique())

# ============================================================
# CHECK WORDS WITH ONLY ONE SAMPLE
# ============================================================

word_counts = y.value_counts()

single_sample_words = word_counts[word_counts < 2]

if len(single_sample_words) > 0:

    print("\n⚠ Words with fewer than 2 samples:")
    print(single_sample_words)

    raise ValueError(
        "Some words do not have enough samples for train/test splitting."
    )

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ============================================================
# CREATE DATAFRAMES
# ============================================================

train_df = X_train.copy()
train_df["label"] = y_train.values

test_df = X_test.copy()
test_df["label"] = y_test.values

# Put label first
train_df = train_df[["label"] + feature_columns]
test_df = test_df[["label"] + feature_columns]

# ============================================================
# SAVE
# ============================================================

Path("dataset").mkdir(exist_ok=True)

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print("\nTraining samples :", len(train_df))
print("Testing samples  :", len(test_df))

print("Training words   :", train_df["label"].nunique())
print("Testing words    :", test_df["label"].nunique())

print("\nTrain file:")
print(TRAIN_FILE)

print("\nTest file:")
print(TEST_FILE)

# ============================================================
# VERIFY EVERY WORD EXISTS IN BOTH SETS
# ============================================================

train_words = set(train_df["label"].unique())
test_words = set(test_df["label"].unique())

missing_from_train = sorted(set(y.unique()) - train_words)
missing_from_test = sorted(set(y.unique()) - test_words)

print("\n" + "=" * 60)
print("WORD DISTRIBUTION CHECK")
print("=" * 60)

if not missing_from_train:
    print("✓ Every word exists in training data")
else:
    print("✗ Words missing from training:")
    print(missing_from_train)

if not missing_from_test:
    print("✓ Every word exists in testing data")
else:
    print("✗ Words missing from testing:")
    print(missing_from_test)

# ============================================================
# FINAL CHECK
# ============================================================

if (
    train_df["label"].nunique() == 2667
    and test_df["label"].nunique() == 2667
    and not missing_from_train
    and not missing_from_test
):

    print("\n✓ ALL CHECKS PASSED")

else:

    print("\n⚠ CHECKS REQUIRE ATTENTION")

print("=" * 60)