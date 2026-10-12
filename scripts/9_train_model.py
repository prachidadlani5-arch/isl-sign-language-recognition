import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# SCRIPT 9 - TRAIN SIGN LANGUAGE RANDOM FOREST
# ============================================================

TRAIN_FILE = "dataset/train.csv"
TEST_FILE = "dataset/test.csv"

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "sign_language_random_forest.pkl"

# ============================================================
# MEMORY-SAFE MODEL SETTINGS
# ============================================================

N_ESTIMATORS = 100
MAX_DEPTH = 20
MAX_SAMPLES = 0.60
MIN_SAMPLES_LEAF = 2

# IMPORTANT:
# We are NOT using all 12 CPU threads simultaneously.
# This reduces peak RAM usage.
N_JOBS = 4

RANDOM_STATE = 42

# ============================================================
# START
# ============================================================

print("=" * 60)
print("SCRIPT 9 - TRAIN SIGN LANGUAGE MODEL")
print("=" * 60)

start_time = time.time()

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(TRAIN_FILE)

print("✓ Training data loaded")
print("Training samples:", len(train_df))

# ============================================================
# LOAD TESTING DATA
# ============================================================

print("\nLoading testing data...")

test_df = pd.read_csv(TEST_FILE)

print("✓ Testing data loaded")
print("Testing samples:", len(test_df))

# ============================================================
# FIND LANDMARK FEATURES
# ============================================================

feature_columns = [
    column
    for column in train_df.columns
    if column.startswith("f")
]

print("\nNumber of features:", len(feature_columns))

# Safety check
if len(feature_columns) != 126:
    raise ValueError(
        f"Expected 126 features, "
        f"but found {len(feature_columns)}"
    )

# ============================================================
# PREPARE TRAINING DATA
# ============================================================

X_train = train_df[feature_columns]
y_train = train_df["label"]

# ============================================================
# PREPARE TESTING DATA
# ============================================================

X_test = test_df[feature_columns]
y_test = test_df["label"]

print("Training words:", y_train.nunique())
print("Testing words :", y_test.nunique())

# ============================================================
# CREATE RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("CREATING RANDOM FOREST")
print("=" * 60)

print("Number of trees :", N_ESTIMATORS)
print("Maximum depth   :", MAX_DEPTH)
print("Samples/tree    :", "60%")
print("Minimum leaf    :", MIN_SAMPLES_LEAF)
print("CPU workers     :", N_JOBS)
print("Random state    :", RANDOM_STATE)

model = RandomForestClassifier(

    # Number of trees
    n_estimators=N_ESTIMATORS,

    # Limit tree size
    max_depth=MAX_DEPTH,

    # Consider sqrt(number of features)
    # at each split
    max_features="sqrt",

    # Only use 60% of training samples
    # for each bootstrap tree
    max_samples=MAX_SAMPLES,

    # Prevent extremely tiny leaves
    min_samples_leaf=MIN_SAMPLES_LEAF,

    # Bootstrap sampling
    bootstrap=True,

    # Handle uneven word frequencies
    class_weight="balanced_subsample",

    # Reproducible results
    random_state=RANDOM_STATE,

    # Use only 4 workers to reduce RAM pressure
    n_jobs=N_JOBS,

    # Show progress
    verbose=1
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)

training_start = time.time()

try:

    model.fit(
        X_train,
        y_train
    )

except KeyboardInterrupt:

    print("\n")
    print("=" * 60)
    print("TRAINING INTERRUPTED")
    print("=" * 60)

    print("No incomplete model was saved.")
    print("Your existing dataset files are safe.")

    raise

except MemoryError:

    print("\n")
    print("=" * 60)
    print("MEMORY ERROR")
    print("=" * 60)

    print("The computer still does not have")
    print("enough available RAM for this model.")

    print("\nNo incomplete model was saved.")

    raise

training_time = time.time() - training_start

print("\n✓ TRAINING COMPLETE")

print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)

# ============================================================
# TEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("TESTING MODEL")
print("=" * 60)

prediction_start = time.time()

y_pred = model.predict(X_test)

prediction_time = time.time() - prediction_start

print(
    f"Prediction time: "
    f"{prediction_time / 60:.2f} minutes"
)

# ============================================================
# CALCULATE METRICS
# ============================================================

print("\nCalculating performance metrics...")

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# ============================================================
# DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model_package = {
    "model": model,
    "features": feature_columns,
    "classes": list(model.classes_)
}

joblib.dump(
    model_package,
    MODEL_FILE
)

print("✓ Model saved successfully")
print("Location:", MODEL_FILE)

# ============================================================
# FINAL SUMMARY
# ============================================================

total_time = time.time() - start_time

print("\n" + "=" * 60)
print("SCRIPT 9 COMPLETE")
print("=" * 60)

print(
    f"Total time      : "
    f"{total_time / 60:.2f} minutes"
)

print(
    f"Classes learned : "
    f"{len(model.classes_)}"
)

print(
    f"Trees           : "
    f"{N_ESTIMATORS}"
)

print(
    f"Maximum depth   : "
    f"{MAX_DEPTH}"
)

print(
    f"Features        : "
    f"{len(feature_columns)}"
)

print("\nModel:")
print(MODEL_FILE)

print("=" * 60)