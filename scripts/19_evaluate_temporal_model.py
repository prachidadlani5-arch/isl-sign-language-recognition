import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# SCRIPT 19 - DETAILED TEMPORAL MODEL EVALUATION
# ============================================================

print("=" * 70)
print("SCRIPT 19 - DETAILED TEMPORAL MODEL EVALUATION")
print("=" * 70)

# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 64
FEATURES = 126
NUM_CLASSES = 2667

BATCH_SIZE = 16

DATA_DIR = Path("dataset/sequences")

MODEL_PATH = (
    Path("models/sign_language_temporal")
    / "best_closed_temporal_model.keras"
)

EVALUATION_DIR = (
    Path("models/sign_language_temporal")
    / "evaluation"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 1. CUSTOM TEMPORAL ATTENTION LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="SignLanguage"
)
class TemporalAttention(
    tf.keras.layers.Layer
):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.score = tf.keras.layers.Dense(
            1,
            activation="tanh"
        )

    def build(self, input_shape):

        # input_shape:
        # (batch, time, features)

        self.score.build(
            (
                input_shape[0],
                input_shape[-1]
            )
        )

        super().build(input_shape)

    def call(self, inputs):

        # inputs:
        # (batch, time, features)

        scores = self.score(inputs)

        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        context = tf.reduce_sum(
            inputs * weights,
            axis=1
        )

        return context

    def get_config(self):

        config = super().get_config()

        return config


# ============================================================
# 2. ENVIRONMENT
# ============================================================

print("\nTensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected")
else:
    print("Running on CPU")

# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

X_test = np.load(
    DATA_DIR / "X_closed_test.npy"
)

y_test = np.load(
    DATA_DIR / "y_closed_test.npy"
)

metadata = pd.read_csv(
    DATA_DIR / "closed_test_metadata.csv"
)

label_map = pd.read_csv(
    DATA_DIR / "temporal_label_map.csv"
)

print("✓ X_test loaded")
print("✓ y_test loaded")
print("✓ Metadata loaded")
print("✓ Label map loaded")

print("\nX_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("Metadata:", len(metadata))
print("Classes:", len(label_map))

# ============================================================
# 4. VERIFY TEST DATA
# ============================================================

print("\nChecking data...")

if X_test.ndim != 3:
    raise ValueError(
        f"X_test must be 3-dimensional. "
        f"Found {X_test.ndim} dimensions."
    )

if X_test.shape[1] != SEQUENCE_LENGTH:
    raise ValueError(
        f"Expected {SEQUENCE_LENGTH} frames, "
        f"found {X_test.shape[1]}."
    )

if X_test.shape[2] != FEATURES:
    raise ValueError(
        f"Expected {FEATURES} features, "
        f"found {X_test.shape[2]}."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test sizes do not match."
    )

if len(X_test) != len(metadata):
    raise ValueError(
        "X_test and metadata sizes do not match."
    )

if np.isnan(X_test).any():
    raise ValueError(
        "NaN values found in X_test."
    )

if np.isinf(X_test).any():
    raise ValueError(
        "Infinite values found in X_test."
    )

if np.min(y_test) < 0:
    raise ValueError(
        "Negative class IDs found."
    )

if np.max(y_test) >= NUM_CLASSES:
    raise ValueError(
        "Invalid class IDs found."
    )

print("✓ Test data verified")

# ============================================================
# 5. CREATE LABEL DICTIONARY
# ============================================================

print("\nCreating label dictionary...")

id_to_label = {}

for _, row in label_map.iterrows():

    label_id = int(
        row["label_id"]
    )

    label = str(
        row["label"]
    )

    id_to_label[label_id] = label

print(
    "✓ Labels loaded:",
    len(id_to_label)
)

# ============================================================
# 6. LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

print("\nModel:", MODEL_PATH)

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

# ------------------------------------------------------------
# IMPORTANT:
#
# TemporalAttention is a custom layer.
#
# custom_objects tells Keras how to reconstruct it.
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "TemporalAttention": TemporalAttention,
        "SignLanguage>TemporalAttention": TemporalAttention
    },
    compile=True
)

print("✓ Best model loaded")

print(
    "\nModel input shape:",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)

# ============================================================
# 7. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

test_results = model.evaluate(

    X_test,
    y_test,

    batch_size=BATCH_SIZE,

    verbose=1,

    return_dict=True
)

print("\nOfficial test results:")

for key, value in test_results.items():

    print(
        f"{key}: {value:.6f}"
    )

# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

predictions = model.predict(

    X_test,

    batch_size=BATCH_SIZE,

    verbose=1
)

print(
    "\nPrediction shape:",
    predictions.shape
)

if predictions.shape != (
    len(X_test),
    NUM_CLASSES
):

    raise ValueError(
        "Unexpected prediction shape."
    )

print("✓ Predictions generated")

# ============================================================
# 9. TOP-1 PREDICTIONS
# ============================================================

predicted_ids = np.argmax(
    predictions,
    axis=1
)

correct_mask = (
    predicted_ids == y_test
)

correct_count = int(
    np.sum(correct_mask)
)

incorrect_count = int(
    np.sum(~correct_mask)
)

top1_accuracy = (
    correct_count /
    len(y_test)
)

print("\n" + "=" * 70)
print("TOP-1 RESULTS")
print("=" * 70)

print(
    "\nCorrect predictions:",
    correct_count
)

print(
    "Incorrect predictions:",
    incorrect_count
)

print(
    "Accuracy:",
    f"{top1_accuracy * 100:.2f}%"
)

# ============================================================
# 10. TOP-5 ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TOP-5 ANALYSIS")
print("=" * 70)

top5_ids = np.argsort(
    predictions,
    axis=1
)[:, -5:]

top5_correct_mask = np.array([

    true_id in predicted_list

    for true_id, predicted_list
    in zip(y_test, top5_ids)

])

top5_correct_count = int(
    np.sum(top5_correct_mask)
)

top5_accuracy = (
    top5_correct_count /
    len(y_test)
)

print(
    "\nTop-5 correct:",
    top5_correct_count
)

print(
    "Top-5 accuracy:",
    f"{top5_accuracy * 100:.2f}%"
)

# ============================================================
# 11. CREATE PREDICTION TABLE
# ============================================================

print("\n" + "=" * 70)
print("CREATING PREDICTION TABLE")
print("=" * 70)

prediction_table = pd.DataFrame({

    "sample_index":
        np.arange(len(y_test)),

    "true_id":
        y_test.astype(int),

    "predicted_id":
        predicted_ids.astype(int),

    "correct":
        correct_mask,

    "confidence":
        np.max(
            predictions,
            axis=1
        )

})

prediction_table["true_label"] = (
    prediction_table["true_id"]
    .map(id_to_label)
)

prediction_table["predicted_label"] = (
    prediction_table["predicted_id"]
    .map(id_to_label)
)

# ------------------------------------------------------------
# Add recording metadata
# ------------------------------------------------------------

print("Adding recording metadata...")

for column in metadata.columns:

    if column not in prediction_table.columns:

        prediction_table[column] = (
            metadata[column].values
        )

print("✓ Metadata added")

prediction_file = (
    EVALUATION_DIR /
    "test_predictions.csv"
)

prediction_table.to_csv(
    prediction_file,
    index=False
)

print(
    "\n✓ Saved:",
    prediction_file
)

# ============================================================
# 12. PER-CLASS PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

class_records = []

for class_id in sorted(
    np.unique(y_test)
):

    class_mask = (
        y_test == class_id
    )

    class_predictions = (
        predicted_ids[class_mask]
    )

    class_true = (
        y_test[class_mask]
    )

    class_correct = (
        class_predictions
        == class_true
    )

    samples = int(
        np.sum(class_mask)
    )

    correct = int(
        np.sum(class_correct)
    )

    accuracy = (
        correct / samples
        if samples > 0
        else 0.0
    )

    # --------------------------------------------------------
    # TOP-5 CLASS ACCURACY
    # --------------------------------------------------------

    class_top5 = (
        top5_correct_mask[class_mask]
    )

    top5_class_accuracy = (
        float(np.mean(class_top5))
        if len(class_top5) > 0
        else 0.0
    )

    class_records.append({

        "class_id":
            int(class_id),

        "label":
            id_to_label.get(
                int(class_id),
                "UNKNOWN"
            ),

        "samples":
            samples,

        "correct":
            correct,

        "accuracy":
            accuracy,

        "top5_accuracy":
            top5_class_accuracy

    })

per_class_df = pd.DataFrame(
    class_records
)

per_class_file = (
    EVALUATION_DIR /
    "per_class_performance.csv"
)

per_class_df.to_csv(
    per_class_file,
    index=False
)

print(
    "\n✓ Saved:",
    per_class_file
)

# ============================================================
# 13. WORST PERFORMING CLASSES
# ============================================================

print("\n" + "=" * 70)
print("WORST-PERFORMING CLASSES")
print("=" * 70)

worst_classes = (
    per_class_df
    .sort_values(
        [
            "accuracy",
            "top5_accuracy"
        ],
        ascending=[
            True,
            True
        ]
    )
    .head(30)
)

print("\nBottom 30 classes:\n")

print(
    worst_classes.to_string(
        index=False
    )
)

# ============================================================
# 14. BEST PERFORMING CLASSES
# ============================================================

print("\n" + "=" * 70)
print("BEST-PERFORMING CLASSES")
print("=" * 70)

best_classes = (
    per_class_df
    .sort_values(
        [
            "accuracy",
            "top5_accuracy"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(30)
)

print("\nTop 30 classes:\n")

print(
    best_classes.to_string(
        index=False
    )
)

# ============================================================
# 15. MOST CONFUSED CLASS PAIRS
# ============================================================

print("\n" + "=" * 70)
print("MOST CONFUSED CLASS PAIRS")
print("=" * 70)

wrong_predictions = (
    prediction_table[
        ~prediction_table["correct"]
    ]
)

confusion_records = []

for _, row in wrong_predictions.iterrows():

    confusion_records.append({

        "true_id":
            int(row["true_id"]),

        "true_label":
            row["true_label"],

        "predicted_id":
            int(row["predicted_id"]),

        "predicted_label":
            row["predicted_label"],

        "count":
            1

    })

if confusion_records:

    confusion_df = pd.DataFrame(
        confusion_records
    )

    confusion_df = (
        confusion_df
        .groupby(
            [
                "true_id",
                "true_label",
                "predicted_id",
                "predicted_label"
            ],
            as_index=False
        )
        .agg(
            count=("count", "sum")
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

else:

    confusion_df = pd.DataFrame(
        columns=[
            "true_id",
            "true_label",
            "predicted_id",
            "predicted_label",
            "count"
        ]
    )

confusion_file = (
    EVALUATION_DIR /
    "most_confused_pairs.csv"
)

confusion_df.to_csv(
    confusion_file,
    index=False
)

print(
    "\n✓ Saved:",
    confusion_file
)

print(
    "\nTop 30 confused pairs:\n"
)

print(
    confusion_df
    .head(30)
    .to_string(
        index=False
    )
)

# ============================================================
# 16. CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)

confidence = np.max(
    predictions,
    axis=1
)

correct_confidence = (
    confidence[correct_mask]
)

incorrect_confidence = (
    confidence[~correct_mask]
)

print(
    "\nAverage confidence - correct:",
    f"{np.mean(correct_confidence):.4f}"
)

if len(incorrect_confidence) > 0:

    print(
        "Average confidence - incorrect:",
        f"{np.mean(incorrect_confidence):.4f}"
    )

else:

    print(
        "Average confidence - incorrect: 0.0000"
    )

print(
    "\nMinimum confidence:",
    f"{np.min(confidence):.4f}"
)

print(
    "Maximum confidence:",
    f"{np.max(confidence):.4f}"
)

print(
    "Median confidence:",
    f"{np.median(confidence):.4f}"
)

# ============================================================
# 17. HIGH-CONFIDENCE WRONG PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE WRONG PREDICTIONS")
print("=" * 70)

wrong_table = prediction_table[
    ~prediction_table["correct"]
].copy()

high_confidence_wrong = (
    wrong_table
    .sort_values(
        "confidence",
        ascending=False
    )
    .head(20)
)

high_confidence_file = (
    EVALUATION_DIR /
    "high_confidence_wrong_predictions.csv"
)

high_confidence_wrong.to_csv(
    high_confidence_file,
    index=False
)

print(
    "\n✓ Saved:",
    high_confidence_file
)

print(
    "\nTop 20 high-confidence mistakes:\n"
)

print(
    high_confidence_wrong[
        [
            "sample_index",
            "true_id",
            "true_label",
            "predicted_id",
            "predicted_label",
            "confidence"
        ]
    ].to_string(
        index=False
    )
)

# ============================================================
# 18. ZERO-ACCURACY CLASSES
# ============================================================

zero_accuracy_classes = (
    per_class_df[
        per_class_df["accuracy"] == 0
    ]
)

print("\n" + "=" * 70)
print("ZERO-ACCURACY CLASSES")
print("=" * 70)

print(
    "\nZero-accuracy classes:",
    len(zero_accuracy_classes)
)

# ============================================================
# 19. EVALUATION SUMMARY
# ============================================================

evaluation_summary = pd.DataFrame({

    "metric": [

        "test_samples",

        "classes",

        "top1_accuracy",

        "top5_accuracy",

        "correct_predictions",

        "incorrect_predictions",

        "classes_evaluated",

        "zero_accuracy_classes",

        "average_confidence_correct",

        "average_confidence_incorrect",

        "minimum_confidence",

        "maximum_confidence",

        "median_confidence"

    ],

    "value": [

        len(y_test),

        NUM_CLASSES,

        top1_accuracy,

        top5_accuracy,

        correct_count,

        incorrect_count,

        len(per_class_df),

        len(zero_accuracy_classes),

        float(
            np.mean(correct_confidence)
        ),

        float(
            np.mean(incorrect_confidence)
        )
        if len(incorrect_confidence)
        else 0.0,

        float(
            np.min(confidence)
        ),

        float(
            np.max(confidence)
        ),

        float(
            np.median(confidence)
        )

    ]

})

summary_file = (
    EVALUATION_DIR /
    "evaluation_summary.csv"
)

evaluation_summary.to_csv(
    summary_file,
    index=False
)

print(
    "\n✓ Saved:",
    summary_file
)

# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SCRIPT 19 COMPLETE")
print("=" * 70)

print(
    "\nTest samples:",
    len(y_test)
)

print(
    "Classes:",
    NUM_CLASSES
)

print(
    "\nTop-1 accuracy:",
    f"{top1_accuracy * 100:.2f}%"
)

print(
    "Top-5 accuracy:",
    f"{top5_accuracy * 100:.2f}%"
)

print(
    "\nCorrect:",
    correct_count
)

print(
    "Incorrect:",
    incorrect_count
)

print(
    "\nClasses evaluated:",
    len(per_class_df)
)

print(
    "Zero-accuracy classes:",
    len(zero_accuracy_classes)
)

print(
    "\nEvaluation files saved to:"
)

print(
    EVALUATION_DIR
)

print(
    "\nIMPORTANT:"
)

print(
    "These are CLOSED-SET temporal results."
)

print(
    "They must NOT be described as "
    "signer-independent or independent-recording accuracy."
)

print(
    "\n✓ Script 19 finished successfully."
)