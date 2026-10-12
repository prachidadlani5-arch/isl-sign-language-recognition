import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# SCRIPT 18 - TRAIN CLOSED-SET TEMPORAL SIGN LANGUAGE MODEL
# ============================================================

print("=" * 60)
print("SCRIPT 18 - TRAIN CLOSED-SET TEMPORAL SIGN LANGUAGE MODEL")
print("=" * 60)

# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 64
FEATURES = 126
NUM_CLASSES = 2667

BATCH_SIZE = 16
EPOCHS = 80

LEARNING_RATE = 0.0005

RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_DIR = Path("dataset/sequences")
MODEL_DIR = Path("models/sign_language_temporal")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# ENVIRONMENT
# ============================================================

print("\nTensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("Device: GPU")
else:
    print("Device: CPU")

# ============================================================
# CONFIGURATION
# ============================================================

print("\nConfiguration:")
print("Frames          :", SEQUENCE_LENGTH)
print("Features/frame  :", FEATURES)
print("Classes         :", NUM_CLASSES)
print("Batch size      :", BATCH_SIZE)
print("Maximum epochs  :", EPOCHS)
print("Learning rate   :", LEARNING_RATE)

# ============================================================
# 1. LOAD CLOSED-SET DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING CLOSED-SET DATA")
print("=" * 60)

X_train = np.load(
    DATA_DIR / "X_closed_train.npy"
)

y_train = np.load(
    DATA_DIR / "y_closed_train.npy"
)

X_val = np.load(
    DATA_DIR / "X_closed_val.npy"
)

y_val = np.load(
    DATA_DIR / "y_closed_val.npy"
)

X_test = np.load(
    DATA_DIR / "X_closed_test.npy"
)

y_test = np.load(
    DATA_DIR / "y_closed_test.npy"
)

print("✓ Training data loaded")
print("✓ Validation data loaded")
print("✓ Testing data loaded")

# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nTraining X :", X_train.shape)
print("Training y :", y_train.shape)

print("\nValidation X:", X_val.shape)
print("Validation y:", y_val.shape)

print("\nTesting X  :", X_test.shape)
print("Testing y  :", y_test.shape)

# ============================================================
# 3. VERIFY SHAPES
# ============================================================

print("\nChecking shapes...")

expected_train_shape = (
    None,
    SEQUENCE_LENGTH,
    FEATURES
)

for name, X in [
    ("Training", X_train),
    ("Validation", X_val),
    ("Testing", X_test)
]:

    if X.ndim != 3:
        raise ValueError(
            f"{name} X must be 3-dimensional."
        )

    if X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"{name} has {X.shape[1]} frames."
        )

    if X.shape[2] != FEATURES:
        raise ValueError(
            f"{name} has {X.shape[2]} features."
        )

print("✓ Training shape valid")
print("✓ Validation shape valid")
print("✓ Testing shape valid")

# ============================================================
# 4. VERIFY LABELS
# ============================================================

print("\nChecking labels...")

for name, y in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test)
]:

    if np.min(y) < 0:
        raise ValueError(
            f"{name} contains negative labels."
        )

    if np.max(y) >= NUM_CLASSES:
        raise ValueError(
            f"{name} contains invalid class IDs."
        )

print("✓ Training labels valid")
print("✓ Validation labels valid")
print("✓ Testing labels valid")

# ============================================================
# 5. CLASS COVERAGE
# ============================================================

train_classes = np.unique(y_train)
val_classes = np.unique(y_val)
test_classes = np.unique(y_test)

print("\nTraining classes :", len(train_classes))
print("Validation classes:", len(val_classes))
print("Testing classes  :", len(test_classes))

if len(train_classes) != NUM_CLASSES:
    raise ValueError(
        f"Training contains {len(train_classes)} classes, "
        f"expected {NUM_CLASSES}."
    )

print("✓ All 2667 classes available during training")

# ============================================================
# 6. CHECK VALUES
# ============================================================

print("\nChecking numerical values...")

for name, X in [
    ("Training", X_train),
    ("Validation", X_val),
    ("Testing", X_test)
]:

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    print(
        f"{name}: NaN={nan_count}, Infinite={inf_count}"
    )

    if nan_count != 0:
        raise ValueError(
            f"{name} contains NaN values."
        )

    if inf_count != 0:
        raise ValueError(
            f"{name} contains infinite values."
        )

print("✓ All numerical values valid")

# ============================================================
# 7. FLOAT32
# ============================================================

print("\nConverting arrays to float32...")

X_train = X_train.astype(
    np.float32
)

X_val = X_val.astype(
    np.float32
)

X_test = X_test.astype(
    np.float32
)

print("✓ float32 conversion complete")

# ============================================================
# 8. NORMALIZATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING NORMALIZATION")
print("=" * 60)

normalizer = tf.keras.layers.Normalization(
    axis=-1,
    name="landmark_normalization"
)

print(
    "Adapting normalization using training data only..."
)

normalizer.adapt(
    X_train
)

print("✓ Normalization adapted")

# ============================================================
# 9. TEMPORAL ATTENTION
# ============================================================

class TemporalAttention(
    tf.keras.layers.Layer
):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.score = tf.keras.layers.Dense(
            1,
            activation="tanh"
        )

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


# ============================================================
# 10. BUILD MODEL
# ============================================================

print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

inputs = tf.keras.Input(
    shape=(
        SEQUENCE_LENGTH,
        FEATURES
    ),
    name="landmark_sequence"
)

x = normalizer(inputs)

# ------------------------------------------------------------
# Feature projection
# ------------------------------------------------------------

x = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="feature_projection"
)(x)

x = tf.keras.layers.BatchNormalization()(
    x
)

x = tf.keras.layers.Dropout(
    0.20
)(x)

# ------------------------------------------------------------
# Temporal convolution
# ------------------------------------------------------------

x = tf.keras.layers.Conv1D(
    filters=128,
    kernel_size=5,
    padding="same",
    activation="relu",
    name="temporal_conv_1"
)(x)

x = tf.keras.layers.BatchNormalization()(
    x
)

x = tf.keras.layers.Dropout(
    0.20
)(x)

# ------------------------------------------------------------
# First BiLSTM
# ------------------------------------------------------------

x = tf.keras.layers.Bidirectional(

    tf.keras.layers.LSTM(
        128,
        return_sequences=True,
        dropout=0.15,
        recurrent_dropout=0.0
    ),

    name="bilstm_1"

)(x)

# ------------------------------------------------------------
# Second BiLSTM
# ------------------------------------------------------------

x = tf.keras.layers.Bidirectional(

    tf.keras.layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.15,
        recurrent_dropout=0.0
    ),

    name="bilstm_2"

)(x)

# ------------------------------------------------------------
# Attention
# ------------------------------------------------------------

x = TemporalAttention(
    name="temporal_attention"
)(x)

# ------------------------------------------------------------
# Classifier
# ------------------------------------------------------------

x = tf.keras.layers.Dense(
    256,
    activation="relu",
    name="classifier_dense"
)(x)

x = tf.keras.layers.BatchNormalization()(
    x
)

x = tf.keras.layers.Dropout(
    0.30
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="word_output"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="sign_language_closed_set_model"
)

# ============================================================
# 11. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5,
            name="top5_accuracy"
        )
    ]
)

# ============================================================
# 12. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()

# ============================================================
# 13. CALLBACKS
# ============================================================

print("\n" + "=" * 60)
print("SETTING TRAINING CALLBACKS")
print("=" * 60)

best_model_path = (
    MODEL_DIR /
    "best_closed_temporal_model.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(
            best_model_path
        ),

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=12,

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=4,

        min_lr=1e-6,

        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(

        str(
            MODEL_DIR /
            "training_history.csv"
        )
    )
]

# ============================================================
# 14. TRAIN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)

print("\nTraining samples :", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples  :", len(X_test))

print("\nMaximum epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)

print(
    "\nIMPORTANT:"
)

print(
    "This is a CLOSED-SET evaluation."
)

print(
    "All 2667 classes are represented in training."
)

print(
    "Validation/test windows originate "
    "from the same recordings as training."
)

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# 15. TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_results = model.evaluate(

    X_test,
    y_test,

    batch_size=BATCH_SIZE,

    verbose=1,

    return_dict=True
)

print("\nFINAL RESULTS")

for key, value in test_results.items():

    print(
        f"{key}: {value:.4f}"
    )

# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR /
    "final_closed_temporal_model.keras"
)

model.save(
    final_model_path
)

print(
    "\n✓ Final model saved:"
)

print(
    final_model_path
)

# ============================================================
# 17. BEST VALIDATION SCORE
# ============================================================

val_accuracy_history = (
    history.history["val_accuracy"]
)

best_val_accuracy = max(
    val_accuracy_history
)

best_epoch = (
    np.argmax(
        val_accuracy_history
    ) + 1
)

print("\n" + "=" * 60)
print("BEST VALIDATION RESULT")
print("=" * 60)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.4f}"
)

print(
    "Best epoch:",
    best_epoch
)

# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SCRIPT 18 COMPLETE")
print("=" * 60)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Classes:",
    NUM_CLASSES
)

print(
    "\nBest validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "Test accuracy:",
    f"{test_results['accuracy'] * 100:.2f}%"
)

print(
    "Test top-5 accuracy:",
    f"{test_results['top5_accuracy'] * 100:.2f}%"
)

print(
    "\n✓ Training completed"
)

print(
    "✓ Best model saved"
)

print(
    "✓ Final model saved"
)

print(
    "\nNOTE:"
)

print(
    "These results are CLOSED-SET temporal results."
)

print(
    "They must NOT be described as "
    "signer-independent or independent-recording accuracy."
)

print(
    "\nDone."
)