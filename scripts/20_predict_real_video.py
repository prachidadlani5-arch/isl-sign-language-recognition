import numpy as np
import tensorflow as tf
import cv2
import json
import pandas as pd
from pathlib import Path

import mediapipe as mp


# ============================================================
# SCRIPT 20 - REAL VIDEO SIGN LANGUAGE INFERENCE
# ============================================================

print("=" * 70)
print("SCRIPT 20 - REAL VIDEO SIGN LANGUAGE INFERENCE")
print("=" * 70)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 64
FEATURES = 126
NUM_CLASSES = 2667

TOP_K = 5

MODEL_PATH = Path(
    "models/sign_language_temporal/"
    "best_closed_temporal_model.keras"
)

# ------------------------------------------------------------
# CHANGE ONLY THIS PATH IF YOU WANT ANOTHER VIDEO
# ------------------------------------------------------------

VIDEO_PATH = Path(
    "dataset/videos/Cat.mp4"
)

OUTPUT_DIR = Path(
    "models/sign_language_temporal/inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ENVIRONMENT
# ============================================================

print("\nTensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected")
else:
    print("Running on CPU")


# ============================================================
# CONFIGURATION
# ============================================================

print("\nConfiguration:")
print("Sequence length :", SEQUENCE_LENGTH)
print("Features/frame  :", FEATURES)
print("Classes         :", NUM_CLASSES)
print("Top-K           :", TOP_K)

print("\nModel:")
print(MODEL_PATH)

print("\nVideo:")
print(VIDEO_PATH)


# ============================================================
# 1. TEMPORAL ATTENTION
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
            activation="tanh",
            name="attention_score"
        )

    def build(self, input_shape):

        # Expected:
        # (batch, time, features)
        #
        # For this model:
        # (None, 64, 128)

        feature_dimension = int(
            input_shape[-1]
        )

        # Explicitly build the internal Dense layer.
        # This prevents the Keras 3 deserialization
        # problem encountered previously.

        self.score.build(
            (None, feature_dimension)
        )

        super().build(input_shape)

    def call(self, inputs):

        # inputs:
        # (batch, time, features)

        scores = self.score(
            inputs
        )

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
# 2. CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FILES")
print("=" * 70)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


if not VIDEO_PATH.exists():

    raise FileNotFoundError(
        f"\nVideo not found:\n{VIDEO_PATH}\n\n"
        "Put your video inside the dataset/videos folder "
        "or change VIDEO_PATH in this script."
    )


print("✓ Model found")
print("✓ Video found")


# ============================================================
# 3. FIND LABEL MAP
# ============================================================

print("\n" + "=" * 70)
print("LOADING LABEL MAP")
print("=" * 70)


possible_label_maps = [

    Path(
        "dataset/sequences/"
        "temporal_label_map.csv"
    ),

    Path(
        "dataset/sequences/"
        "label_map.csv"
    ),

    Path(
        "dataset/sequences/"
        "label_map.json"
    ),

    Path(
        "dataset/label_map.csv"
    ),

    Path(
        "dataset/label_map.json"
    ),

    Path(
        "dataset/labels.csv"
    ),

    Path(
        "dataset/labels.json"
    ),

    Path(
        "models/sign_language_temporal/"
        "label_map.csv"
    ),

    Path(
        "models/sign_language_temporal/"
        "label_map.json"
    )

]


label_map_path = None


for path in possible_label_maps:

    if path.exists():

        label_map_path = path

        break


if label_map_path is None:

    raise FileNotFoundError(

        "\nCould not find a label map.\n\n"

        "Expected one of:\n"

        "dataset/sequences/temporal_label_map.csv\n"
        "dataset/sequences/label_map.csv\n"
        "dataset/sequences/label_map.json\n"
        "dataset/label_map.csv\n"
        "dataset/label_map.json\n"
        "dataset/labels.csv\n"
        "dataset/labels.json\n"

    )


print("Label map:")
print(label_map_path)


# ============================================================
# 4. LOAD LABEL MAP
# ============================================================

label_dictionary = {}


# ------------------------------------------------------------
# CSV LABEL MAP
# ------------------------------------------------------------

if label_map_path.suffix.lower() == ".csv":

    label_data = pd.read_csv(
        label_map_path
    )

    print(
        "Label map columns:",
        list(label_data.columns)
    )


    # Expected columns:
    # label_id
    # label

    if (
        "label_id" in label_data.columns
        and
        "label" in label_data.columns
    ):

        for _, row in label_data.iterrows():

            class_id = int(
                row["label_id"]
            )

            label = str(
                row["label"]
            )

            label_dictionary[
                class_id
            ] = label

    else:

        raise ValueError(

            "\nCSV label map must contain "
            "'label_id' and 'label' columns.\n\n"

            f"Found columns: "
            f"{list(label_data.columns)}"

        )


# ------------------------------------------------------------
# JSON LABEL MAP
# ------------------------------------------------------------

elif label_map_path.suffix.lower() == ".json":

    with open(
        label_map_path,
        "r",
        encoding="utf-8"
    ) as f:

        label_data = json.load(f)


    # Case 1:
    #
    # {
    #     "0": "Abbreviation",
    #     "1": "Abeyance"
    # }

    if isinstance(
        label_data,
        dict
    ):

        for key, value in label_data.items():

            try:

                class_id = int(key)

                label_dictionary[
                    class_id
                ] = str(value)

            except:

                pass


    # Case 2:
    #
    # [
    #     {
    #         "id": 0,
    #         "label": "Abbreviation"
    #     }
    # ]

    elif isinstance(
        label_data,
        list
    ):

        for item in label_data:

            if isinstance(
                item,
                dict
            ):

                if (
                    "id" in item
                    and
                    "label" in item
                ):

                    label_dictionary[
                        int(item["id"])
                    ] = str(
                        item["label"]
                    )


# ------------------------------------------------------------
# VERIFY LABEL COUNT
# ------------------------------------------------------------

print(
    "\nLabels found:",
    len(label_dictionary)
)


if len(label_dictionary) != NUM_CLASSES:

    print(
        "\nWARNING:"
    )

    print(
        "Expected:",
        NUM_CLASSES
    )

    print(
        "Found:",
        len(label_dictionary)
    )

else:

    print(
        "✓ Labels loaded:",
        len(label_dictionary)
    )

    print(
        "✓ All 2667 classes available"
    )


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEMPORAL MODEL")
print("=" * 70)


print(
    "\nLoading:",
    MODEL_PATH
)


model = tf.keras.models.load_model(

    MODEL_PATH,

    custom_objects={
        "TemporalAttention":
        TemporalAttention
    },

    compile=False

)


print(
    "✓ Model loaded"
)


print(
    "\nModel input shape:",
    model.input_shape
)


print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# 6. VERIFY MODEL SHAPE
# ============================================================

expected_input_shape = (

    None,

    SEQUENCE_LENGTH,

    FEATURES

)


expected_output_shape = (

    None,

    NUM_CLASSES

)


if model.input_shape != expected_input_shape:

    raise ValueError(

        "\nUnexpected model input shape.\n"

        f"Expected: {expected_input_shape}\n"

        f"Got:      {model.input_shape}"

    )


if model.output_shape != expected_output_shape:

    raise ValueError(

        "\nUnexpected model output shape.\n"

        f"Expected: {expected_output_shape}\n"

        f"Got:      {model.output_shape}"

    )


print(
    "✓ Model input shape verified"
)

print(
    "✓ Model output shape verified"
)


# ============================================================
# 7. INITIALIZE MEDIAPIPE
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING MEDIAPIPE")
print("=" * 70)


mp_hands = mp.solutions.hands


hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=2,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

)


print(
    "✓ MediaPipe Hands initialized"
)


# ============================================================
# 8. EXTRACT 126 FEATURES FROM ONE FRAME
# ============================================================

def extract_landmarks(
    frame
):

    """
    Returns exactly 126 features.

    Left hand:
        21 landmarks × 3
        = 63

    Right hand:
        21 landmarks × 3
        = 63

    Total:
        126 features
    """

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = hands.process(
        rgb
    )


    left_hand = np.zeros(
        63,
        dtype=np.float32
    )


    right_hand = np.zeros(
        63,
        dtype=np.float32
    )


    if results.multi_hand_landmarks:

        if results.multi_handedness:

            for (
                hand_landmarks,
                handedness
            ) in zip(

                results.multi_hand_landmarks,

                results.multi_handedness

            ):

                label = (

                    handedness
                    .classification[0]
                    .label

                )


                landmarks = []


                for landmark in (
                    hand_landmarks.landmark
                ):

                    landmarks.extend([

                        landmark.x,

                        landmark.y,

                        landmark.z

                    ])


                landmarks = np.array(

                    landmarks,

                    dtype=np.float32

                )


                if label == "Left":

                    left_hand = landmarks


                elif label == "Right":

                    right_hand = landmarks


    features = np.concatenate([

        left_hand,

        right_hand

    ])


    return features


# ============================================================
# 9. READ VIDEO
# ============================================================

print("\n" + "=" * 70)
print("READING VIDEO")
print("=" * 70)


cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    hands.close()

    raise RuntimeError(

        "\nCould not open video."

    )


total_frames = int(

    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

)


fps = cap.get(
    cv2.CAP_PROP_FPS
)


print(
    "\nTotal frames:",
    total_frames
)


print(
    "FPS:",
    fps
)


if total_frames == 0:

    cap.release()

    hands.close()

    raise RuntimeError(

        "Video contains no readable frames."

    )


# ============================================================
# 10. EXTRACT VIDEO FEATURES
# ============================================================

print("\nExtracting landmarks...")


all_features = []


frame_number = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    features = extract_landmarks(
        frame
    )


    all_features.append(
        features
    )


    frame_number += 1


    if (
        frame_number % 20 == 0
        or
        frame_number == total_frames
    ):

        print(

            f"\rProcessed frames: "
            f"{frame_number}/{total_frames}",

            end=""

        )


cap.release()


print(
    "\n\n✓ Video processing complete"
)


if len(all_features) == 0:

    hands.close()

    raise RuntimeError(

        "No frames were extracted."

    )


all_features = np.array(

    all_features,

    dtype=np.float32

)


print(
    "Feature array shape:",
    all_features.shape
)


# ============================================================
# 11. VERIFY FEATURES
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING FEATURES")
print("=" * 70)


if all_features.ndim != 2:

    hands.close()

    raise ValueError(

        "Extracted features must "
        "be 2-dimensional."

    )


if all_features.shape[1] != FEATURES:

    hands.close()

    raise ValueError(

        f"Expected {FEATURES} features "
        f"per frame, but got "
        f"{all_features.shape[1]}."

    )


nan_count = np.isnan(
    all_features
).sum()


inf_count = np.isinf(
    all_features
).sum()


print(
    "NaN:",
    nan_count
)


print(
    "Infinite:",
    inf_count
)


if nan_count != 0:

    hands.close()

    raise ValueError(

        "NaN values detected."

    )


if inf_count != 0:

    hands.close()

    raise ValueError(

        "Infinite values detected."

    )


print(
    "✓ Features valid"
)


# ============================================================
# 12. CHECK HAND DETECTION
# ============================================================

non_zero_frames = np.sum(

    np.any(
        all_features != 0,
        axis=1
    )

)


zero_frames = (

    len(all_features)
    -
    non_zero_frames

)


print("\nHand detection summary:")

print(
    "Frames with hand landmarks:",
    non_zero_frames
)


print(
    "Frames without landmarks:",
    zero_frames
)


if non_zero_frames == 0:

    hands.close()

    raise RuntimeError(

        "\nMediaPipe could not detect "
        "any hand landmarks in the video.\n\n"

        "Make sure the hands are visible, "
        "well-lit, and reasonably large "
        "in the video."

    )


# ============================================================
# 13. CREATE 64-FRAME SEQUENCE
# ============================================================

print("\n" + "=" * 70)
print("CREATING 64-FRAME SEQUENCE")
print("=" * 70)


num_frames = len(
    all_features
)


if num_frames >= SEQUENCE_LENGTH:

    # --------------------------------------------------------
    # Uniformly sample exactly 64 frames
    # --------------------------------------------------------

    indices = np.linspace(

        0,

        num_frames - 1,

        SEQUENCE_LENGTH

    ).astype(int)


    sequence = all_features[
        indices
    ]


    print(
        f"Video has {num_frames} frames."
    )


    print(
        "Uniformly sampled 64 frames."
    )


else:

    # --------------------------------------------------------
    # Pad short video using last frame
    # --------------------------------------------------------

    padding_count = (

        SEQUENCE_LENGTH
        -
        num_frames

    )


    last_frame = all_features[
        -1
    ]


    padding = np.tile(

        last_frame,

        (
            padding_count,
            1
        )

    )


    sequence = np.vstack([

        all_features,

        padding

    ])


    print(
        f"Video has only {num_frames} frames."
    )


    print(
        f"Padded with "
        f"{padding_count} frames."
    )


print(
    "\nSequence shape:",
    sequence.shape
)


if sequence.shape != (

    SEQUENCE_LENGTH,

    FEATURES

):

    hands.close()

    raise ValueError(

        f"Invalid sequence shape: "
        f"{sequence.shape}"

    )


print(
    "✓ Sequence shape valid"
)


# ============================================================
# 14. PREPARE MODEL INPUT
# ============================================================

X_input = np.expand_dims(

    sequence,

    axis=0

).astype(
    np.float32
)


print(
    "\nModel input shape:",
    X_input.shape
)


if X_input.shape != (

    1,

    SEQUENCE_LENGTH,

    FEATURES

):

    hands.close()

    raise ValueError(

        f"Invalid model input shape: "
        f"{X_input.shape}"

    )


# ============================================================
# 15. PREDICT
# ============================================================

print("\n" + "=" * 70)
print("RUNNING MODEL PREDICTION")
print("=" * 70)


predictions = model.predict(

    X_input,

    verbose=0

)


print(
    "Prediction shape:",
    predictions.shape
)


# ============================================================
# 16. VERIFY PREDICTIONS
# ============================================================

if predictions.shape != (

    1,

    NUM_CLASSES

):

    hands.close()

    raise ValueError(

        f"Unexpected prediction shape: "
        f"{predictions.shape}"

    )


probabilities = predictions[0]


if np.isnan(
    probabilities
).any():

    hands.close()

    raise ValueError(

        "Prediction contains NaN values."

    )


if np.isinf(
    probabilities
).any():

    hands.close()

    raise ValueError(

        "Prediction contains infinite values."

    )


# ============================================================
# 17. TOP-5 PREDICTIONS
# ============================================================

top_indices = np.argsort(

    probabilities

)[::-1][:TOP_K]


print("\n" + "=" * 70)
print("TOP-5 PREDICTIONS")
print("=" * 70)


results = []


for rank, class_id in enumerate(

    top_indices,

    start=1

):

    confidence = float(

        probabilities[
            class_id
        ]

    )


    label = label_dictionary.get(

        int(class_id),

        f"Class {class_id}"

    )


    results.append({

        "rank":
            rank,

        "class_id":
            int(class_id),

        "label":
            label,

        "confidence":
            confidence,

        "confidence_percent":
            confidence * 100

    })


    print(

        f"{rank}. "
        f"{label} "
        f"({confidence * 100:.2f}%)"

    )


# ============================================================
# 18. FINAL PREDICTION
# ============================================================

best_class_id = int(

    top_indices[0]

)


best_label = label_dictionary.get(

    best_class_id,

    f"Class {best_class_id}"

)


best_confidence = float(

    probabilities[
        best_class_id
    ]

)


print("\n" + "=" * 70)
print("FINAL PREDICTION")
print("=" * 70)


print(
    "\nPredicted sign:",
    best_label
)


print(
    "Class ID:",
    best_class_id
)


print(
    "Confidence:",
    f"{best_confidence * 100:.2f}%"
)


# ============================================================
# 19. SAVE RESULT
# ============================================================

result_file = (

    OUTPUT_DIR
    /
    "latest_prediction.json"

)


result_data = {

    "video":
        str(VIDEO_PATH),

    "predicted_class_id":
        best_class_id,

    "predicted_label":
        best_label,

    "confidence":
        best_confidence,

    "confidence_percent":
        best_confidence * 100,

    "top_5_predictions":
        results,

    "frames_in_video":
        num_frames,

    "fps":
        float(fps),

    "frames_with_landmarks":
        int(non_zero_frames),

    "frames_without_landmarks":
        int(zero_frames),

    "sequence_length":
        SEQUENCE_LENGTH,

    "features_per_frame":
        FEATURES,

    "num_classes":
        NUM_CLASSES

}


with open(

    result_file,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        result_data,

        f,

        indent=4

    )


print(
    "\n✓ Result saved:"
)


print(
    result_file
)


# ============================================================
# 20. CLEANUP
# ============================================================

hands.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("SCRIPT 20 COMPLETE")
print("=" * 70)


print(
    "\n✓ Model loaded"
)


print(
    "✓ Video processed"
)


print(
    "✓ 126 features extracted per frame"
)


print(
    "✓ 64-frame temporal sequence created"
)


print(
    "✓ Prediction generated"
)


print(
    "✓ Top-5 predictions generated"
)


print(
    "\nFINAL SIGN:",
    best_label
)


print(
    "CONFIDENCE:",
    f"{best_confidence * 100:.2f}%"
)


print(
    "\nDone."
)