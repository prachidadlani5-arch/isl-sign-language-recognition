import os
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# ============================================================
# SCRIPT 6 - IMPROVED HAND LANDMARK EXTRACTION
# ============================================================
#
# Improvements:
# 1. Preserves label
# 2. Preserves video/frame identity
# 3. Keeps LEFT and RIGHT hands in fixed feature positions
# 4. Keeps exactly 126 landmark features
# 5. Stores hand detection information
#
# Feature layout:
#
# LEFT HAND:
#   f0  - f62
#
# RIGHT HAND:
#   f63 - f125
#
# Each hand:
#   21 landmarks x (x,y,z) = 63 values
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

FRAMES_DIR = Path("dataset/frames")

OUTPUT_FILE = Path("dataset/landmarks_v2.csv")
TEMP_FILE = Path("dataset/landmarks_v2_temp.csv")


# ============================================================
# SETTINGS
# ============================================================

MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
VALUES_PER_LANDMARK = 3

FEATURES_PER_HAND = (
    LANDMARKS_PER_HAND *
    VALUES_PER_LANDMARK
)

TOTAL_FEATURES = (
    MAX_HANDS *
    FEATURES_PER_HAND
)

MIN_DETECTION_CONFIDENCE = 0.5


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

FRAMES_DIR.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    f"f{i}"
    for i in range(TOTAL_FEATURES)
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SCRIPT 6 - IMPROVED HAND LANDMARK EXTRACTION")
print("=" * 70)

print()
print("Input directory :", FRAMES_DIR)
print("Output file     :", OUTPUT_FILE)
print("Features/frame  :", TOTAL_FEATURES)
print("Left hand       :", "f0-f62")
print("Right hand      :", "f63-f125")
print("=" * 70)


# ============================================================
# FIND WORD FOLDERS
# ============================================================

word_folders = sorted(
    [
        folder
        for folder in FRAMES_DIR.iterdir()
        if folder.is_dir()
    ],
    key=lambda x: x.name.lower()
)

total_words = len(word_folders)

print()
print("Found words:", total_words)

if total_words == 0:

    raise ValueError(
        "No word folders found in dataset/frames."
    )


# ============================================================
# LOAD PREVIOUS PROGRESS
# ============================================================

rows = []

completed_frames = set()

if TEMP_FILE.exists():

    print()
    print("Previous progress found.")
    print("Loading:", TEMP_FILE)

    try:

        old_df = pd.read_csv(
            TEMP_FILE
        )

        if not old_df.empty:

            rows = old_df.to_dict(
                "records"
            )

            if (
                "label" in old_df.columns
                and "frame_id" in old_df.columns
            ):

                completed_frames = set(
                    zip(
                        old_df["label"],
                        old_df["frame_id"]
                    )
                )

            print(
                "Existing rows:",
                len(rows)
            )

            print(
                "Completed frames:",
                len(completed_frames)
            )

    except Exception as e:

        print(
            "Could not load previous progress."
        )

        print(
            "Error:",
            e
        )


# ============================================================
# PROCESS DATA
# ============================================================

try:

    for word_index, folder in enumerate(
        word_folders,
        start=1
    ):

        word = folder.name

        print()
        print(
            f"[{word_index}/{total_words}] {word}"
        )


        # ====================================================
        # FIND FRAMES
        # ====================================================

        images = sorted(
            folder.glob("frame_*.jpg")
        )

        if not images:

            print(
                "   No JPG frames found."
            )

            continue


        print(
            "   Frames:",
            len(images)
        )


        word_new_rows = []


        # ====================================================
        # PROCESS EACH FRAME
        # ====================================================

        for frame_index, img_path in enumerate(
            images
        ):

            frame_id = (
                img_path.stem
            )

            identity = (
                word,
                frame_id
            )


            # ----------------------------------------------
            # SKIP ALREADY PROCESSED FRAME
            # ----------------------------------------------

            if identity in completed_frames:

                continue


            # ----------------------------------------------
            # READ IMAGE
            # ----------------------------------------------

            image = cv2.imread(
                str(img_path)
            )

            if image is None:

                print(
                    f"   WARNING: Could not read "
                    f"{img_path}"
                )

                continue


            # ----------------------------------------------
            # BGR -> RGB
            # ----------------------------------------------

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )


            # ----------------------------------------------
            # MEDIAPIPE
            # ----------------------------------------------

            result = hands.process(
                rgb
            )


            # =================================================
            # CREATE FIXED HAND FEATURE ARRAYS
            # =================================================

            left_values = [
                0.0
            ] * FEATURES_PER_HAND

            right_values = [
                0.0
            ] * FEATURES_PER_HAND


            left_detected = 0
            right_detected = 0


            # =================================================
            # EXTRACT HANDS
            # =================================================

            if (
                result.multi_hand_landmarks
                and result.multi_handedness
            ):

                for hand_landmarks, handedness in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                ):

                    hand_label = (
                        handedness.classification[0].label
                    )

                    values = []

                    for landmark in (
                        hand_landmarks.landmark
                    ):

                        values.extend([
                            float(landmark.x),
                            float(landmark.y),
                            float(landmark.z)
                        ])


                    # ----------------------------------------
                    # SAFETY CHECK
                    # ----------------------------------------

                    if len(values) != FEATURES_PER_HAND:

                        print(
                            f"   WARNING: Unexpected "
                            f"landmark count in "
                            f"{word}/{frame_id}"
                        )

                        continue


                    # ----------------------------------------
                    # PLACE HAND IN FIXED SLOT
                    # ----------------------------------------

                    if hand_label == "Left":

                        left_values = values
                        left_detected = 1


                    elif hand_label == "Right":

                        right_values = values
                        right_detected = 1


            # =================================================
            # COMBINE FEATURES
            # =================================================

            values = (
                left_values
                +
                right_values
            )


            if len(values) != TOTAL_FEATURES:

                raise ValueError(
                    "Incorrect feature count."
                )


            # =================================================
            # CREATE ROW
            # =================================================

            row = {

                "label":
                    word,

                "video_id":
                    word,

                "frame_id":
                    frame_id,

                "frame_index":
                    frame_index,

                "left_hand_detected":
                    left_detected,

                "right_hand_detected":
                    right_detected

            }


            # =================================================
            # ADD LANDMARK FEATURES
            # =================================================

            for feature_index, value in enumerate(
                values
            ):

                row[
                    f"f{feature_index}"
                ] = value


            word_new_rows.append(
                row
            )

            completed_frames.add(
                identity
            )


        # ====================================================
        # ADD WORD DATA
        # ====================================================

        if word_new_rows:

            rows.extend(
                word_new_rows
            )

            print(
                "   Added frames:",
                len(word_new_rows)
            )


            # =================================================
            # SAVE PROGRESS
            # =================================================

            progress_df = pd.DataFrame(
                rows
            )

            progress_df.to_csv(
                TEMP_FILE,
                index=False
            )

            print(
                "   Progress saved."
            )

        else:

            print(
                "   No new frames."
            )


        # ====================================================
        # PERIODIC PROGRESS
        # ====================================================

        if word_index % 100 == 0:

            print()
            print(
                f"Processed "
                f"{word_index}/{total_words} words"
            )


# ============================================================
# INTERRUPT HANDLING
# ============================================================

except KeyboardInterrupt:

    print()
    print("=" * 70)
    print("PROCESS INTERRUPTED")
    print("=" * 70)

    print(
        "Progress is saved in:"
    )

    print(
        TEMP_FILE
    )

    hands.close()

    raise SystemExit


finally:

    hands.close()


# ============================================================
# FINAL SAVE
# ============================================================

if rows:

    final_df = pd.DataFrame(
        rows
    )


    # ========================================================
    # COLUMN ORDER
    # ========================================================

    metadata_columns = [

        "label",

        "video_id",

        "frame_id",

        "frame_index",

        "left_hand_detected",

        "right_hand_detected"

    ]


    final_columns = (
        metadata_columns
        +
        FEATURE_COLUMNS
    )


    # Keep only columns that exist

    final_df = final_df[
        [
            column
            for column in final_columns
            if column in final_df.columns
        ]
    ]


    # ========================================================
    # SORT DATA
    # ========================================================

    final_df = final_df.sort_values(
        [
            "label",
            "frame_index"
        ]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # SAVE
    # ========================================================

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # REMOVE TEMP FILE
    # ========================================================

    if TEMP_FILE.exists():

        try:

            TEMP_FILE.unlink()

        except PermissionError:

            print(
                "WARNING: Could not remove "
                "temporary file."
            )


    # ========================================================
    # STATISTICS
    # ========================================================

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )

    print(
        "Total rows:",
        len(final_df)
    )

    print(
        "Total words:",
        final_df["label"].nunique()
    )

    print(
        "Features/frame:",
        len(FEATURE_COLUMNS)
    )

    print()
    print(
        "Left-hand detections:",
        int(
            final_df[
                "left_hand_detected"
            ].sum()
        )
    )

    print(
        "Right-hand detections:",
        int(
            final_df[
                "right_hand_detected"
            ].sum()
        )
    )

    print()
    print(
        "Feature layout:"
    )

    print(
        "  Left  hand -> f0-f62"
    )

    print(
        "  Right hand -> f63-f125"
    )

    print()
    print("=" * 70)


else:

    print()
    print(
        "No landmark data was generated."
    )
    