"""Turn MediaPipe's hand results into a list of numbers (a "feature vector").

The machine learning model can only read numbers, so this file is the bridge.
BOTH collect_data.py and main.py use build_features(), so the numbers
are always made the same way. Never copy this logic anywhere else.

The list always has the same length (130 numbers):
    left hand slot  : 1 "is present" flag + 21 points x 3 numbers = 64
    right hand slot : the same = 64
    wrist offset    : how far apart the two wrists are (x and y) = 2
"""

import numpy as np

# Our frames are 960x540, but MediaPipe gives x and y as fractions from 0 to 1.
# Without this, a distance sideways would count as "shorter" than the same
# distance up/down. Multiplying x by the aspect ratio fixes that.
ASPECT = 960 / 540

NUM_LANDMARKS = 21     # points per hand
WRIST = 0              # landmark number of the wrist
MIDDLE_KNUCKLE = 9     # landmark number of the base of the middle finger
SLOTS = ["left", "right"]
NUMBERS_PER_HAND = 1 + NUM_LANDMARKS * 3


def get_hand_label(handedness):
    """Return "Left" or "Right" as the USER sees it in the mirrored picture.
    MediaPipe assumes the picture is NOT mirrored, so its answer is the opposite
    of what we want. We flip it here."""
    label = handedness[0].category_name
    return "Right" if label == "Left" else "Left"


def hand_to_numbers(landmarks):
    """Turn one hand's 21 points into positions relative to its own wrist.
    Returns (relative points, wrist position, hand size)."""
    points = np.array([[p.x * ASPECT, p.y, p.z * ASPECT] for p in landmarks])

    wrist = points[WRIST]
    # "Hand size" = distance from the wrist to the middle finger knuckle (x and y only).
    # Dividing by it means a hand close to the camera and far away look the same.
    size = np.linalg.norm(points[MIDDLE_KNUCKLE][:2] - wrist[:2])
    size = max(size, 1e-6)  # never divide by zero

    relative = (points - wrist) / size
    return relative, wrist, size


def build_features(result):
    """Turn a MediaPipe result into a list of 130 numbers."""
    # Put each hand in its fixed slot, using the Left/Right label.
    hands = {"left": None, "right": None}
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        slot = get_hand_label(handedness).lower()
        score = handedness[0].score
        # If MediaPipe calls both hands "Left", keep the one it is surest about
        if hands[slot] is None or score > hands[slot]["score"]:
            relative, wrist, size = hand_to_numbers(landmarks)
            hands[slot] = {"relative": relative, "wrist": wrist, "size": size, "score": score}

    features = []
    for slot in SLOTS:
        hand = hands[slot]
        if hand is None:
            features += [0.0] * NUMBERS_PER_HAND          # missing hand: all zeros, flag 0
        else:
            features.append(1.0)                          # flag: this hand is present
            features += hand["relative"].flatten().tolist()

    # How the hands sit relative to each other (only if both are visible)
    if hands["left"] is not None and hands["right"] is not None:
        offset = hands["right"]["wrist"][:2] - hands["left"]["wrist"][:2]
        average_size = (hands["left"]["size"] + hands["right"]["size"]) / 2
        features += (offset / average_size).tolist()
    else:
        features += [0.0, 0.0]

    return features


def feature_names():
    """Column names for the CSV file, in the same order as build_features()."""
    names = []
    for slot in SLOTS:
        names.append(f"{slot}_present")
        for i in range(NUM_LANDMARKS):
            names += [f"{slot}_{i}_x", f"{slot}_{i}_y", f"{slot}_{i}_z"]
    names += ["wrist_offset_x", "wrist_offset_y"]
    return names
