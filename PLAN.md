# Naruto Jutsu Detector: Project Plan

## Goal
Build a webcam app where the user makes Naruto hand seals (Tiger, Ram, Monkey, etc.). The app recognizes each seal with machine learning, matches the sequence to a jutsu, and triggers visual effects and sound (e.g., a fireball) on screen.

## About the user (important)
- Beginner in Python. Explain code in simple terms, keep files small and readable, add comments, and avoid advanced patterns (no heavy abstractions, no async).
- Has a GPU (type not yet confirmed). Not needed until the optional YOLO stage.
- Wants to learn, so build ONE stage at a time. After each stage, stop, briefly explain what the new code does, and let the user test before moving on. Never build several stages at once, even if it seems faster.

## Tech stack
- Python 3.10 or 3.11 (MediaPipe can lag on newer versions)
- OpenCV (`opencv-python`): webcam capture only
- pygame: the display window, particle effects, and sound
- MediaPipe Tasks API (`HandLandmarker`): 21 landmarks per hand
- scikit-learn: Random Forest classifier on landmark features
- pandas + NumPy: store and handle landmark data
- joblib: save and load the trained model
- Plain Python `deque` for sequence matching
- git: version control, commit after every working stage
- Optional later: Ultralytics YOLO (YOLOv8/11), Roboflow/LabelImg, Google Colab

## Architecture rules
- Do NOT use `cv2.imshow()`. OpenCV only grabs frames. Convert each frame (BGR -> RGB, `swapaxes(0, 1)`) to a pygame surface and draw everything in pygame.
- Mirror the camera view with `cv2.flip(frame, 1)`, and process frames the same way in every script so features always match.
- Initialize audio with `pygame.mixer.pre_init(44100, -16, 2, 512)` before `pygame.init()`.
- Play sounds once per trigger (use a flag or cooldown), never every frame.
- Use additive blending (`BLEND_RGBA_ADD`) for fire and glow effects.
- All feature-building logic lives in `features.py`. Both `collect_data.py` and `main.py` must import it; never duplicate this logic.

## Project structure
```
naruto-jutsu/
├── main.py              # main loop (camera + pygame + prediction + effects)
├── features.py          # ONE function: MediaPipe results -> feature vector
├── collect_data.py      # record features for each seal into CSV
├── train.py             # train, evaluate, and save the classifier
├── effects.py           # Particle class and other effects
├── jutsu.py             # sequence matching logic
├── data/seals.csv
├── models/
│   ├── hand_landmarker.task   # MediaPipe model file
│   └── seal_classifier.pkl
├── assets/              # .wav sound effects, optional sprites
├── requirements.txt
├── .gitignore
└── PLAN.md
```

## Stages (do one at a time; stop after each for the user to test)

### Stage 0: Setup
- Create a virtual environment, `requirements.txt` (opencv-python, pygame, mediapipe, scikit-learn, pandas, numpy, joblib), and a `.gitignore` (ignore the venv folder and `__pycache__`).
- Initialize a git repo and make a first commit.
- Done when: `python -c "import cv2, pygame, mediapipe, sklearn"` runs without errors.

### Stage 1: Webcam in a pygame window
- Show the mirrored webcam feed at 960x540, ~30 FPS. Quit on window close or ESC.
- If camera index 0 doesn't work, try 1 or 2.
- Done when: the live camera shows in a pygame window with correct colors.

### Stage 2: Fire effect and sound on key press
- Add a `Particle` class (position, velocity, life, size, fading glow) in `effects.py`.
- Hold F to spawn fire particles at screen center. Play a sound once per press.
- Done when: fire appears on F, the sound plays once per press, no lag.

### Stage 3: MediaPipe hand tracking
- Use the MediaPipe Tasks API (`HandLandmarker`, VIDEO or LIVE_STREAM mode) with the `hand_landmarker.task` model saved in `models/`. Only use the legacy `mp.solutions` API if the installed version still supports it and the Tasks API causes problems.
- Detect up to 2 hands and draw landmarks and connections on the pygame window.
- Done when: landmarks follow both hands in real time.

### Stage 4: Data collection
- Feature vector (built in `features.py`):
  - Always place hands in fixed slots (left slot, right slot) using the handedness label. A missing hand is filled with zeros plus a 0/1 "hand present" flag.
  - Each hand's landmarks relative to its own wrist, scaled by hand size.
  - PLUS the offset between the two wrists (scaled), since seals depend on how the hands are positioned relative to each other.
- `collect_data.py`: press a number key to choose a label, hold SPACE to record features into `data/seals.csv`. Show the current label and sample count on screen.
- Labels: 3-4 easy seals plus a "none" class (hands down, random poses, half-made seals, hands moving between seals).
- Record several short sessions per label (different lighting, distance, angles) and store a `session_id` column in the CSV.
- Aim for a few hundred samples per label.
- Done when: the CSV has balanced samples for every label, including "none".

### Stage 5: Train and predict live
- `train.py`: split train/test BY SESSION, not by random rows (neighboring frames are nearly identical, so a random split gives misleadingly high accuracy).
- Train a Random Forest, print accuracy and a confusion matrix, save with joblib.
- In `main.py`: load the model, show the predicted seal and confidence on screen. Ignore "none" and low-confidence predictions, and smooth over about 5 frames to avoid flicker.
- Done when: accuracy on unseen sessions is good AND live predictions look stable.

### Stage 6: Sequence matching and jutsu trigger
- `jutsu.py`: keep a `deque` of recently confirmed seals. A seal counts once it is held steady for about 0.5 seconds. Match against jutsu definitions, e.g. `{"Fireball": ["tiger", "horse", "snake"]}` (placeholder sequence, can be changed).
- Add a short "seal detected" sound for each confirmed seal, and a timeout that resets the sequence if the user pauses too long.
- If a jutsu uses the same seal twice in a row, require a short release ("none") between them so it counts twice.
- On a match: play the jutsu sound once, spawn a big fire particle effect near the user's hands, flash the jutsu name on screen, then clear the history.
- Done when: performing the sequence triggers the effect and sound reliably.

### Stage 7 (optional): Upgrade to YOLO
- Only if MediaPipe struggles with overlapping two-handed seals.
- Label images in Roboflow/LabelImg (include a background/none set), train with Ultralytics (local NVIDIA GPU or Google Colab), and swap the detector in `main.py`, keeping the sequence and effects code unchanged.

### Stage 8 (optional): Polish
- Chakra aura, smoke puff, shadow clone using MediaPipe image segmentation, more jutsu, background music, simple on-screen UI.

## Known pitfalls
- Without a "none" class, the model always guesses a seal, even with hands at rest.
- MediaPipe loses tracking when both hands overlap heavily. This is the signal to consider Stage 7.
- High training accuracy but bad live results usually means features are built differently in collection vs live use, or the train/test split leaked.
- Wrong colors mean a missing BGR -> RGB conversion; a sideways image means a missing `swapaxes`.
- A sound replaying every frame means the trigger isn't gated by a flag or cooldown.
- Free sound effects: freesound.org or pixabay.com/sound-effects (check licenses).
