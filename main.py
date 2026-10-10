"""Stage 6: webcam + hand tracking + seal prediction + jutsu sequences.

OpenCV only grabs frames from the camera.
MediaPipe finds the hands (21 landmark points per hand).
The trained classifier (from train.py) guesses which seal you are making.
jutsu.py watches the seals go by and spots a complete jutsu.
pygame draws everything on screen (and F still makes test fire).
"""

import sys
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pygame

from effects import Particle, spawn_burst, warm_up_glows
from features import build_features, get_hand_label
from jutsu import JUTSUS, SealSequence

WIDTH, HEIGHT = 960, 540  # window size
FPS = 30                  # frames per second we aim for
PARTICLES_PER_FRAME = 6   # how much fire to make each frame while F is held
MAX_PARTICLES = 400       # safety limit so the game never gets slow

BURST_FRAMES = 45         # the jutsu fire keeps growing for this many frames (1.5 s)
MAX_BURST_PARTICLES = 700 # safety limit while the big fire is going
FLASH_SECONDS = 2.5       # how long the jutsu name stays on screen

SMOOTH_FRAMES = 5         # average the guess over this many frames to stop flicker
MIN_CONFIDENCE = 0.6      # ignore guesses the model is less sure about than this

MODEL_PATH = "models/hand_landmarker.task"
SEAL_MODEL_PATH = "models/seal_classifier.pkl"
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS  # which points join with a line

# Colours for drawing each hand (R, G, B)
HAND_COLORS = {"Left": (0, 220, 255), "Right": (80, 255, 80)}


def open_camera():
    """Try camera 0, then 1, then 2. Return the first one that works."""
    for index in range(3):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW opens faster on Windows
        if cap.isOpened():
            ok, _ = cap.read()  # make sure it really gives us a picture
            if ok:
                print(f"Using camera {index}")
                return cap
        cap.release()
    return None


def prepare_frame(frame):
    """Mirror, resize and convert a camera frame to RGB.
    EVERY script must prepare frames this way so the hand data always matches."""
    frame = cv2.flip(frame, 1)                       # mirror, like a real mirror
    frame = cv2.resize(frame, (WIDTH, HEIGHT))       # fit the window
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # OpenCV uses BGR, pygame and MediaPipe use RGB


def rgb_to_surface(rgb):
    """Turn an RGB frame into a pygame picture (surface)."""
    # OpenCV is (height, width), pygame is (width, height), so swap the first two axes
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def create_hand_detector():
    """Load the MediaPipe hand model. It can find up to 2 hands."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,  # we feed it video frames one by one
        num_hands=2,
        # The defaults are 0.5. Lower numbers make it keep tracking a hand that is
        # partly hidden (like when hands cross), at the price of a few more false alarms.
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def find_hands(detector, rgb, timestamp_ms):
    """Ask MediaPipe where the hands are in this RGB frame."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return detector.detect_for_video(image, timestamp_ms)


def load_seal_model():
    """Load the classifier made by train.py (or None if it hasn't been trained yet)."""
    try:
        return joblib.load(SEAL_MODEL_PATH)
    except FileNotFoundError:
        print("No trained model found. Run train.py first - running without seal prediction.")
        return None


def predict_probabilities(model, result):
    """How likely is each label right now? Returns one number per label; they add up to 1."""
    labels = list(model.classes_)
    if len(result.hand_landmarks) < 2:
        # All our seals use both hands, so with fewer than 2 hands it can only be "none"
        probabilities = np.zeros(len(labels))
        probabilities[labels.index("none")] = 1.0
        return probabilities
    return model.predict_proba([build_features(result)])[0]


def pick_seal(model, recent):
    """Average the last few frames' guesses, then pick the winner.
    Returns (seal name or None, confidence). "none" and unsure guesses give None."""
    average = np.mean(recent, axis=0)
    best = int(np.argmax(average))
    label = model.classes_[best]
    confidence = float(average[best])
    if label == "none" or confidence < MIN_CONFIDENCE:
        return None, confidence
    return label, confidence


def draw_prediction(screen, font, big_font, model, recent, seal, confidence):
    """Show the detected seal at the bottom, with every label's chance underneath."""
    if seal:
        text = big_font.render(f"{seal.upper()}  {confidence:.0%}", True, (255, 220, 80))
    else:
        text = big_font.render("no seal", True, (170, 170, 170))
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 90))

    # Small line with the (smoothed) chance of each label, handy for checking stability
    average = np.mean(recent, axis=0)
    chances = "   ".join(f"{name} {p:.0%}" for name, p in zip(model.classes_, average))
    line = font.render(chances, True, (255, 255, 255))
    screen.blit(line, (WIDTH // 2 - line.get_width() // 2, HEIGHT - 40))


def load_sound(path):
    """Load a sound file. The game still works (silently) if it is missing."""
    try:
        return pygame.mixer.Sound(path)
    except (pygame.error, FileNotFoundError):
        print(f"Could not load {path} - running without that sound.")
        return None


def hands_center(result):
    """Middle of all the hand points on screen, in pixels (None if no hands)."""
    xs = [p.x for hand in result.hand_landmarks for p in hand]
    ys = [p.y for hand in result.hand_landmarks for p in hand]
    if not xs:
        return None
    return int(np.mean(xs) * WIDTH), int(np.mean(ys) * HEIGHT)


def draw_sequence_info(screen, font, sequence, now):
    """Show the jutsu recipes, the seals confirmed so far, and the hold progress bar."""
    # Recipes (top left, under the hint)
    y = 40
    for name, seals in JUTSUS.items():
        line = font.render(f"{name}: {' > '.join(seals)}", True, (255, 255, 255))
        screen.blit(line, (10, y))
        y += 26

    # Seals confirmed so far (bottom middle, above the prediction)
    if sequence.history:
        so_far = font.render(" > ".join(sequence.history), True, (255, 150, 40))
        screen.blit(so_far, (WIDTH // 2 - so_far.get_width() // 2, HEIGHT - 160))

    # Progress bar: fills up while you hold a seal steadily
    bar_x, bar_y, bar_w, bar_h = WIDTH // 2 - 120, HEIGHT - 125, 240, 12
    pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
    fill = int(bar_w * sequence.hold_progress(now))
    if fill > 0:
        pygame.draw.rect(screen, (255, 200, 60), (bar_x, bar_y, fill, bar_h))
    pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 1)


def draw_flash(screen, huge_font, name, age):
    """Flash the jutsu name in the middle of the screen, fading out at the end."""
    if age > FLASH_SECONDS:
        return
    fade_start = FLASH_SECONDS * 0.6
    alpha = 255 if age < fade_start else int(255 * (1 - (age - fade_start) / (FLASH_SECONDS - fade_start)))
    words = f"{name} Jutsu!"
    shadow = huge_font.render(words, True, (0, 0, 0))
    text = huge_font.render(words, True, (255, 140, 30))
    shadow.set_alpha(alpha)
    text.set_alpha(alpha)
    x = WIDTH // 2 - text.get_width() // 2
    y = HEIGHT // 2 - 150
    screen.blit(shadow, (x + 4, y + 4))
    screen.blit(text, (x, y))


def draw_hands(screen, result, font):
    """Draw the landmarks (dots) and connections (lines) for every hand found."""
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = get_hand_label(handedness)  # "Left" or "Right"
        color = HAND_COLORS.get(label, (255, 255, 255))

        # Landmark positions are numbers from 0 to 1, so multiply by the window size
        points = [(int(p.x * WIDTH), int(p.y * HEIGHT)) for p in landmarks]

        for connection in HAND_CONNECTIONS:
            pygame.draw.line(screen, color, points[connection.start], points[connection.end], 3)
        for point in points:
            pygame.draw.circle(screen, (255, 255, 255), point, 5)
            pygame.draw.circle(screen, color, point, 5, 2)

        # Write "Left" / "Right" next to the wrist (landmark 0)
        text = font.render(label, True, color)
        screen.blit(text, (points[0][0] + 10, points[0][1] + 10))


def main():
    cap = open_camera()
    if cap is None:
        print("Could not open any camera (tried 0, 1 and 2).")
        sys.exit(1)

    # Audio settings must be set BEFORE pygame.init() or sounds can lag
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Naruto Jutsu")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)
    big_font = pygame.font.SysFont("arial", 44, bold=True)
    huge_font = pygame.font.SysFont("arial", 90, bold=True)
    warm_up_glows()  # prepare the fire pictures now, so there is no stutter later

    detector = create_hand_detector()
    last_timestamp = -1  # MediaPipe needs each frame's timestamp to be bigger than the last

    seal_model = load_seal_model()
    recent = deque(maxlen=SMOOTH_FRAMES)  # the last few frames' guesses (old ones drop off automatically)
    sequence = SealSequence()             # remembers confirmed seals and spots jutsu

    fire_sound = load_sound("assets/fire.wav")    # test fire (F key)
    seal_sound = load_sound("assets/seal.wav")    # a short ding for each confirmed seal
    jutsu_sound = load_sound("assets/jutsu.wav")  # the big sound when a jutsu fires

    particles = []  # all the fire blobs that are alive right now
    burst_frames_left = 0                          # how long the jutsu fire still has to run
    last_center = (WIDTH // 2, HEIGHT // 2)        # where the hands were last seen
    flash_name, flash_start = None, 0.0            # the jutsu name on screen and when it appeared

    running = True
    while running:
        # 1. Handle events: closing the window or pressing ESC quits
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f and fire_sound:
                    # KEYDOWN happens ONCE per press (not every frame while held),
                    # so the sound plays once per press.
                    fire_sound.play()

        # 2. Grab a frame from the camera and prepare it (mirror + RGB)
        ok, frame = cap.read()
        if not ok:
            continue  # skip this round if the camera had no picture
        rgb = prepare_frame(frame)

        # 3. Find the hands
        timestamp = max(pygame.time.get_ticks(), last_timestamp + 1)
        last_timestamp = timestamp
        result = find_hands(detector, rgb, timestamp)

        # 3b. Guess the seal (if we have a trained model)
        seal, confidence = None, 0.0
        if seal_model:
            recent.append(predict_probabilities(seal_model, result))
            seal, confidence = pick_seal(seal_model, recent)

        # 3c. Feed the seal to the sequence matcher (it needs the time in seconds)
        now = pygame.time.get_ticks() / 1000
        confirmed, jutsu = sequence.update(seal, now)
        if jutsu:
            # A whole jutsu! These lines run ONCE, at the moment it is recognised,
            # so the sound plays once (not every frame).
            if jutsu_sound:
                jutsu_sound.play()
            burst_frames_left = BURST_FRAMES
            flash_name, flash_start = jutsu, now
        elif confirmed and seal_sound:
            seal_sound.play()  # one more seal confirmed

        # 4. Fire: hold F for test fire at the centre; a jutsu makes a big fire at your hands
        center = hands_center(result)
        if center:
            last_center = center
        keys = pygame.key.get_pressed()
        if keys[pygame.K_f] and len(particles) < MAX_PARTICLES:
            for _ in range(PARTICLES_PER_FRAME):
                particles.append(Particle(WIDTH // 2, HEIGHT // 2))
        if burst_frames_left > 0:
            burst_frames_left -= 1
            if len(particles) < MAX_BURST_PARTICLES:
                spawn_burst(particles, last_center[0], last_center[1])

        # 5. Move every particle, then throw away the ones that faded out
        for p in particles:
            p.update()
        particles = [p for p in particles if not p.is_dead()]

        # 6. Draw: camera first, then hands, then fire on top, then a hint text
        screen.blit(rgb_to_surface(rgb), (0, 0))
        draw_hands(screen, result, font)
        for p in particles:
            p.draw(screen)
        if seal_model:
            draw_prediction(screen, font, big_font, seal_model, recent, seal, confidence)
        draw_sequence_info(screen, font, sequence, now)
        if flash_name:
            draw_flash(screen, huge_font, flash_name, now - flash_start)
        hint = font.render("Hold F for test fire - ESC to quit", True, (255, 255, 255))
        screen.blit(hint, (10, 10))
        pygame.display.flip()

        # 7. Wait so we run at about 30 FPS
        clock.tick(FPS)

    detector.close()
    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
