"""Stage 7: webcam + YOLO seal detection + jutsu sequences + fire effects.

OpenCV only grabs frames from the camera.
YOLO (trained by train_yolo.py) finds the seal in the picture, boxed, and names it.
jutsu.py watches the seals go by and spots a complete jutsu.
pygame draws everything on screen (F makes test fire, H shows MediaPipe's hand skeleton).
"""

import os
import random
import sys
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame

from effects import (Particle, spawn_aura, spawn_ring, spawn_smoke, run_effect, warm_up_glows,
                     EFFECT_FRAMES, EFFECT_FLASH, EFFECT_SHAKE, EFFECT_TEXT)
from features import get_hand_label
from jutsu import JUTSUS, JUTSU_EFFECTS, JUTSU_SOUNDS, SealSequence

WIDTH, HEIGHT = 960, 540  # window size
FPS = 30                  # frames per second we aim for
PARTICLES_PER_FRAME = 6   # how much fire to make each frame while F is held
MAX_PARTICLES = 400       # safety limit so the game never gets slow

MAX_BURST_PARTICLES = 700 # safety limit while a jutsu effect is going
FLASH_SECONDS = 2.5       # how long the jutsu name stays on screen
SCREEN_FLASH_FRAMES = 10  # the whole screen flashes for this many frames when a jutsu fires
SHAKE_FRAMES = 14         # and shakes for this many frames

YOLO_MODEL_PATH = "models/seal_yolo.pt"
YOLO_SIZE = 640           # YOLO shrinks the picture so its long side is this many pixels
# On photos from sessions it had never seen, the real seals scored 0.66 or more, and
# photos with NO seal never got above 0.5.
YOLO_CONF = 0.4           # YOLO throws away boxes it is less than 40% sure about
SMOOTH_FRAMES = 5         # average the last 5 frames to stop flicker
MIN_CONFIDENCE = 0.5      # after averaging, a seal must reach this confidence to count

MODEL_PATH = "models/hand_landmarker.task"  # MediaPipe's hand model (only used for the H overlay)
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
    EVERY script must prepare frames this way, so the picture always matches
    what the models were trained on."""
    frame = cv2.flip(frame, 1)                       # mirror, like a real mirror
    frame = cv2.resize(frame, (WIDTH, HEIGHT))       # fit the window
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # OpenCV uses BGR, pygame and MediaPipe use RGB


def rgb_to_surface(rgb):
    """Turn an RGB frame into a pygame picture (surface)."""
    # OpenCV is (height, width), pygame is (width, height), so swap the first two axes
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def load_yolo():
    """Load the trained YOLO model. Returns (model, device), or (None, None) if it isn't trained yet."""
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"{YOLO_MODEL_PATH} not found. Run prepare_dataset.py and train_yolo.py first.")
        return None, None

    # Imported here (not at the top) because YOLO takes a few seconds to load, and
    # the other scripts that borrow functions from this file don't need it.
    import torch
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"  # 0 = your graphics card
    print("YOLO running on", "the GPU" if device == 0 else "the CPU (slow!)")
    # The very first prediction is slow (the GPU wakes up), so do one now on a blank picture
    model.predict(np.zeros((HEIGHT, WIDTH, 3), np.uint8), imgsz=YOLO_SIZE, device=device, verbose=False)
    return model, device


def find_seals(model, device, rgb):
    """Ask YOLO which seals are in this frame.
    Returns (scores, best):
        scores = the best confidence for each seal (0 if YOLO didn't see that seal)
        best   = (x1, y1, x2, y2, name, confidence) for the single surest box, or None"""
    # YOLO expects OpenCV's blue-green-red order (like the photos it trained on),
    # so we flip our RGB frame back. [:, :, ::-1] reverses the colour channels.
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    boxes = model.predict(bgr, imgsz=YOLO_SIZE, conf=YOLO_CONF, device=device, verbose=False)[0].boxes

    scores = np.zeros(len(model.names))
    best = None
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        scores[class_id] = max(scores[class_id], confidence)
        if best is None or confidence > best[5]:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best = (x1, y1, x2, y2, model.names[class_id], confidence)
    return scores, best


def pick_seal(names, recent):
    """Average the last few frames, then pick the seal with the highest average.
    Returns (seal name or None, confidence). Not sure enough gives None."""
    average = np.mean(recent, axis=0)
    best = int(np.argmax(average))
    confidence = float(average[best])
    if confidence < MIN_CONFIDENCE:
        return None, confidence
    return names[best], confidence


def draw_prediction(screen, font, big_font, names, recent, seal, confidence):
    """Show the detected seal at the bottom, with every seal's average score underneath."""
    if seal:
        text = big_font.render(f"{seal.upper()}  {confidence:.0%}", True, (255, 220, 80))
    else:
        text = big_font.render("no seal", True, (170, 170, 170))
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 90))

    # Small line with the (smoothed) score of each seal, handy for checking stability
    average = np.mean(recent, axis=0)
    scores = "   ".join(f"{name} {p:.0%}" for name, p in zip(names, average))
    line = font.render(scores, True, (255, 255, 255))
    screen.blit(line, (WIDTH // 2 - line.get_width() // 2, HEIGHT - 40))


def draw_seal_box(screen, font, best):
    """Draw the box YOLO found around the seal, with its name."""
    x1, y1, x2, y2, name, confidence = best
    color = (255, 220, 80)
    pygame.draw.rect(screen, color, (x1, y1, x2 - x1, y2 - y1), 3)
    tag = font.render(f"{name} {confidence:.0%}", True, (0, 0, 0), color)
    screen.blit(tag, (x1, max(0, y1 - 28)))


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


def load_sound(path):
    """Load a sound file. The game still works (silently) if it is missing."""
    try:
        return pygame.mixer.Sound(path)
    except (pygame.error, FileNotFoundError):
        print(f"Could not load {path} - running without that sound.")
        return None


def draw_panel(screen, x, y, w, h, alpha=140):
    """A dark see-through box, so white text stays easy to read on any camera picture."""
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (0, 0, 0, alpha), (0, 0, w, h), border_radius=10)
    screen.blit(panel, (x, y))


def draw_sequence_info(screen, font, sequence, now):
    """Show the jutsu recipes, the seals confirmed so far, and the hold progress bar."""
    # Recipes (top left, under the hint), on a dark panel
    lines = [font.render(f"{name}: {' > '.join(seals)}", True, (255, 255, 255)) for name, seals in JUTSUS.items()]
    draw_panel(screen, 6, 42, max(line.get_width() for line in lines) + 20, 26 * len(lines) + 12)
    y = 48
    for line in lines:
        screen.blit(line, (16, y))
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


def draw_flash(screen, huge_font, name, age, color):
    """Flash the jutsu name in the middle of the screen, fading out at the end."""
    if age > FLASH_SECONDS:
        return
    fade_start = FLASH_SECONDS * 0.6
    alpha = 255 if age < fade_start else int(255 * (1 - (age - fade_start) / (FLASH_SECONDS - fade_start)))
    words = f"{name} Jutsu!"
    shadow = huge_font.render(words, True, (0, 0, 0))
    text = huge_font.render(words, True, color)
    shadow.set_alpha(alpha)
    text.set_alpha(alpha)
    x = WIDTH // 2 - text.get_width() // 2
    y = HEIGHT // 2 - 95   # below the recipe list, above the seal prompt
    screen.blit(shadow, (x + 4, y + 4))
    screen.blit(text, (x, y))


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

    yolo, device = load_yolo()
    names = list(yolo.names.values()) if yolo else []   # seal names, in YOLO's class order
    recent = deque(maxlen=SMOOTH_FRAMES)  # the last few frames' scores (old ones drop off automatically)
    sequence = SealSequence()             # remembers confirmed seals and spots jutsu

    hand_detector = None   # MediaPipe is only loaded if you press H
    show_hands = False
    last_timestamp = -1    # MediaPipe needs each frame's timestamp to be bigger than the last

    fire_sound = load_sound("assets/fire.wav")    # test fire (F key)
    seal_sound = load_sound("assets/seal.wav")    # a short ding for each confirmed seal
    # Each jutsu has its own sound (see JUTSU_SOUNDS in jutsu.py)
    jutsu_sounds = {name: load_sound("assets/" + file) for name, file in JUTSU_SOUNDS.items()}

    world = pygame.Surface((WIDTH, HEIGHT))        # camera + effects are drawn here, so they can shake
    flash_surface = pygame.Surface((WIDTH, HEIGHT))  # a plain colour, laid over the screen for the flash

    particles = []  # all the fire / chakra / smoke blobs that are alive right now
    bolts = []      # lightning strikes that are alive right now
    effect_kind = "fire"                           # which effect the last jutsu used
    effect_age = 0                                 # frames since that effect started
    effect_frames_left = 0                         # how long it still has to run (0 = none running)
    flash_frames = 0                               # frames of screen flash left
    shake_frames = 0                               # frames of screen shake left
    last_center = (WIDTH // 2, HEIGHT // 2)        # where the seal was last seen
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
                elif event.key == pygame.K_h:
                    show_hands = not show_hands
                    if show_hands and hand_detector is None:
                        hand_detector = create_hand_detector()

        # 2. Grab a frame from the camera and prepare it (mirror + RGB)
        ok, frame = cap.read()
        if not ok:
            continue  # skip this round if the camera had no picture
        rgb = prepare_frame(frame)

        # 3. Find the seal with YOLO, and smooth it over the last few frames
        seal, confidence, best_box = None, 0.0, None
        if yolo:
            scores, best_box = find_seals(yolo, device, rgb)
            recent.append(scores)
            seal, confidence = pick_seal(names, recent)
            if best_box:  # remember where the seal is: the jutsu fire will appear there
                last_center = (int((best_box[0] + best_box[2]) / 2), int((best_box[1] + best_box[3]) / 2))

        # 3b. Optional: MediaPipe's hand skeleton, only for show
        hands_result = None
        if show_hands:
            last_timestamp = max(pygame.time.get_ticks(), last_timestamp + 1)
            hands_result = find_hands(hand_detector, rgb, last_timestamp)

        # 3c. Feed the seal to the sequence matcher (it needs the time in seconds)
        now = pygame.time.get_ticks() / 1000
        confirmed, jutsu = sequence.update(seal, now)
        if jutsu:
            # A whole jutsu! These lines run ONCE, at the moment it is recognised,
            # so the sound plays once (not every frame).
            sound = jutsu_sounds.get(jutsu)
            if sound:
                sound.play()
            effect_kind = JUTSU_EFFECTS.get(jutsu, "fire")
            effect_age, effect_frames_left = 0, EFFECT_FRAMES[effect_kind]
            flash_frames, shake_frames = SCREEN_FLASH_FRAMES, SHAKE_FRAMES
            flash_name, flash_start = jutsu, now
        elif confirmed:
            # One more seal confirmed: a ding, a puff of smoke and a small ring of chakra
            if seal_sound:
                seal_sound.play()
            spawn_smoke(particles, last_center[0], last_center[1], count=4)
            spawn_ring(particles, last_center[0], last_center[1], count=16, speed=(3, 6))

        # 4. Effects
        keys = pygame.key.get_pressed()
        if keys[pygame.K_f] and len(particles) < MAX_PARTICLES:
            for _ in range(PARTICLES_PER_FRAME):
                particles.append(Particle(WIDTH // 2, HEIGHT // 2))   # test fire at the centre
        if best_box and seal and len(particles) < MAX_PARTICLES:
            # Chakra gathers around the seal while you hold it, and grows as the hold fills up
            spawn_aura(particles, best_box[:4], sequence.hold_progress(now))
        if effect_frames_left > 0:
            # The big jutsu effect, at your hands
            run_effect(effect_kind, effect_age, last_center[0], last_center[1], particles, bolts)
            effect_age += 1
            effect_frames_left -= 1
            if len(particles) > MAX_BURST_PARTICLES:
                del particles[:len(particles) - MAX_BURST_PARTICLES]   # drop the oldest ones

        # 5. Move everything, then throw away what has faded out
        for p in particles:
            p.update()
        particles = [p for p in particles if not p.is_dead()]
        for b in bolts:
            b.update()
        bolts = [b for b in bolts if not b.is_dead()]

        # 6. Draw. First the "world" (camera, hands, seal box, effects), which can shake.
        world.blit(rgb_to_surface(rgb), (0, 0))
        if hands_result:
            draw_hands(world, hands_result, font)
        if best_box:
            draw_seal_box(world, font, best_box)
        for p in particles:
            if not p.additive:
                p.draw(world)        # smoke is painted first...
        for p in particles:
            if p.additive:
                p.draw(world)        # ...then the glowing fire and chakra on top of it
        for b in bolts:
            b.draw(world)

        shake_x = shake_y = 0
        if shake_frames > 0:
            strength = int(EFFECT_SHAKE[effect_kind] * shake_frames / SHAKE_FRAMES)  # shakes less and less
            shake_x, shake_y = random.randint(-strength, strength), random.randint(-strength, strength)
            shake_frames -= 1
        screen.fill((0, 0, 0))
        screen.blit(world, (shake_x, shake_y))

        if flash_frames > 0:
            flash_surface.fill(EFFECT_FLASH[effect_kind])
            flash_surface.set_alpha(int(190 * flash_frames / SCREEN_FLASH_FRAMES))   # fades out quickly
            screen.blit(flash_surface, (0, 0))
            flash_frames -= 1

        # Then the text, which stays still. A dark panel behind it keeps it readable.
        draw_panel(screen, WIDTH // 2 - 290, HEIGHT - 172, 580, 168)
        if yolo:
            draw_prediction(screen, font, big_font, names, recent, seal, confidence)
        else:
            warning = font.render("No YOLO model yet - run train_yolo.py", True, (255, 120, 120))
            screen.blit(warning, (WIDTH // 2 - warning.get_width() // 2, HEIGHT - 60))
        draw_sequence_info(screen, font, sequence, now)
        if flash_name:
            draw_flash(screen, huge_font, flash_name, now - flash_start, EFFECT_TEXT[effect_kind])
        hint = font.render("F = test fire   H = hand skeleton   ESC = quit", True, (255, 255, 255))
        draw_panel(screen, 6, 6, hint.get_width() + 20, 32)
        screen.blit(hint, (16, 10))
        pygame.display.flip()

        # 7. Wait so we run at about 30 FPS
        clock.tick(FPS)

    if hand_detector:
        hand_detector.close()
    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
