"""Stage 7a: take photos of each real Naruto seal, to train a YOLO detector.

MediaPipe can't see hands that overlap, but YOLO looks at the whole picture,
so it can learn the real overlapping seals. It learns from photos.

How to use (same idea as collect_data.py):
    1. Press a number key to choose which seal you are about to make.
    2. Make the seal and HOLD the SPACE bar. A photo is saved a few times a second.
    3. Let go of SPACE to stop. Each hold is one "session".

Photos go to dataset/raw/<seal name>/. Press 0 to photograph "none": pictures with
NO seal in them (hands down, relaxed hands, half-made seals, hands moving).
YOLO needs these too, or it will see seals everywhere.

Run it with:  .\\venv\\Scripts\\python.exe capture_images.py
"""

import os

import cv2
import pygame

from main import WIDTH, HEIGHT, FPS, open_camera, prepare_frame, rgb_to_surface
from seals import SEAL_NAMES

RAW_DIR = "dataset/raw"
SAVE_EVERY = 3  # save every 3rd frame (about 8 photos a second), so we don't get 100 identical photos

# Which number key picks which label (the seal names live in seals.py)
LABELS = [
    (pygame.K_0, "none"),
    (pygame.K_1, SEAL_NAMES[0]),
    (pygame.K_2, SEAL_NAMES[1]),
    (pygame.K_3, SEAL_NAMES[2]),
    (pygame.K_4, SEAL_NAMES[3]),
]


def count_existing():
    """Look at the photos already saved. Return how many photos each label has,
    which sessions they came from, and the biggest session number used so far.
    Photo names look like  tiger_s003_0012.jpg  (label, session, photo number)."""
    counts = {name: 0 for _, name in LABELS}
    sessions = {name: set() for _, name in LABELS}
    last_session = 0
    for _, name in LABELS:
        folder = os.path.join(RAW_DIR, name)
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            if not filename.endswith(".jpg"):
                continue
            session = int(filename.split("_s")[1].split("_")[0])
            counts[name] += 1
            sessions[name].add(session)
            last_session = max(last_session, session)
    return counts, sessions, last_session


def save_photo(rgb, label, session_id, photo_number):
    """Save one frame as a JPEG. We save the same mirrored picture the app
    shows, because YOLO must see the same kind of picture when it runs live."""
    folder = os.path.join(RAW_DIR, label)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{label}_s{session_id:03d}_{photo_number:04d}.jpg")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # OpenCV saves in BGR order
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])


def draw_status(screen, font, big_font, label, counts, sessions, session_id, recording):
    """Draw the info text on top of the camera picture."""
    panel = pygame.Surface((400, 232))
    panel.set_alpha(170)
    panel.fill((0, 0, 0))
    screen.blit(panel, (10, 10))

    screen.blit(big_font.render(f"Label: {label}", True, (255, 220, 80)), (20, 16))

    y = 62
    for _, name in LABELS:
        marker = ">" if name == label else " "
        color = (255, 255, 255) if name == label else (170, 170, 170)
        text = f"{marker} {name}: {counts[name]} photos, {len(sessions[name])} sessions"
        screen.blit(font.render(text, True, color), (20, y))
        y += 26

    screen.blit(font.render(f"Session: {session_id}", True, (255, 255, 255)), (20, y + 4))
    screen.blit(font.render("0-4 pick label, hold SPACE, ESC quit", True, (200, 200, 200)), (20, y + 30))

    if recording:
        pygame.draw.circle(screen, (255, 40, 40), (WIDTH - 40, 40), 16)
        screen.blit(big_font.render("REC", True, (255, 40, 40)), (WIDTH - 130, 22))


def main():
    cap = open_camera()
    if cap is None:
        print("Could not open any camera.")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Naruto Jutsu - capture photos")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)
    big_font = pygame.font.SysFont("arial", 32, bold=True)

    counts, sessions, session_id = count_existing()
    label = LABELS[0][1]   # start on "none"
    recording = False
    frame_number = 0       # counts frames while recording, to save every few frames
    photo_number = 0       # numbers the photos inside one session

    running = True
    while running:
        # 1. Keys
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not recording:
                    recording = True       # start a new session
                    session_id += 1
                    frame_number = 0
                    photo_number = 0
                elif not recording:
                    for key, name in LABELS:   # number keys change the label
                        if event.key == key:
                            label = name
            elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                recording = False

        # 2. Camera
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = prepare_frame(frame)

        # 3. Save a photo every few frames while SPACE is held
        if recording:
            if frame_number % SAVE_EVERY == 0:
                photo_number += 1
                save_photo(rgb, label, session_id, photo_number)
                counts[label] += 1
                sessions[label].add(session_id)
            frame_number += 1

        # 4. Draw
        screen.blit(rgb_to_surface(rgb), (0, 0))
        draw_status(screen, font, big_font, label, counts, sessions, session_id, recording)
        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
