"""Stage 4: record examples of each hand seal into data/seals.csv.

How to use:
    1. Press a number key to choose which seal you are about to make.
    2. Make the seal and HOLD the SPACE bar. Every frame while you hold it
       is saved as one row (one "sample") in the CSV file.
    3. Let go of SPACE to stop. Each hold is saved as its own "session".

Run it with:  .\\venv\\Scripts\\python.exe collect_data.py
"""

import csv
import os

import pandas as pd
import pygame

from features import build_features, feature_names
from main import (WIDTH, HEIGHT, FPS, open_camera, prepare_frame, rgb_to_surface,
                  create_hand_detector, find_hands, draw_hands)

CSV_PATH = "data/seals.csv"

# Which number key picks which label.
# MediaPipe loses a hand when the hands overlap, so for now we use four shapes
# where BOTH hands stay fully visible, held a little apart (we can swap in the
# real Naruto seals later with YOLO):
#   palms  = both hands open, palms facing each other, fingers up
#   fists  = both hands closed into fists
#   points = both index fingers pointing up, other fingers curled
#   peace  = both hands making a V sign (index + middle up)
# "none" is the "no seal" class: hands down, random poses, half-made seals,
# hands moving between seals. Without it the model would call EVERYTHING a seal.
LABELS = [
    (pygame.K_0, "none"),
    (pygame.K_1, "palms"),
    (pygame.K_2, "fists"),
    (pygame.K_3, "points"),
    (pygame.K_4, "peace"),
]


def load_existing():
    """Read the CSV (if it exists). Return how many samples each label has
    and the biggest session number used so far."""
    counts = {name: 0 for _, name in LABELS}
    last_session = 0
    if os.path.exists(CSV_PATH):
        data = pd.read_csv(CSV_PATH)
        for label, how_many in data["label"].value_counts().items():
            counts[label] = int(how_many)
        if len(data) > 0:
            last_session = int(data["session_id"].max())
    return counts, last_session


def save_rows(rows):
    """Add rows to the end of the CSV file (and make the file + header if new)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    is_new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["session_id", "label"] + feature_names())
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples to {CSV_PATH}")


def draw_status(screen, font, big_font, label, counts, session_id, recording, hands_found):
    """Draw the info text on top of the camera picture."""
    # Dark box so the text is easy to read
    panel = pygame.Surface((330, 250))
    panel.set_alpha(170)
    panel.fill((0, 0, 0))
    screen.blit(panel, (10, 10))

    screen.blit(big_font.render(f"Label: {label}", True, (255, 220, 80)), (20, 16))

    y = 62
    for _, name in LABELS:
        marker = ">" if name == label else " "
        color = (255, 255, 255) if name == label else (170, 170, 170)
        screen.blit(font.render(f"{marker} {name}: {counts[name]}", True, color), (20, y))
        y += 26

    screen.blit(font.render(f"Hands seen: {hands_found}   Session: {session_id}", True, (255, 255, 255)), (20, y + 4))
    screen.blit(font.render("0-4 pick label, hold SPACE, ESC quit", True, (200, 200, 200)), (20, y + 30))

    if recording:
        pygame.draw.circle(screen, (255, 40, 40), (WIDTH - 40, 40), 16)
        screen.blit(big_font.render("REC", True, (255, 40, 40)), (WIDTH - 110, 22))


def main():
    cap = open_camera()
    if cap is None:
        print("Could not open any camera.")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Naruto Jutsu - collect data")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)
    big_font = pygame.font.SysFont("arial", 32, bold=True)

    detector = create_hand_detector()
    last_timestamp = -1

    counts, session_id = load_existing()
    label = LABELS[0][1]   # start on "none"
    recording = False
    rows = []              # samples recorded during the current SPACE hold

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
                    rows = []
                elif not recording:
                    # Number keys change the label (but not in the middle of recording)
                    for key, name in LABELS:
                        if event.key == key:
                            label = name
            elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE and recording:
                recording = False          # stop: write this session to the file
                save_rows(rows)
                rows = []

        # 2. Camera + hands (frames are prepared exactly like in main.py)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = prepare_frame(frame)
        timestamp = max(pygame.time.get_ticks(), last_timestamp + 1)
        last_timestamp = timestamp
        result = find_hands(detector, rgb, timestamp)
        hands_found = len(result.hand_landmarks)

        # 3. Record a sample. All our seals use BOTH hands, so for a seal we skip
        #    frames where MediaPipe lost a hand (saving those would teach the model
        #    that "one hand" means a seal). "none" keeps everything, because one
        #    hand, two hands or no hands are all normal parts of "none".
        if recording and (label == "none" or hands_found == 2):
            features = [round(x, 5) for x in build_features(result)]
            rows.append([session_id, label] + features)
            counts[label] += 1

        # 4. Draw
        screen.blit(rgb_to_surface(rgb), (0, 0))
        draw_hands(screen, result, font)
        draw_status(screen, font, big_font, label, counts, session_id, recording, hands_found)
        if recording and label != "none" and hands_found < 2:
            warning = big_font.render("Need BOTH hands visible - not saving!", True, (255, 60, 60))
            screen.blit(warning, (WIDTH // 2 - warning.get_width() // 2, HEIGHT - 60))
        pygame.display.flip()
        clock.tick(FPS)

    # If the window was closed while recording, still save what we have
    save_rows(rows)
    detector.close()
    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
