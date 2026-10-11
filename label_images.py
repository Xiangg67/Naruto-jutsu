"""Stage 7b: draw a box around the seal in every photo, so YOLO knows WHERE it is.

YOLO learns from examples: "in THIS photo, the seal is in THIS box".
Each box is saved in a small text file next to its photo, in YOLO's format:
    class_number  box_center_x  box_center_y  box_width  box_height
(the last four are fractions of the picture size, from 0 to 1).

How to use:
    * The box from the previous photo is copied onto each new photo.
      If it already fits the seal, just press SPACE.
    * If it doesn't fit, drag the mouse to draw a new box. Draw it TIGHT around
      BOTH hands of the seal (the hands, not your whole body).
    * SPACE (or ENTER) saves the box and goes to the next photo.
    * X rejects a bad photo (blurry, a hand is missing, wrong seal). It is moved
      to dataset/rejected/, not deleted.
    * LEFT / RIGHT arrows move between photos without saving. ESC quits;
      your work is saved after every photo, so you can continue later.

Photos of "none" need no box (there is nothing to find), so they are skipped.

Run it with:  .\\venv\\Scripts\\python.exe label_images.py
"""

import os
import shutil

import pygame

from seals import SEAL_NAMES

RAW_DIR = "dataset/raw"
REJECT_DIR = "dataset/rejected"
IMG_W, IMG_H = 960, 540   # photo size (capture_images.py saves photos this size)
BAR_H = 70                # space under the photo for the text
MIN_BOX = 10              # boxes smaller than this many pixels are ignored (accidental clicks)


def find_photos():
    """List every seal photo as (seal name, file path), in order. Skips "none"."""
    items = []
    for name in SEAL_NAMES:
        folder = os.path.join(RAW_DIR, name)
        if os.path.isdir(folder):
            for filename in sorted(os.listdir(folder)):
                if filename.endswith(".jpg"):
                    items.append((name, os.path.join(folder, filename)))
    return items


def label_path(photo_path):
    """Where a photo's label file lives: same name, but .txt instead of .jpg."""
    return photo_path[:-4] + ".txt"


def read_box(photo_path):
    """Load a saved box as pixels (x1, y1, x2, y2), or None if the photo has no label yet."""
    path = label_path(photo_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        _, cx, cy, w, h = f.read().split()
    cx, cy, w, h = float(cx) * IMG_W, float(cy) * IMG_H, float(w) * IMG_W, float(h) * IMG_H
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def save_box(photo_path, class_id, box):
    """Save a pixel box (x1, y1, x2, y2) in YOLO format."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / IMG_W, (y1 + y2) / 2 / IMG_H   # centre, as fractions
    w, h = (x2 - x1) / IMG_W, (y2 - y1) / IMG_H              # size, as fractions
    with open(label_path(photo_path), "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def reject_photo(name, photo_path):
    """Move a bad photo (and its label, if any) out of the way, to dataset/rejected/."""
    folder = os.path.join(REJECT_DIR, name)
    os.makedirs(folder, exist_ok=True)
    shutil.move(photo_path, os.path.join(folder, os.path.basename(photo_path)))
    if os.path.exists(label_path(photo_path)):
        shutil.move(label_path(photo_path), os.path.join(folder, os.path.basename(label_path(photo_path))))


def make_box(start, end):
    """Turn two corner points (in any order) into a box (x1, y1, x2, y2)."""
    return min(start[0], end[0]), min(start[1], end[1]), max(start[0], end[0]), max(start[1], end[1])


def clamp(pos):
    """Keep a mouse position inside the photo."""
    return max(0, min(IMG_W - 1, pos[0])), max(0, min(IMG_H - 1, pos[1]))


def main():
    items = find_photos()
    if not items:
        print("No seal photos found in dataset/raw/. Run capture_images.py first.")
        return

    pygame.init()
    screen = pygame.display.set_mode((IMG_W, IMG_H + BAR_H))
    pygame.display.set_caption("Naruto Jutsu - label photos")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)
    big_font = pygame.font.SysFont("arial", 24, bold=True)

    index = 0
    # Start at the first photo that has no label yet
    for i, (_, path) in enumerate(items):
        if not os.path.exists(label_path(path)):
            index = i
            break
    else:
        index = len(items)  # everything is already labelled

    last_box = None        # the box from the previous photo, copied onto the next one
    box = None             # the box on the current photo
    box_state = "none"     # "saved" (loaded from file), "copied" (from previous photo) or "new" (drawn now)
    image = None
    drag_start = None      # where the mouse button went down (None = not dragging)
    message = ""

    def load_current():
        """Load the photo at `index` and decide which box to show on it."""
        nonlocal image, box, box_state
        if index >= len(items):
            return
        _, path = items[index]
        image = pygame.image.load(path).convert()
        if image.get_size() != (IMG_W, IMG_H):
            image = pygame.transform.smoothscale(image, (IMG_W, IMG_H))
        saved = read_box(path)
        if saved:
            box, box_state = saved, "saved"
        elif last_box:
            box, box_state = last_box, "copied"
        else:
            box, box_state = None, "none"

    load_current()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and event.pos[1] < IMG_H:
                drag_start = clamp(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and drag_start:
                new_box = make_box(drag_start, clamp(event.pos))
                if new_box[2] - new_box[0] >= MIN_BOX and new_box[3] - new_box[1] >= MIN_BOX:
                    box, box_state = new_box, "new"
                    message = ""
                drag_start = None

            elif event.type == pygame.KEYDOWN and index < len(items):
                name, path = items[index]
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if box:
                        save_box(path, SEAL_NAMES.index(name), box)
                        last_box = box
                        index += 1
                        message = ""
                        load_current()
                    else:
                        message = "Draw a box first (drag with the mouse)"
                elif event.key == pygame.K_x:
                    reject_photo(name, path)
                    del items[index]       # the next photo slides into this position
                    message = "Photo rejected (moved to dataset/rejected)"
                    load_current()
                elif event.key == pygame.K_RIGHT and index < len(items) - 1:
                    index += 1
                    load_current()
                elif event.key == pygame.K_LEFT and index > 0:
                    index -= 1
                    load_current()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT and index > 0:
                index -= 1     # from the "all done" screen, go back one photo
                load_current()

        # --- Draw ---
        screen.fill((25, 25, 25))
        done = sum(os.path.exists(label_path(path)) for _, path in items)

        if index >= len(items):
            text = big_font.render("All photos are labelled!  Press ESC to finish.", True, (120, 255, 120))
            screen.blit(text, (IMG_W // 2 - text.get_width() // 2, IMG_H // 2))
        else:
            name, path = items[index]
            screen.blit(image, (0, 0))

            # The box (green = saved, yellow = copied from the previous photo, cyan = just drawn)
            shown = make_box(drag_start, clamp(pygame.mouse.get_pos())) if drag_start else box
            if shown:
                color = {"saved": (80, 255, 80), "copied": (255, 220, 60), "new": (60, 220, 255)}.get(box_state, (60, 220, 255))
                if drag_start:
                    color = (60, 220, 255)
                rect = pygame.Rect(shown[0], shown[1], shown[2] - shown[0], shown[3] - shown[1])
                pygame.draw.rect(screen, color, rect, 3)
                tag = font.render(name, True, (0, 0, 0), color)
                screen.blit(tag, (rect.x, max(0, rect.y - 24)))

            states = {"saved": "already labelled", "copied": "box COPIED from the previous photo - check it fits!",
                      "new": "new box - press SPACE to save", "none": "no box yet - drag one"}
            line1 = f"{name}   photo {index + 1}/{len(items)}   labelled: {done}/{len(items)}   {states[box_state]}"
            screen.blit(big_font.render(line1, True, (255, 255, 255)), (10, IMG_H + 4))

        line2 = "Drag = draw box | SPACE = save + next | X = reject photo | arrows = move | ESC = quit"
        screen.blit(font.render(line2, True, (170, 170, 170)), (10, IMG_H + 32))
        if message:
            screen.blit(font.render(message, True, (255, 120, 120)), (10, IMG_H + 50))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    done = sum(os.path.exists(label_path(path)) for _, path in items)
    print(f"Labelled {done} of {len(items)} photos.")


if __name__ == "__main__":
    main()
