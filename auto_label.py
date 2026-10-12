"""Stage 7 helper: let the trained YOLO model draw the boxes on NEW photos, so you
only have to label the hard ones by hand.

For every photo in dataset/raw/<seal>/ that has no label yet, the model looks at
it. If it finds the seal that the FOLDER says it is (the folder name is the truth,
because you chose that label when recording), and it is very sure, its box is
saved as the label. Photos it is unsure about are left alone. Those are exactly the
ones worth labelling yourself with label_images.py, because they are what the
model finds hard.

It never changes photos and never overwrites a label that already exists.

Run it with:  .\\venv\\Scripts\\python.exe auto_label.py
"""

import os

from seals import SEAL_NAMES

RAW_DIR = "dataset/raw"
MODEL_PATH = "models/seal_yolo.pt"
AUTO_CONF = 0.75   # only trust the model's box if it is at least this sure


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"{MODEL_PATH} not found. Train a model first (train_yolo.py).")

    import torch
    from ultralytics import YOLO   # imported here because it takes a few seconds to load

    model = YOLO(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"{'seal':<7}{'new photos':>11}{'auto-labelled':>15}{'left for you':>14}")
    total_left = 0
    for class_id, name in enumerate(SEAL_NAMES):
        folder = os.path.join(RAW_DIR, name)
        if not os.path.isdir(folder):
            continue
        new_photos = [f for f in sorted(os.listdir(folder))
                      if f.endswith(".jpg") and not os.path.exists(os.path.join(folder, f[:-4] + ".txt"))]

        auto = 0
        for filename in new_photos:
            path = os.path.join(folder, filename)
            boxes = model.predict(path, imgsz=640, conf=AUTO_CONF, device=device, verbose=False)[0].boxes

            # Keep only boxes for the RIGHT seal (the one the folder says), take the surest
            best = None
            for box in boxes:
                if int(box.cls[0]) == class_id and (best is None or float(box.conf[0]) > float(best.conf[0])):
                    best = box
            if best is None:
                continue   # not sure enough: leave it for you to label by hand

            cx, cy, w, h = best.xywhn[0].tolist()   # the box as fractions of the picture size
            with open(os.path.join(folder, filename[:-4] + ".txt"), "w") as f:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            auto += 1

        left = len(new_photos) - auto
        total_left += left
        print(f"{name:<7}{len(new_photos):>11}{auto:>15}{left:>14}")

    print()
    if total_left:
        print(f"{total_left} photos still need a box: run label_images.py (it starts at the first unlabelled photo).")
    else:
        print("Nothing left to label.")


if __name__ == "__main__":
    main()
