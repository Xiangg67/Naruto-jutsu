"""Stage 7c (part 2): train YOLO on your photos, using the GPU.

We don't train from nothing. We start from yolo11n.pt, a small YOLO model that
already knows how to see edges, shapes and hands (it was trained on millions of
photos). Training then teaches it OUR four seals. This is called fine-tuning.
It is much faster and needs far fewer photos than starting from scratch.

Run prepare_dataset.py first, then:  .\\venv\\Scripts\\python.exe train_yolo.py
When it finishes, the best model is copied to models/seal_yolo.pt.
"""

import os
import shutil

from ultralytics import YOLO, settings

BASE_MODEL = "yolo11n.pt"            # the pre-trained starting point (in the project folder)
DATA = "dataset/yolo/data.yaml"      # made by prepare_dataset.py
FINAL_MODEL = "models/seal_yolo.pt"  # where we keep our trained model
EPOCHS = 60                          # one epoch = one full pass over all training photos


def main():
    if not os.path.exists(BASE_MODEL):
        raise SystemExit(f"{BASE_MODEL} not found in the project folder.")
    if not os.path.exists(DATA):
        raise SystemExit("dataset/yolo/data.yaml not found. Run prepare_dataset.py first.")

    # Ultralytics sends anonymous usage stats by default. We switch that off.
    settings.update({"sync": False})

    model = YOLO(BASE_MODEL)
    model.train(
        data=DATA,
        epochs=EPOCHS,
        patience=20,      # stop early if it hasn't improved for 20 epochs
        imgsz=640,        # photos are shrunk so the long side is 640 pixels
        batch=16,         # photos processed at once (fits easily in 8 GB)
        device=0,         # 0 = the first GPU
        workers=2,        # helper processes that load photos
        project=os.path.abspath("runs"),   # results go in runs/seals/
        name="seals",
        exist_ok=True,    # reuse the runs/seals folder instead of making seals2, seals3...
        fliplr=0.0,       # do NOT mirror photos while training: our photos are already
                          # mirrored the same way as the live camera, and flipping would
                          # swap left and right hands, which changes how a seal looks
    )

    # Keep the best version (the epoch that scored best on the val photos)
    best = model.trainer.best
    shutil.copy(best, FINAL_MODEL)
    print(f"\nBest model copied to {FINAL_MODEL}")
    print(f"Charts, the confusion matrix and example predictions are in {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
