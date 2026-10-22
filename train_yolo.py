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

BASE_MODEL = "yolo11s.pt"            # the pre-trained starting point (a bit bigger than yolo11n; downloaded on first use)
DATA = "dataset/yolo/data.yaml"      # made by prepare_dataset.py
FINAL_MODEL = "models/seal_yolo.pt"  # where we keep our trained model
EPOCHS = 80                          # one epoch = one full pass over all training photos


def main():
    if not os.path.exists(DATA):
        raise SystemExit("dataset/yolo/data.yaml not found. Run prepare_dataset.py first.")

    # Ultralytics sends anonymous usage stats by default. We switch that off.
    settings.update({"sync": False})

    model = YOLO(BASE_MODEL)
    model.train(
        data=DATA,
        epochs=EPOCHS,
        patience=40,      # stop early if it hasn't improved for 40 epochs
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

        # Augmentation: every epoch, each photo is randomly changed a little (different
        # light and colours, shifted, zoomed, slightly rotated, patches blanked out, two
        # photos blended). We have few photos, so this "makes" many more variations.
        # More importantly it stops YOLO from memorising the ROOM (e.g. "blue board =
        # horse") instead of the hands. These stronger settings made it far more
        # reliable on photos from sessions it had never seen.
        degrees=8,        # rotate up to 8 degrees
        translate=0.2,    # shift up to 20%
        scale=0.7,        # zoom in or out by up to 70%
        hsv_h=0.03,       # shift colours
        hsv_s=0.8,        # change colour strength
        hsv_v=0.5,        # change brightness a lot (dark room, bright room)
        mixup=0.15,       # sometimes blend two photos together
        erasing=0.4,      # sometimes blank out a patch (hands partly hidden)
        close_mosaic=15,  # turn the "photo collage" trick off for the last 15 epochs
    )

    # Keep the best version (the epoch that scored best on the val photos)
    best = model.trainer.best
    shutil.copy(best, FINAL_MODEL)
    print(f"\nBest model copied to {FINAL_MODEL}")
    print(f"Charts, the confusion matrix and example predictions are in {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
