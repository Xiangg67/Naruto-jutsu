"""Stage 7c (part 1): arrange the labelled photos the way YOLO wants them.

YOLO needs two groups of photos:
    train = photos it studies
    val   = photos it is tested on, which it never studies

Just like in Stage 5, we split BY SESSION. Photos from the same SPACE hold are
nearly identical, so putting some in train and some in val would make the test
too easy and the score meaningless. Whole sessions go to one side or the other.

Run it with:  .\\venv\\Scripts\\python.exe prepare_dataset.py
It copies photos (your originals in dataset/raw/ are not touched) into dataset/yolo/.
"""

import itertools
import os
import shutil

from seals import SEAL_NAMES

RAW_DIR = "dataset/raw"
OUT_DIR = "dataset/yolo"
VAL_FRACTION = 0.25   # aim to keep about a quarter of each class's photos for testing


def session_of(filename):
    """Photo names look like  tiger_s027_0005.jpg  -> session 27."""
    return int(filename.split("_s")[1].split("_")[0])


def choose_val_sessions(sessions):
    """sessions = {session number: how many photos}. Pick whole sessions to hold out,
    so that their photos add up as close as possible to VAL_FRACTION of the class.
    We try every combination (there are only a few sessions, so this is quick)."""
    target = sum(sessions.values()) * VAL_FRACTION
    numbers = sorted(sessions)
    best, best_gap = None, None
    for size in range(1, len(numbers)):   # at least 1 session to test, at least 1 left to train
        for combo in itertools.combinations(numbers, size):
            gap = abs(sum(sessions[s] for s in combo) - target)
            if best_gap is None or gap < best_gap:
                best, best_gap = combo, gap
    return set(best) if best else set()


def main():
    # Start clean, so old files from an earlier run can't sneak in
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for part in ("train", "val"):
        os.makedirs(f"{OUT_DIR}/images/{part}")
        os.makedirs(f"{OUT_DIR}/labels/{part}")

    print(f"{'class':<7} {'train photos':>13} {'val photos':>11}   sessions held out for testing")
    for name in SEAL_NAMES + ["none"]:
        folder = os.path.join(RAW_DIR, name)
        if not os.path.isdir(folder):
            print(f"{name:<7} (no photos found)")
            continue
        photos = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))

        sessions = {}
        for photo in photos:
            sessions[session_of(photo)] = sessions.get(session_of(photo), 0) + 1
        val_sessions = choose_val_sessions(sessions)
        if not val_sessions:
            print(f"WARNING: '{name}' has only 1 session, so none of it can be used for testing!")

        counts = {"train": 0, "val": 0}
        for photo in photos:
            part = "val" if session_of(photo) in val_sessions else "train"
            shutil.copy(os.path.join(folder, photo), f"{OUT_DIR}/images/{part}/{photo}")

            label_file = photo[:-4] + ".txt"
            target = f"{OUT_DIR}/labels/{part}/{label_file}"
            if name == "none":
                open(target, "w").close()   # "none" photos get an EMPTY label file: nothing to find
            else:
                shutil.copy(os.path.join(folder, label_file), target)
            counts[part] += 1

        print(f"{name:<7} {counts['train']:>13} {counts['val']:>11}   {sorted(val_sessions)}")

    # data.yaml tells YOLO where the photos are and what the class numbers mean
    path = os.path.abspath(OUT_DIR).replace("\\", "/")
    lines = [f"path: {path}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(SEAL_NAMES)]
    with open(f"{OUT_DIR}/data.yaml", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR}/data.yaml")


if __name__ == "__main__":
    main()
