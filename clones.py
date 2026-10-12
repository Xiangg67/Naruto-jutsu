"""Shadow clone: cut YOU out of the camera picture and paste ghostly copies of you
beside you.

MediaPipe's "selfie segmenter" looks at a picture and says, for every pixel, how
likely it is to belong to a person. We turn that into a mask (True = person).
To make a clone, we copy the person's pixels and paste them a few hundred pixels
to the side. Each clone uses a slightly OLDER camera frame, so the clones copy
your moves a moment late, like echoes.
"""

import os

import cv2
import mediapipe as mp
import numpy as np

SEGMENTER_PATH = "models/selfie_segmenter.tflite"

# Each clone is (sideways shift in pixels, how many frames behind it is).
# Negative shift = to the left. The far ones come first, so the near ones are drawn on top.
CLONE_SLOTS = [(-480, 14), (480, 14), (-250, 7), (250, 7)]
HISTORY_FRAMES = 16    # how many past frames we remember (must be more than the biggest delay)
GHOST_ALPHA = 0.8      # 1 = solid, lower = more see-through
GHOST_TINT = (0, 15, 40, 0)   # a touch of chakra blue, as (red, green, blue, unused)


def create_segmenter():
    """Load the person cut-out model. Returns None if the model file is missing."""
    if not os.path.exists(SEGMENTER_PATH):
        print(f"{SEGMENTER_PATH} not found - the Shadow Clone will only make smoke.")
        return None
    vision = mp.tasks.vision
    options = vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=SEGMENTER_PATH),
        running_mode=vision.RunningMode.IMAGE,   # one picture at a time
        output_confidence_masks=True,            # give us "how sure" for every pixel
        output_category_mask=False,
    )
    return vision.ImageSegmenter.create_from_options(options)


def person_mask(segmenter, rgb):
    """Return a (height, width) array that is True where the person is."""
    result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    confidence = result.confidence_masks[0].numpy_view()[:, :, 0]   # 0 = surely not a person, 1 = surely a person
    return confidence > 0.5


def make_entry(rgb, mask):
    """Prepare one camera frame to be remembered. We store four things:
    the frame, a blue-tinted ghost version of it (made ONCE here, not once per clone),
    the person mask, and the box around the person (so we only work on that part)."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return rgb, rgb, mask, None            # nobody in the picture: nothing to paste later
    box = (columns[0], rows[0], columns[-1] + 1, rows[-1] + 1)   # left, top, right, bottom
    ghost = cv2.add(cv2.convertScaleAbs(rgb, alpha=0.9), GHOST_TINT)   # a bit darker + a bit blue
    return rgb, ghost, mask, box


def paste_ghost(display, entry, shift):
    """Copy the person from a remembered frame onto `display`, moved `shift` pixels sideways."""
    _, ghost, mask, box = entry
    if box is None:
        return
    x0, y0, x1, y1 = box
    width = display.shape[1]

    # Where the moved person lands. Cut off whatever would fall outside the screen.
    left, right = x0 + shift, x1 + shift
    if left < 0:
        x0 -= left                             # (left is negative here, so x0 moves right)
        left = 0
    if right > width:
        x1 -= right - width
        right = width
    if x1 <= x0:
        return

    window = display[y0:y1, left:right]        # a window into display (changing it changes display)
    blended = cv2.addWeighted(window, 1 - GHOST_ALPHA, ghost[y0:y1, x0:x1], GHOST_ALPHA, 0)
    # Only the pixels that really are the person get the blended colours
    np.copyto(window, blended, where=mask[y0:y1, x0:x1, None])


def draw_clones(rgb, mask, history):
    """Return a copy of the current camera frame with the clones added.
    history = the recent entries from make_entry(), oldest first (the newest is the current frame)."""
    display = rgb.copy()
    for shift, delay in CLONE_SLOTS:
        if delay < len(history):               # wait until we remember enough frames
            paste_ghost(display, history[-1 - delay], shift)
    box = history[-1][3] if history else None
    if box:                                    # put the real you back on top, in front of the clones
        x0, y0, x1, y1 = box
        np.copyto(display[y0:y1, x0:x1], rgb[y0:y1, x0:x1], where=mask[y0:y1, x0:x1, None])
    return display
