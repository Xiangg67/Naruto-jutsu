"""The real seals YOLO learns. Every Stage 7 script imports this one list.

The ORDER matters: YOLO numbers the classes 0, 1, 2, 3 in this order
(tiger = 0, ram = 1, ...). If you change the list after labelling photos,
the labels will point at the wrong seals!
"""

SEAL_NAMES = ["tiger", "ram", "snake", "horse"]
