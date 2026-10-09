"""Stage 1: show the mirrored webcam in a pygame window.

OpenCV only grabs frames from the camera.
pygame is what actually draws them on screen.
"""

import sys

import cv2
import pygame

WIDTH, HEIGHT = 960, 540  # window size
FPS = 30                  # frames per second we aim for


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


def frame_to_surface(frame):
    """Turn an OpenCV frame into a pygame picture (surface)."""
    frame = cv2.flip(frame, 1)                       # mirror, like a real mirror
    frame = cv2.resize(frame, (WIDTH, HEIGHT))       # fit the window
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV uses BGR, pygame uses RGB
    frame = frame.swapaxes(0, 1)                     # OpenCV is (height, width), pygame is (width, height)
    return pygame.surfarray.make_surface(frame)


def main():
    cap = open_camera()
    if cap is None:
        print("Could not open any camera (tried 0, 1 and 2).")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Naruto Jutsu")
    clock = pygame.time.Clock()

    running = True
    while running:
        # 1. Handle events: closing the window or pressing ESC quits
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # 2. Grab a frame from the camera
        ok, frame = cap.read()
        if not ok:
            continue  # skip this round if the camera had no picture

        # 3. Draw it
        screen.blit(frame_to_surface(frame), (0, 0))
        pygame.display.flip()

        # 4. Wait so we run at about 30 FPS
        clock.tick(FPS)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
