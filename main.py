"""Stage 2: mirrored webcam in a pygame window, plus fire on the F key.

OpenCV only grabs frames from the camera.
pygame is what actually draws them on screen.
"""

import sys

import cv2
import pygame

from effects import Particle

WIDTH, HEIGHT = 960, 540  # window size
FPS = 30                  # frames per second we aim for
PARTICLES_PER_FRAME = 6   # how much fire to make each frame while F is held
MAX_PARTICLES = 400       # safety limit so the game never gets slow


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

    # Audio settings must be set BEFORE pygame.init() or sounds can lag
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Naruto Jutsu")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    # Load the fire sound (the game still works without it)
    try:
        fire_sound = pygame.mixer.Sound("assets/fire.wav")
    except (pygame.error, FileNotFoundError):
        print("Could not load assets/fire.wav - running without sound.")
        fire_sound = None

    particles = []  # all the fire blobs that are alive right now

    running = True
    while running:
        # 1. Handle events: closing the window or pressing ESC quits
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f and fire_sound:
                    # KEYDOWN happens ONCE per press (not every frame while held),
                    # so the sound plays once per press.
                    fire_sound.play()

        # 2. Grab a frame from the camera
        ok, frame = cap.read()
        if not ok:
            continue  # skip this round if the camera had no picture

        # 3. While F is held down, make new fire at the centre of the screen
        keys = pygame.key.get_pressed()
        if keys[pygame.K_f] and len(particles) < MAX_PARTICLES:
            for _ in range(PARTICLES_PER_FRAME):
                particles.append(Particle(WIDTH // 2, HEIGHT // 2))

        # 4. Move every particle, then throw away the ones that faded out
        for p in particles:
            p.update()
        particles = [p for p in particles if not p.is_dead()]

        # 5. Draw: camera first, fire on top, then a hint text
        screen.blit(frame_to_surface(frame), (0, 0))
        for p in particles:
            p.draw(screen)
        hint = font.render("Hold F for fire - ESC to quit", True, (255, 255, 255))
        screen.blit(hint, (10, 10))
        pygame.display.flip()

        # 6. Wait so we run at about 30 FPS
        clock.tick(FPS)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
