"""Visual effects. For now: fire particles.

A "particle" is one tiny glowing blob. Draw a few hundred of them
moving upward and fading out, and together they look like fire.
"""

import random

import numpy as np
import pygame

# Glow pictures are made once and reused (making them every frame would be slow).
# The key is (radius, fade step) and the value is the ready-made picture.
glow_cache = {}


def make_glow(radius, t):
    """Make a soft round glow picture. t goes from 1 (fresh, bright) to 0 (dying)."""
    # Colour changes as the flame ages: yellow -> orange -> dark red
    brightness = t ** 0.7
    red = 255 * brightness
    green = (60 + 150 * t) * brightness
    blue = 20 * t * brightness

    # For every pixel, how far is it from the centre of the circle?
    ys, xs = np.mgrid[-radius:radius, -radius:radius]
    distance = np.sqrt(xs ** 2 + ys ** 2)

    # Bright in the middle, fading to black at the edge
    strength = np.clip(1 - distance / radius, 0, 1) ** 2

    # Build an (width, height, 3) colour array and turn it into a picture
    pixels = np.zeros((radius * 2, radius * 2, 3))
    pixels[:, :, 0] = strength * red
    pixels[:, :, 1] = strength * green
    pixels[:, :, 2] = strength * blue
    return pygame.surfarray.make_surface(pixels.astype(np.uint8)).convert()


def get_glow(radius, t):
    """Get a glow picture from the cache, making it if it doesn't exist yet."""
    radius = max(4, (radius // 4) * 4)  # round to steps of 4 so we make fewer pictures
    step = int(t * 12)                  # round the fade to 12 steps for the same reason
    key = (radius, step)
    if key not in glow_cache:
        glow_cache[key] = make_glow(radius, step / 12)
    return glow_cache[key]


def warm_up_glows():
    """Make every glow picture ahead of time (at startup), so the game never
    freezes for a moment later when a big flame needs a size it hasn't made yet."""
    for radius in range(4, 72, 4):   # up to the biggest jutsu flame (about 68 pixels)
        for step in range(13):
            glow_cache[(radius, step)] = make_glow(radius, step / 12)


def spawn_burst(particles, x, y, count=18):
    """Add a big cloud of fire around (x, y). Call it every frame for a while
    to build up a roaring fireball."""
    for _ in range(count):
        px = x + random.uniform(-70, 70)   # spread the start points around
        py = y + random.uniform(-40, 40)
        particles.append(Particle(px, py, scale=2.0))


class Particle:
    """One flame blob: it moves up, shrinks, and fades until its life runs out."""

    def __init__(self, x, y, scale=1.0):
        # scale makes the blob bigger and faster (1 = normal fire, 2 = big jutsu fire)
        self.x = x
        self.y = y
        self.vx = random.uniform(-1.5, 1.5) * scale   # sideways speed (pixels per frame)
        self.vy = random.uniform(-6, -2) * scale      # upward speed (negative = up on screen)
        self.max_life = random.randint(20, 40)        # lives for 20-40 frames
        self.life = self.max_life
        self.size = random.uniform(18, 34) * scale    # starting radius in pixels

    def update(self):
        """Move one frame forward."""
        self.x += self.vx
        self.y += self.vy
        self.vx += random.uniform(-0.3, 0.3)  # small random wobble makes it flicker
        self.vy *= 0.97                       # slow down a little as it rises
        self.life -= 1

    def is_dead(self):
        return self.life <= 0

    def draw(self, screen):
        t = self.life / self.max_life         # 1 when new, 0 when dead
        radius = int(self.size * (0.4 + 0.6 * t))  # shrinks as it ages
        glow = get_glow(radius, t)
        half = glow.get_width() // 2
        # BLEND_RGBA_ADD adds the glow's colours to what is already on screen,
        # so overlapping particles get brighter, just like real light.
        screen.blit(glow, (self.x - half, self.y - half), special_flags=pygame.BLEND_RGBA_ADD)
