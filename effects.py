"""Visual effects: fire, chakra, smoke and lightning.

A "particle" is one tiny soft blob. Draw a few hundred of them moving and
fading and together they look like fire, a glowing aura or a smoke cloud.
Lightning is different: it is drawn as jagged glowing lines.

main.py only needs three things from here:
    spawn_aura(...)   chakra rising around a seal while you hold it
    run_effect(...)   the big effect for a finished jutsu
    Particle / Lightning objects to update() and draw() every frame
"""

import math
import random

import numpy as np
import pygame

# Glow pictures are made once and reused (making them every frame would be slow).
# The key is (palette, radius, fade step) and the value is the ready-made picture.
glow_cache = {}


def make_glow(radius, t, palette):
    """Make a soft round picture. t goes from 1 (fresh, bright) to 0 (dying).
    Glowing palettes ("fire", "chakra") are black at the edge, because they get ADDED
    to the camera picture. Smoke is grey and see-through at the edge, because it is
    painted OVER the camera picture."""
    # For every pixel, how far is it from the centre of the circle?
    ys, xs = np.mgrid[-radius:radius, -radius:radius]
    distance = np.sqrt(xs ** 2 + ys ** 2)

    if palette == "smoke":
        strength = np.clip(1 - distance / radius, 0, 1) ** 1.3
        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        surface.fill((210, 210, 215, 0))               # light grey, fully see-through for now
        alpha = pygame.surfarray.pixels_alpha(surface)
        alpha[:, :] = (strength * 215 * t ** 0.7).astype(np.uint8)  # thick in the middle, thin at the edge
        del alpha                                      # let go of the pixels so the picture is usable again
        return surface.convert_alpha()

    brightness = t ** 0.7
    if palette == "chakra":
        # Blue energy: light blue when fresh, deep blue when fading. (Mostly blue, with
        # little red, so that many glows stacked together stay blue instead of going white.)
        red = (10 + 50 * t) * brightness
        green = (70 + 110 * t) * brightness
        blue = 255 * brightness
    else:
        # Fire: yellow -> orange -> dark red as the flame ages
        red = 255 * brightness
        green = (60 + 150 * t) * brightness
        blue = 20 * t * brightness

    strength = np.clip(1 - distance / radius, 0, 1) ** 2   # bright in the middle, black at the edge
    pixels = np.zeros((radius * 2, radius * 2, 3))
    pixels[:, :, 0] = strength * red
    pixels[:, :, 1] = strength * green
    pixels[:, :, 2] = strength * blue
    return pygame.surfarray.make_surface(pixels.astype(np.uint8)).convert()


def get_glow(radius, t, palette):
    """Get a picture from the cache, making it if it doesn't exist yet."""
    radius = max(4, (radius // 4) * 4)  # round to steps of 4 so we make fewer pictures
    step = int(t * 12)                  # round the fade to 12 steps for the same reason
    key = (palette, radius, step)
    if key not in glow_cache:
        glow_cache[key] = make_glow(radius, step / 12, palette)
    return glow_cache[key]


def warm_up_glows():
    """Make every picture ahead of time (at startup), so the game never freezes
    for a moment later when an effect needs a size it hasn't made yet."""
    for palette, biggest in (("fire", 72), ("chakra", 72), ("smoke", 116)):
        for radius in range(4, biggest, 4):
            for step in range(13):
                glow_cache[(palette, radius, step)] = make_glow(radius, step / 12, palette)


class Particle:
    """One blob: it moves, changes size and fades until its life runs out."""

    def __init__(self, x, y, scale=1.0, palette="fire", vx=None, vy=None, size=None, life=None, drag=0.97):
        # scale makes the blob bigger and faster (1 = normal fire, 2 = big jutsu fire)
        # vx, vy, size and life are random unless you give them
        self.x = x
        self.y = y
        self.palette = palette
        self.additive = palette != "smoke"     # fire and chakra add light; smoke paints over
        self.grows = palette == "smoke"        # smoke spreads out; fire and chakra shrink
        self.vx = vx if vx is not None else random.uniform(-1.5, 1.5) * scale   # sideways speed (pixels per frame)
        self.vy = vy if vy is not None else random.uniform(-6, -2) * scale      # upward speed (negative = up)
        self.max_life = life if life is not None else random.randint(20, 40)    # lives for this many frames
        self.life = self.max_life
        self.size = size if size is not None else random.uniform(18, 34) * scale  # starting radius in pixels
        self.drag = drag                       # 1 = never slows down, smaller = slows faster

    def update(self):
        """Move one frame forward."""
        self.x += self.vx
        self.y += self.vy
        self.vx += random.uniform(-0.3, 0.3)   # small random wobble makes it flicker
        self.vx *= self.drag
        self.vy *= self.drag
        self.life -= 1

    def is_dead(self):
        return self.life <= 0

    def draw(self, screen):
        t = self.life / self.max_life          # 1 when new, 0 when dead
        if self.grows:
            radius = int(self.size * (0.5 + 0.9 * (1 - t)))   # smoke spreads as it ages
        else:
            radius = int(self.size * (0.4 + 0.6 * t))         # fire and chakra shrink as they age
        glow = get_glow(radius, t, self.palette)
        half = glow.get_width() // 2
        # BLEND_RGBA_ADD adds the glow's colours to what is already on screen, so
        # overlapping particles get brighter, just like real light. Smoke uses no
        # special flag: it is normal see-through painting.
        flags = pygame.BLEND_RGBA_ADD if self.additive else 0
        screen.blit(glow, (self.x - half, self.y - half), special_flags=flags)


# ---------------------------------------------------------------- particle makers

def spawn_burst(particles, x, y, count=18, palette="fire", scale=2.0):
    """Add a big cloud of fire (or chakra) around (x, y). Call it every frame for a
    while to build up a roaring fireball."""
    for _ in range(count):
        px = x + random.uniform(-70, 70)   # spread the start points around
        py = y + random.uniform(-40, 40)
        particles.append(Particle(px, py, scale=scale, palette=palette))


def spawn_aura(particles, box, strength):
    """Blue chakra rising around a seal. box is (x1, y1, x2, y2). strength goes from 0 to 1:
    the longer you hold the seal, the more and bigger the chakra."""
    x1, y1, x2, y2 = box
    for _ in range(1 + int(strength * 5)):
        # Start at a random spot along the bottom and sides of the box, then drift up
        edge = random.random()
        if edge < 0.5:
            px, py = random.uniform(x1, x2), random.uniform(y2 - 40, y2)   # bottom edge
        elif edge < 0.75:
            px, py = x1 + random.uniform(-10, 20), random.uniform(y1, y2)  # left edge
        else:
            px, py = x2 + random.uniform(-20, 10), random.uniform(y1, y2)  # right edge
        particles.append(Particle(
            px, py, palette="chakra",
            vx=random.uniform(-0.8, 0.8), vy=random.uniform(-3.5, -1.2),
            size=random.uniform(9, 16) * (0.8 + strength * 0.8),
            life=random.randint(14, 26)))


def spawn_ring(particles, x, y, count=40, speed=(6, 11)):
    """A ring of chakra that flies outward from (x, y), like a shock wave."""
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        fast = random.uniform(*speed)
        particles.append(Particle(
            x, y, palette="chakra",
            vx=math.cos(angle) * fast, vy=math.sin(angle) * fast,
            size=random.uniform(16, 26), life=random.randint(20, 32), drag=0.93))


def spawn_smoke(particles, x, y, count=6, scale=1.0):
    """Puffs of grey smoke that drift up and spread out."""
    for _ in range(count):
        particles.append(Particle(
            x + random.uniform(-40, 40) * scale, y + random.uniform(-25, 25) * scale, palette="smoke",
            vx=random.uniform(-1.2, 1.2) * scale, vy=random.uniform(-1.5, -0.3) * scale,
            size=random.uniform(22, 36) * scale, life=random.randint(35, 60), drag=0.98))


# ---------------------------------------------------------------- lightning

lightning_layer = None   # a black picture the bolts are drawn on, then added to the screen


def make_bolt(x1, y1, x2, y2, offset):
    """A jagged line from (x1, y1) to (x2, y2). We start with a straight line and
    keep bending the middle of every piece sideways by a random amount that gets
    smaller each round. That is what makes it look like a lightning bolt."""
    points = [(x1, y1), (x2, y2)]
    for _ in range(6):
        bent = [points[0]]
        for a, b in zip(points, points[1:]):
            middle = ((a[0] + b[0]) / 2 + random.uniform(-offset, offset),
                      (a[1] + b[1]) / 2 + random.uniform(-offset, offset) * 0.3)
            bent += [middle, b]
        points = bent
        offset /= 1.9
    return points


class Lightning:
    """One lightning strike from the top of the screen down to (x, y). It flickers:
    the bolt's shape is redrawn every few frames."""

    def __init__(self, x, y, frames=24):
        self.x = x
        self.y = y
        self.frames = frames
        self.life = frames
        self.build()

    def build(self):
        """Make a new random shape: one main bolt plus three side branches."""
        main = make_bolt(self.x + random.uniform(-80, 80), -10, self.x, self.y, 110)
        self.bolts = [main]
        for _ in range(3):
            px, py = random.choice(main[len(main) // 4: 3 * len(main) // 4])   # somewhere in the middle
            self.bolts.append(make_bolt(px, py, px + random.uniform(-260, 260), py + random.uniform(60, 220), 45))

    def update(self):
        self.life -= 1
        if self.life % 3 == 0:
            self.build()                      # new shape = flicker

    def is_dead(self):
        return self.life <= 0

    def draw(self, screen):
        global lightning_layer
        if lightning_layer is None or lightning_layer.get_size() != screen.get_size():
            lightning_layer = pygame.Surface(screen.get_size())
        lightning_layer.fill((0, 0, 0))

        # Flickering brightness, fading out over the second half of its life
        brightness = random.uniform(0.65, 1.0) * min(1.0, self.life / (self.frames * 0.5))
        # Three lines on top of each other, wide and dim to thin and white, give a glow
        for width, color in ((11, (20, 45, 150)), (6, (70, 120, 255)), (2, (235, 245, 255))):
            shade = tuple(int(c * brightness) for c in color)
            for bolt in self.bolts:
                pygame.draw.lines(lightning_layer, shade, False, bolt, width)
        screen.blit(lightning_layer, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


# ---------------------------------------------------------------- jutsu effects

# How many frames each jutsu effect keeps running (30 frames = 1 second)
EFFECT_FRAMES = {"fire": 45, "lightning": 36, "chakra": 40, "smoke": 30, "clone": 12}

# The colour of the screen flash and how hard the screen shakes (pixels) for each effect
EFFECT_FLASH = {"fire": (255, 190, 110), "lightning": (200, 220, 255), "chakra": (120, 180, 255),
                "smoke": (230, 230, 230), "clone": (235, 235, 245)}
EFFECT_SHAKE = {"fire": 6, "lightning": 11, "chakra": 8, "smoke": 3, "clone": 2}

# The colour of the jutsu name that appears on screen
EFFECT_TEXT = {"fire": (255, 140, 30), "lightning": (150, 195, 255), "chakra": (90, 175, 255),
               "smoke": (235, 235, 240), "clone": (190, 225, 255)}


def run_effect(kind, age, x, y, particles, bolts):
    """Call this every frame while a jutsu effect is running.
    age = how many frames since it started (0, 1, 2, ...). (x, y) = where the seal is."""
    if kind == "fire":
        spawn_burst(particles, x, y)
    elif kind == "lightning":
        if age % 9 == 0:                              # a new strike every 9 frames
            bolts.append(Lightning(x + random.uniform(-120, 120), y))
        spawn_burst(particles, x, y, count=4, palette="chakra", scale=1.4)   # blue glow where it hits
    elif kind == "chakra":
        if age in (0, 12):                            # two shock-wave rings
            spawn_ring(particles, x, y, count=50)
        spawn_burst(particles, x, y, count=5, palette="chakra", scale=1.5)
    elif kind == "smoke":
        spawn_smoke(particles, x, y, count=12, scale=2.4)
    elif kind == "clone":
        if age == 0:                                  # a shock-wave where the seal was made
            spawn_ring(particles, x, y, count=40)
        # (the puffs of smoke where the clones appear are made in main.py, because
        # only main.py knows where the clones are)
