"""Stage 6: turn "which seal is shown right now" into confirmed seals and jutsu.

main.py calls update() once per frame. This file knows nothing about the
camera or the screen, it only watches the seal names go by.
"""

from collections import deque

# The jutsu and the seals needed, in order. Change these however you like!
# A seal used twice in a row (like "palms" twice) needs a short "let go"
# in between, otherwise holding it would count as only one.
JUTSUS = {
    "Fireball": ["palms", "fists", "points"],
    "Thunder": ["peace", "palms", "palms"],
}

HOLD_TIME = 0.5          # seconds a seal must be held steadily to count
RELEASE_TIME = 0.15      # seconds of "no seal" that count as letting go
SEQUENCE_TIMEOUT = 3.0   # forget the seals so far if you pause this long


class SealSequence:
    """Remembers the recently confirmed seals and spots when they spell a jutsu."""

    def __init__(self):
        longest = max(len(seals) for seals in JUTSUS.values())
        self.history = deque(maxlen=longest)  # confirmed seals, oldest first

        self.candidate = None        # the seal being shown right now (None = no seal)
        self.candidate_since = 0.0   # when it started being shown
        self.counted = False         # did this hold already count as a confirmed seal?
        self.released = True         # have we let go since the last confirmed seal?
        self.last_confirmed = None   # the most recent confirmed seal
        self.last_confirm_time = 0.0

    def hold_progress(self, now):
        """How far along the 0.5 s hold is, from 0 to 1 (for a progress bar)."""
        if self.candidate is None:
            return 0.0
        if self.counted:
            return 1.0
        return min(1.0, (now - self.candidate_since) / HOLD_TIME)

    def update(self, seal, now):
        """Call once per frame with the current seal name (or None) and the time in seconds.
        Returns (confirmed seal or None, jutsu name or None)."""
        # Paused too long? Start over.
        if self.history and now - self.last_confirm_time > SEQUENCE_TIMEOUT:
            self.history.clear()
            self.last_confirmed = None

        # Did the shown seal change? Then the hold timer starts again.
        if seal != self.candidate:
            self.candidate = seal
            self.candidate_since = now
            self.counted = False

        # No seal: if it lasts a moment, that counts as "letting go"
        if seal is None:
            if now - self.candidate_since >= RELEASE_TIME:
                self.released = True
            return None, None

        # A seal is shown. Has it been held long enough, and not counted yet?
        if now - self.candidate_since < HOLD_TIME or self.counted:
            return None, None
        self.counted = True  # whatever happens below, this hold is now dealt with

        # The same seal as last time only counts again if hands were let go in between
        if seal == self.last_confirmed and not self.released:
            return None, None

        # Confirmed!
        self.history.append(seal)
        self.last_confirmed = seal
        self.last_confirm_time = now
        self.released = False

        jutsu = self.find_match()
        if jutsu:
            self.history.clear()      # start fresh for the next jutsu
            self.last_confirmed = None
        return seal, jutsu

    def find_match(self):
        """Do the newest confirmed seals spell one of the jutsu?"""
        recent = list(self.history)
        for name, seals in JUTSUS.items():
            if recent[-len(seals):] == seals:
                return name
        return None
