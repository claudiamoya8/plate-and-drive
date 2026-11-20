"""Controls given to the game on every frame, and a keyboard version of them.

Controls is the interface between whatever drives the car and the game. The wheel
tracker will produce it from the camera later on; until then, KeyboardControls
produces it from the keyboard, so the game can be developed and tested without a camera.

Keys (one OpenCV window and one cv2.waitKey per frame, like the live loops of Labs 1 and 5):
    player 1: W accelerate, S brake, A left, D right
    player 2: I accelerate, K brake, J left, L right

How the keyboard is turned into steering and throttle:
- Every key press adds a small step in its direction (STEP). The opposite key
  (S while accelerating, D while turning left...) first drops the value to 0, so
  braking or turning the other way reacts at once.
- cv2.waitKey only reports presses, not whether a key is still down. When a key is
  held, Windows sends one press, waits about 0.5 s and then repeats it about 30 times
  per second. So a key counts as released when no press of it has arrived for
  HOLD_TIME (0.6 s); with less time, the control would drop during that first wait.
- Once released, steering and throttle go back to 0 little by little (RETURN_SPEED).
  A throttle of 0 means "keep the current speed", like the middle zone of the wheel.
- Known limitation (accepted): with two keys held at the same time, Windows only
  repeats the last one, so two players cannot hold keys at once. The keyboard is
  only a test mode; the real two-player mode is tested with the wheels.

Test window, from the repository root:
    python -m src.controls
Click on the window first so it receives the keys. q / ESC (or closing the window) = quit.
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.camera import FpsCounter, draw_fps, put_text


@dataclass
class Controls:
    """What the tracker (or the keyboard, at first) gives the game on every frame, per player."""
    steering: float  # -1 (full left) .. +1 (full right)
    throttle: float  # -1 (full brake) .. 0 (keep speed) .. +1 (full throttle)
    visible: bool    # False if the wheel is lost -> the game brakes the car


HOLD_TIME = 0.6     # seconds without presses before a key counts as released
STEP = 0.2          # how much each key press adds to steering or throttle
RETURN_SPEED = 3.0  # how fast steering and throttle go back to 0 once released (units per second)

# key -> (player, axis, direction)
KEYS = {
    "w": (0, "throttle", +1), "s": (0, "throttle", -1), "a": (0, "steering", -1), "d": (0, "steering", +1),
    "i": (1, "throttle", +1), "k": (1, "throttle", -1), "j": (1, "steering", -1), "l": (1, "steering", +1),
}
AXES = ("steering", "throttle")


class KeyboardControls:
    """Turns the keys read with cv2.waitKey into one Controls per player."""

    def __init__(self, players=1):
        """
        Args:
            players (int): 1 or 2. With 1 player, the keys of player 2 are ignored.
        """
        if players not in (1, 2):
            raise ValueError("players must be 1 or 2")
        self.players = players
        self.values = [{axis: 0.0 for axis in AXES} for _ in range(players)]
        self.last_press = [{axis: None for axis in AXES} for _ in range(players)]
        self.clock = 0.0  # seconds since the start, built from the dt of every frame

    def update(self, key, dt):
        """Call it once per frame with the key read in that frame.

        Args:
            key (int): cv2.waitKey(1) & 0xFF. 255 (or -1) means that no key was pressed.
            dt (float): real time since the previous frame, in seconds.

        Returns:
            list[Controls]: one per player. The keyboard never loses the wheel, so visible is True.
        """
        self.clock += dt
        if 0 <= key < 255:
            action = KEYS.get(chr(key).lower())  # lower(): also works with Caps Lock on
            if action is not None and action[0] < self.players:
                self._press(*action)

        for player in range(self.players):
            for axis in AXES:
                last = self.last_press[player][axis]
                if last is None or self.clock - last > HOLD_TIME:  # released: back to 0 little by little
                    self.values[player][axis] = _towards_zero(self.values[player][axis], RETURN_SPEED * dt)

        return [Controls(v["steering"], v["throttle"], True) for v in self.values]

    def _press(self, player, axis, direction):
        value = self.values[player][axis]
        if value * direction < 0:
            # The opposite key: start from 0, like lifting the other pedal, so braking
            # (or turning the other way) reacts at once instead of after several presses
            value = 0.0
        self.values[player][axis] = float(np.clip(value + STEP * direction, -1.0, 1.0))
        self.last_press[player][axis] = self.clock


def _towards_zero(value, amount):
    """Moves value towards 0 by at most amount, without going past 0."""
    if value > 0:
        return max(0.0, value - amount)
    return min(0.0, value + amount)


# ---------------------------------------------------------------------------
# Test window: shows the steering and throttle of both players as bars
# ---------------------------------------------------------------------------

WINDOW_SIZE = (900, 420)  # width, height
BAR_SIZE = (300, 28)


def draw_bar(image, x, y, value, label):
    """Bar from -1 to +1 with its centre marked: green to the right of the centre, red to the left."""
    w, h = BAR_SIZE
    cx = x + w // 2
    end = cx + int(round(value * w / 2))
    color = (0, 200, 0) if value >= 0 else (0, 0, 220)
    if end != cx:
        cv2.rectangle(image, (min(cx, end), y), (max(cx, end), y + h), color, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.line(image, (cx, y - 6), (cx, y + h + 6), (255, 255, 255), 2)
    put_text(image, f"{label}: {value:+.2f}", (x, y - 14), (255, 255, 255), 0.7)


def draw_controls(controls):
    """Image with one column of bars per player."""
    width, height = WINDOW_SIZE
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (50, 50, 50)
    keys = ["W/S/A/D", "I/K/J/L"]
    for player, c in enumerate(controls):
        x = 60 + player * (width // 2)
        put_text(image, f"Player {player + 1} ({keys[player]})", (x, 110), (0, 255, 255), 0.8)
        draw_bar(image, x, 180, c.steering, "steering")
        draw_bar(image, x, 280, c.throttle, "throttle")
    put_text(image, "Click on this window, then press the keys. q / ESC or close the window: quit",
             (60, height - 30), (200, 200, 200), 0.6)
    return image


def main():
    keyboard = KeyboardControls(players=2)
    fps_counter = FpsCounter()
    window_name = "Keyboard controls test"
    cv2.namedWindow(window_name)  # like the live loop of Lab 1 (utils.py)
    t_prev = time.perf_counter()
    key = 255
    while True:
        t = time.perf_counter()
        dt, t_prev = t - t_prev, t  # real dt between frames
        controls = keyboard.update(key, dt)

        frame = draw_controls(controls)
        draw_fps(frame, fps_counter.tick())
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        # Closing the window with its X also quits, as in Lab 1 (otherwise imshow would open it again)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
