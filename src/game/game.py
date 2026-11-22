"""The game: a time trial with free laps, a countdown, checkpoints, penalties and the leaderboard.

What happens:
1. Countdown "3, 2, 1" with the car stopped on the start. If the wheel cannot be seen
   (Controls.visible is False), the countdown waits and asks to show it; with the keyboard it
   never waits. When the countdown ends, started() becomes True: that is the moment when
   main.py takes the neutral position of the wheels (WheelTracker.set_neutral).
2. "GO!": the timer of the first lap starts and the car is driven with the Controls of every frame.
3. A lap is valid if the car goes through the four invisible checkpoints in order (C1, C2,
   C3, C4) and then crosses the finish line. The game reads the pixel of the zones mask of the
   track (see track.py) at every position the car went through in the frame, so a line is
   never missed. Crossing the finish line just after the start does not count, because the
   car has not gone through the checkpoints yet.
4. Every valid lap is saved at once in data/leaderboard.json, so no time is lost if the
   program is closed: {"plate", "time", "penalties"}, where time is the final time of the lap
   with its penalties included and penalties is how many seconds of it they are. Then the
   timer, the checkpoints and the penalties start again from 0, and the next lap goes on
   without stopping.
5. The player drives as many laps as they want. main.py ends the turn with a key and shows
   the leaderboard (leaderboard() and draw_leaderboard()): the best lap of each plate.

Penalties, added to the lap in which they happen:
- STOP: the car has to stop completely (car.stopped) inside the STOP area, before the line.
  Crossing the line without having stopped there adds STOP_PENALTY.
- Traffic light: green GREEN_TIME, yellow YELLOW_TIME and red RED_TIME, over and over, from the
  "GO!" (it is green during the countdown). Crossing its line while it is red adds
  RED_LIGHT_PENALTY.
Each line is checked the first time the car touches it in a lap, so a penalty is never counted
twice even though the car takes several frames to cross the line.

On screen (draw(): 1280 x 720, without the FPS): the track with the traffic light and the car,
the scoreboard at the top (plate, lap, time of this lap, penalties, last lap and best lap) and,
in a dark panel drawn with rectangle on the grass inside the loop, the countdown (or the
message to show the wheel), "GO!" and, for 2 s, the time of each lap or each penalty. All texts
are written with put_text of src/camera.py, at fixed positions. The two top corners are left
free for the FPS and the small camera view.

The lap time is game time: each frame adds its dt, limited to MAX_DT. At the normal frame
rate (about 30 FPS with the camera) this is the real time; if the program freezes for a
moment, the car does not jump and the time does not run away. The traffic light runs on the
same game time.

data/leaderboard.json is not uploaded to GitHub: every computer keeps its own laps. If it does
not exist, the leaderboard is empty and the file is created with the first lap, as it is done
with the plates in src/security/decoder.py.
"""
import json
import os

import cv2
import numpy as np

from src.camera import put_text
from src.game.car import Car
from src.game.track import ZONE_CHECKPOINTS, ZONE_FINISH, ZONE_LIGHT_LINE, ZONE_STOP_AREA, ZONE_STOP_LINE, Track

HERE = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD_FILE = os.path.normpath(os.path.join(HERE, "..", "..", "data", "leaderboard.json"))

COUNTDOWN = 3.0           # s of countdown before the start
GO_TIME = 0.8             # s the "GO!" stays on screen
NOTICE_TIME = 2.0         # s the panel shows the time of a lap or a penalty
MAX_DT = 0.05             # s: a longer frame counts as this
STOP_PENALTY = 5.0        # s for crossing the STOP line without having stopped
RED_LIGHT_PENALTY = 5.0   # s for crossing the traffic light line while it is red
GREEN_TIME, YELLOW_TIME, RED_TIME = 4.0, 1.0, 3.0   # s of each colour of the traffic light
LEADERBOARD_ROWS = 5      # laps shown in the leaderboard

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
PANEL_COLOR = (40, 40, 40)
PANEL_TOP_LEFT = (330, 360)              # the panel goes on the grass inside the loop,
PANEL_BOTTOM_RIGHT = (630, 510)          # clear of the road, the walls and the traffic light
LEADERBOARD_TOP_LEFT = (440, 184)        # the leaderboard panel, in the middle of the image
LEADERBOARD_BOTTOM_RIGHT = (840, 536)
SCOREBOARD = ((285, 32), (285, 62))      # its two lines, between the FPS corner and the camera corner


def format_time(seconds):
    """Time as minutes:seconds.tenths, for example 0:12.4."""
    tenths = round(seconds * 10)   # rounded first, so 59.96 s is 1:00.0 and not 0:60.0
    return f"{tenths // 600}:{tenths % 600 / 10:04.1f}"


def load_laps(path=LEADERBOARD_FILE):
    """Returns every saved lap, {"plate", "time", "penalties"}, in the order they were driven (empty if the file does not exist)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_laps(laps, path=LEADERBOARD_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(laps, f, indent=2)


def leaderboard(n=LEADERBOARD_ROWS, path=LEADERBOARD_FILE):
    """The n best laps ever, one per plate (its best lap), the fastest first.

    Returns:
        list: (plate, seconds) of each of those laps, with its penalties included.
    """
    best = {}
    for lap in load_laps(path):
        if lap["plate"] not in best or lap["time"] < best[lap["plate"]]:
            best[lap["plate"]] = lap["time"]
    return sorted(best.items(), key=lambda item: item[1])[:n]


def draw_panel(image, top_left=PANEL_TOP_LEFT, bottom_right=PANEL_BOTTOM_RIGHT):
    """Dark panel with a white border, drawn with rectangle (Lab 5)."""
    cv2.rectangle(image, top_left, bottom_right, PANEL_COLOR, -1)
    cv2.rectangle(image, top_left, bottom_right, WHITE, 2)


def draw_leaderboard(image, rows):
    """Draws the leaderboard in a dark panel in the middle of the image, for the end of a turn.

    Args:
        rows (list): (plate, seconds) of the best laps, as leaderboard() gives them. 5 fit.
    """
    draw_panel(image, LEADERBOARD_TOP_LEFT, LEADERBOARD_BOTTOM_RIGHT)
    # Fixed positions, centred in the panel: every row has the same width (4 digits and m:ss.s)
    put_text(image, "LEADERBOARD", (487, 254), YELLOW, 1.4)
    if not rows:
        put_text(image, "No laps yet", (548, 316), WHITE, 1.0)
    for k, (plate, seconds) in enumerate(rows[:LEADERBOARD_ROWS]):
        put_text(image, f"{k + 1}.   {plate}   {format_time(seconds)}", (486, 316 + 45 * k), WHITE, 1.0)


class Game:
    """The turn of one player, with free laps. Interface of the README (Module interfaces)."""

    def __init__(self, plates, leaderboard_file=LEADERBOARD_FILE):
        """
        Args:
            plates (list[str]): the license plate of the player (for now, one player).
            leaderboard_file (str): where the laps are saved (data/leaderboard.json).
        """
        if len(plates) != 1:
            raise ValueError("For now the game is for one player: give one plate")
        self.track = Track()
        self.plate = plates[0]
        self.car = Car(self.track, self.plate)
        self.leaderboard_file = leaderboard_file
        self.countdown = COUNTDOWN    # seconds left before the start
        self.waiting_wheel = False    # True while the countdown waits for the wheel to be seen
        self.light_time = 0.0         # seconds of the traffic light cycle, which starts at the "GO!"
        self.lap = 1                  # number of the lap being driven
        self.lap_time = 0.0           # seconds since this lap started (game time)
        self.penalties = 0.0          # seconds of penalties in this lap
        self.next_checkpoint = 0      # how many checkpoints of this lap, in order, the car has gone through
        self.stopped_at_stop = False  # True once the car has stopped inside the STOP area in this lap
        self.stop_passed = False      # True once the car has touched the STOP line in this lap
        self.light_passed = False     # True once the car has touched the traffic light line in this lap
        self.last_lap = None          # final time of the last lap (None until the first one is complete)
        self.best_lap = None          # final time of the best lap of this turn
        self.notice = ""              # what the panel shows for a moment: "LAP TIME", "STOP" or "RED LIGHT"
        self.notice_time = 0.0        # seconds left to show it

    def started(self):
        """True once the countdown has ended (main.py takes the neutral position of the wheels then)."""
        return self.countdown <= 0

    def best_laps(self):
        """Best lap of this turn, with its penalties included: {plate: seconds} (empty until a lap is complete)."""
        return {self.plate: self.best_lap} if self.best_lap is not None else {}

    def light(self):
        """Colour of the traffic light now: "green", "yellow" or "red"."""
        t = self.light_time % (GREEN_TIME + YELLOW_TIME + RED_TIME)
        if t < GREEN_TIME:
            return "green"
        if t < GREEN_TIME + YELLOW_TIME:
            return "yellow"
        return "red"

    def update(self, controls, dt):
        """One frame of the game.

        Args:
            controls (list[Controls]): one per player (for now, one).
            dt (float): real time since the previous frame, in seconds.
        """
        dt = min(max(dt, 0.0), MAX_DT)
        control = controls[0]

        if not self.started():
            # The car waits on the start; the countdown only runs while the wheel can be seen
            self.waiting_wheel = not control.visible
            if not self.waiting_wheel:
                self.countdown -= dt
            return

        self.lap_time += dt
        self.light_time += dt
        self.notice_time = max(self.notice_time - dt, 0.0)
        path = self.car.update(control, dt)
        for k, (x, y) in enumerate(path):
            zone = self.track.zone_at(x, y)
            # STOP line and traffic light line: checked the first time the car touches them in this lap
            if zone == ZONE_STOP_LINE and not self.stop_passed:
                self.stop_passed = True
                if not self.stopped_at_stop:
                    self._penalty(STOP_PENALTY, "STOP")
            elif zone == ZONE_LIGHT_LINE and not self.light_passed:
                self.light_passed = True
                if self.light() == "red":
                    self._penalty(RED_LIGHT_PENALTY, "RED LIGHT")
            # Checkpoints in order, and then the finish line
            if self.next_checkpoint < len(ZONE_CHECKPOINTS):
                if zone == ZONE_CHECKPOINTS[self.next_checkpoint]:
                    self.next_checkpoint += 1
            elif zone == ZONE_FINISH:
                # Lap complete at step k of this frame: the rest of the frame already belongs to the next lap
                rest = dt * (len(path) - 1 - k) / len(path)
                self._complete_lap(self.lap_time - rest)
                self.lap_time = rest
                break
        # A full stop only counts inside the STOP area, before its line
        if self.car.stopped and self.track.zone_at(self.car.x, self.car.y) == ZONE_STOP_AREA:
            self.stopped_at_stop = True

    def _penalty(self, seconds, notice):
        """Adds a penalty to this lap and shows it in the panel for a moment."""
        self.penalties += seconds
        self.notice, self.notice_time = notice, NOTICE_TIME

    def _complete_lap(self, driving_time):
        """Saves the lap that has just been completed and starts the next one."""
        lap = round(driving_time + self.penalties, 2)   # final time of the lap, with its penalties
        laps = load_laps(self.leaderboard_file)
        laps.append({"plate": self.plate, "time": lap, "penalties": self.penalties})
        save_laps(laps, self.leaderboard_file)   # saved at once: nothing is lost if the program is closed
        self.last_lap = lap
        if self.best_lap is None or lap < self.best_lap:
            self.best_lap = lap
        self.lap += 1
        self.penalties = 0.0
        self.next_checkpoint = 0
        self.stopped_at_stop = False
        self.stop_passed = False
        self.light_passed = False
        self.notice, self.notice_time = "LAP TIME", NOTICE_TIME

    def draw(self):
        """Image of this frame: 1280 x 720 BGR, without the FPS."""
        image = self.track.draw(self.light())
        self.car.draw(image)

        last = format_time(self.last_lap) if self.last_lap is not None else "-"
        best = format_time(self.best_lap) if self.best_lap is not None else "-"
        put_text(image, f"Plate {self.plate}   Lap {self.lap}   Time {format_time(self.lap_time)}   "
                        f"Penalties +{self.penalties:.0f} s", SCOREBOARD[0], WHITE, 0.75)
        put_text(image, f"Last lap {last}   Best lap {best}", SCOREBOARD[1], WHITE, 0.75)

        # The texts of the panel go at fixed positions, centred in it
        if not self.started():
            draw_panel(image)
            if self.waiting_wheel:
                put_text(image, "Show your wheel", (360, 427), WHITE, 0.9)
                put_text(image, "to the camera", (376, 463), WHITE, 0.9)
            else:
                put_text(image, str(int(np.ceil(self.countdown))), (454, 462), YELLOW, 2.5)
        elif self.lap == 1 and self.lap_time < GO_TIME:
            draw_panel(image)
            put_text(image, "GO!", (413, 462), YELLOW, 2.5)
        elif self.notice_time > 0:
            draw_panel(image)
            if self.notice == "LAP TIME":
                put_text(image, "LAP TIME", (368, 425), YELLOW, 1.6)
                put_text(image, format_time(self.last_lap), (399, 480), WHITE, 1.6)
            elif self.notice == "STOP":
                put_text(image, "STOP", (416, 425), RED, 1.6)
                put_text(image, f"+{STOP_PENALTY:.0f} s", (416, 480), WHITE, 1.6)
            else:
                put_text(image, "RED LIGHT", (350, 425), RED, 1.6)
                put_text(image, f"+{RED_LIGHT_PENALTY:.0f} s", (416, 480), WHITE, 1.6)
        return image
