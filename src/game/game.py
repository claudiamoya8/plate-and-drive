"""The game: a time trial with free laps, for one or two players, with a countdown, checkpoints, penalties and the leaderboard.

What happens:
1. Countdown "3, 2, 1" with the cars stopped on the start. If a wheel cannot be seen
   (Controls.visible is False), the countdown waits and asks to show it; with the keyboard it
   never waits. When the countdown ends, started() becomes True: that is the moment when
   main.py takes the neutral position of the wheels (WheelTracker.set_neutral).
2. "GO!": the timer of the first lap starts and each car is driven with its Controls of every
   frame (controls[0] for player 1, the one on the left half of the camera image).
3. A lap is valid if the car goes through the four invisible checkpoints in order (C1, C2,
   C3, C4) and then crosses the finish line. The game reads the pixel of the zones mask of the
   track (see track.py) at every position the car went through in the frame, so a line is
   never missed. Crossing the finish line just after the start does not count, because the
   car has not gone through the checkpoints yet.
4. Every valid lap is saved at once in data/leaderboard.json, so no time is lost if the
   program is closed: {"plate", "time", "penalties"}, where time is the final time of the lap
   with its penalties included and penalties is how many seconds of it they are. Then the
   timer, the checkpoints and the penalties of that car start again from 0, and its next lap
   goes on without stopping.
5. The players drive as many laps as they want. main.py ends the turn with a key and shows
   the leaderboard (leaderboard() and draw_leaderboard()): the best lap of each plate.

With two players, the two cars share the track and the screen but do not crash into each
other: they go through each other, and each one does its own time trial (its own laps,
penalties and timer). They share the traffic light.

Penalties, added to the lap in which they happen:
- STOP: the car has to stop completely (car.stopped) inside the STOP area, before the line.
  Crossing the line without having stopped there adds STOP_PENALTY.
- Traffic light: green GREEN_TIME, yellow YELLOW_TIME and red RED_TIME, over and over, from the
  "GO!" (it is green during the countdown). Crossing its line while it is red adds
  RED_LIGHT_PENALTY.
Each line is checked the first time a car touches it in a lap, so a penalty is never counted
twice even though the car takes several frames to cross the line.

On screen (draw(): 1280 x 720, without the FPS): the track with the traffic light and the cars,
each with its plate above it in the colour of the car (draw_cars: first all the cars and then
all the plates, so a car never hides the plate of the other); the scoreboard at the top (plate,
lap, time of this lap, penalties, last lap and best lap: in two lines with one player, and in
one line per player, after a square of the colour of its car, with two); and, in a dark panel
drawn with rectangle on the grass inside the loop, the countdown (or the message to show the
wheels), "GO!" and, for 2 s, the time of each lap or each penalty. Each player has their own
notice: with one player the panel shows it big, with the plate of the car; with two, it shows
one line for each player with a notice, in the colour of their car, so two notices at the same
time are both seen. All texts are written with put_text of src/camera.py, at fixed positions.
The two top corners are left free for the FPS and the small camera view.

The lap time is game time: each frame adds its dt, limited to MAX_DT. At the normal frame
rate (about 30 FPS with the camera) this is the real time; if the program freezes for a
moment, the cars do not jump and the time does not run away. The traffic light runs on the
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
from src.game.car import CAR_COLOR, Car
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

PLAYER_COLORS = (CAR_COLOR, (220, 90, 0))     # BGR: red car for player 1 and blue car for player 2
START_SIDES = {1: (0.0,), 2: (-40.0, 40.0)}   # px across the road where each car starts (player 1 on the left)

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


class Player:
    """What each player has of their own: the car, the laps and the penalties."""

    def __init__(self, track, plate, color, side):
        """
        Args:
            track (Track): the track.
            plate (str): license plate of the player's car.
            color (tuple): BGR colour of the car.
            side (float): where the car starts across the road (see Track.start_pose).
        """
        self.plate = plate
        self.color = color
        self.car = Car(track, plate, color, side)
        self.lap = 1                  # number of the lap being driven
        self.lap_time = 0.0           # seconds since this lap started (game time)
        self.penalties = 0.0          # seconds of penalties in this lap
        self.next_checkpoint = 0      # how many checkpoints of this lap, in order, the car has gone through
        self.stopped_at_stop = False  # True once the car has stopped inside the STOP area in this lap
        self.stop_passed = False      # True once the car has touched the STOP line in this lap
        self.light_passed = False     # True once the car has touched the traffic light line in this lap
        self.last_lap = None          # final time of the last lap (None until the first one is complete)
        self.best_lap = None          # final time of the best lap of this turn
        self.notice = ""              # what the panel shows about this player for a moment: "LAP TIME", "STOP" or "RED LIGHT"
        self.notice_time = 0.0        # seconds left to show it


class Game:
    """The turn of one or two players, with free laps. Interface of the README (Module interfaces)."""

    def __init__(self, plates, leaderboard_file=LEADERBOARD_FILE):
        """
        Args:
            plates (list[str]): the license plates of the players: player 1 (left half of the camera) first.
            leaderboard_file (str): where the laps are saved (data/leaderboard.json).
        """
        if len(plates) not in START_SIDES:
            raise ValueError("The game is for one or two players: give one or two plates")
        self.track = Track()
        sides = START_SIDES[len(plates)]
        self.players = [Player(self.track, plate, PLAYER_COLORS[k], sides[k]) for k, plate in enumerate(plates)]
        self.leaderboard_file = leaderboard_file
        self.countdown = COUNTDOWN    # seconds left before the start
        self.waiting_wheel = False    # True while the countdown waits for the wheels to be seen
        self.race_time = 0.0          # seconds since the "GO!" (game time): the traffic light runs on it

    @property
    def car(self):
        """The car of player 1 (with one player, the only one)."""
        return self.players[0].car

    def started(self):
        """True once the countdown has ended (main.py takes the neutral position of the wheels then)."""
        return self.countdown <= 0

    def best_laps(self):
        """Best lap of each plate in this turn, with its penalties included: {plate: seconds} (only plates with a complete lap).

        If the two players come with the same plate, it keeps the better of their two laps.
        """
        best = {}
        for player in self.players:
            if player.best_lap is not None and (player.plate not in best or player.best_lap < best[player.plate]):
                best[player.plate] = player.best_lap
        return best

    def light(self):
        """Colour of the traffic light now: "green", "yellow" or "red"."""
        t = self.race_time % (GREEN_TIME + YELLOW_TIME + RED_TIME)
        if t < GREEN_TIME:
            return "green"
        if t < GREEN_TIME + YELLOW_TIME:
            return "yellow"
        return "red"

    def update(self, controls, dt):
        """One frame of the game.

        Args:
            controls (list[Controls]): one per player, player 1 first.
            dt (float): real time since the previous frame, in seconds.
        """
        dt = min(max(dt, 0.0), MAX_DT)

        if not self.started():
            # The cars wait on the start; the countdown only runs while every wheel can be seen
            self.waiting_wheel = not all(control.visible for control in controls[:len(self.players)])
            if not self.waiting_wheel:
                self.countdown -= dt
            return

        self.race_time += dt
        for player in self.players:
            player.notice_time = max(player.notice_time - dt, 0.0)
        for player, control in zip(self.players, controls):
            self._drive(player, control, dt)

    def _drive(self, player, control, dt):
        """Moves the car of one player and checks the lines it touched (the cars never touch each other)."""
        player.lap_time += dt
        path = player.car.update(control, dt)
        for k, (x, y) in enumerate(path):
            zone = self.track.zone_at(x, y)
            # STOP line and traffic light line: checked the first time the car touches them in this lap
            if zone == ZONE_STOP_LINE and not player.stop_passed:
                player.stop_passed = True
                if not player.stopped_at_stop:
                    self._penalty(player, STOP_PENALTY, "STOP")
            elif zone == ZONE_LIGHT_LINE and not player.light_passed:
                player.light_passed = True
                if self.light() == "red":
                    self._penalty(player, RED_LIGHT_PENALTY, "RED LIGHT")
            # Checkpoints in order, and then the finish line
            if player.next_checkpoint < len(ZONE_CHECKPOINTS):
                if zone == ZONE_CHECKPOINTS[player.next_checkpoint]:
                    player.next_checkpoint += 1
            elif zone == ZONE_FINISH:
                # Lap complete at step k of this frame: the rest of the frame already belongs to the next lap
                rest = dt * (len(path) - 1 - k) / len(path)
                self._complete_lap(player, player.lap_time - rest)
                player.lap_time = rest
                break
        # A full stop only counts inside the STOP area, before its line
        if player.car.stopped and self.track.zone_at(player.car.x, player.car.y) == ZONE_STOP_AREA:
            player.stopped_at_stop = True

    def _penalty(self, player, seconds, notice):
        """Adds a penalty to the lap of a player and shows it in the panel for a moment."""
        player.penalties += seconds
        player.notice, player.notice_time = notice, NOTICE_TIME

    def _complete_lap(self, player, driving_time):
        """Saves the lap a player has just completed and starts their next one."""
        lap = round(driving_time + player.penalties, 2)   # final time of the lap, with its penalties
        laps = load_laps(self.leaderboard_file)
        laps.append({"plate": player.plate, "time": lap, "penalties": player.penalties})
        save_laps(laps, self.leaderboard_file)   # saved at once: nothing is lost if the program is closed
        player.last_lap = lap
        if player.best_lap is None or lap < player.best_lap:
            player.best_lap = lap
        player.lap += 1
        player.penalties = 0.0
        player.next_checkpoint = 0
        player.stopped_at_stop = False
        player.stop_passed = False
        player.light_passed = False
        player.notice, player.notice_time = "LAP TIME", NOTICE_TIME

    def draw_cars(self, image):
        """Draws all the cars and then all their plates, so a car never hides the plate of the other.

        draw() uses it, and so can the end-of-turn screen of main.py.
        """
        for player in self.players:
            player.car.draw(image)
        for player in self.players:
            player.car.draw_plate(image)

    def draw(self):
        """Image of this frame: 1280 x 720 BGR, without the FPS."""
        image = self.track.draw(self.light())
        self.draw_cars(image)

        # Scoreboard: two lines with one player, and one line per player with two
        if len(self.players) == 1:
            player = self.players[0]
            last = format_time(player.last_lap) if player.last_lap is not None else "-"
            best = format_time(player.best_lap) if player.best_lap is not None else "-"
            put_text(image, f"Plate {player.plate}   Lap {player.lap}   Time {format_time(player.lap_time)}   "
                            f"Penalties +{player.penalties:.0f} s", SCOREBOARD[0], WHITE, 0.75)
            put_text(image, f"Last lap {last}   Best lap {best}", SCOREBOARD[1], WHITE, 0.75)
        else:
            for (x, y), player in zip(SCOREBOARD, self.players):
                last = format_time(player.last_lap) if player.last_lap is not None else "-"
                best = format_time(player.best_lap) if player.best_lap is not None else "-"
                cv2.rectangle(image, (x, y - 13), (x + 13, y), player.color, -1)   # the colour of the car (Lab 5)
                cv2.rectangle(image, (x, y - 13), (x + 13, y), WHITE, 1)
                put_text(image, f"{player.plate}  Lap {player.lap}  Time {format_time(player.lap_time)}  "
                                f"+{player.penalties:.0f} s  Last {last}  Best {best}", (x + 22, y), WHITE, 0.6)

        # The texts of the panel go at fixed positions, centred in it
        noticed = [player for player in self.players if player.notice_time > 0]   # players with a notice now
        if not self.started():
            draw_panel(image)
            if self.waiting_wheel:
                put_text(image, "Show your wheel", (360, 427), WHITE, 0.9)
                put_text(image, "to the camera", (376, 463), WHITE, 0.9)
            else:
                put_text(image, str(int(np.ceil(self.countdown))), (454, 462), YELLOW, 2.5)
        elif self.race_time < GO_TIME:
            draw_panel(image)
            put_text(image, "GO!", (413, 462), YELLOW, 2.5)
        elif noticed and len(self.players) == 1:
            # One player: the notice in big letters, with the plate of the car under it
            draw_panel(image)
            player = noticed[0]
            if player.notice == "LAP TIME":
                put_text(image, "LAP TIME", (368, 431), YELLOW, 1.6)
                put_text(image, f"{player.plate}   {format_time(player.last_lap)}", (354, 475), WHITE, 1.1)
            elif player.notice == "STOP":
                put_text(image, "STOP", (416, 431), RED, 1.6)
                put_text(image, f"{player.plate}   +{STOP_PENALTY:.0f} s", (365, 475), WHITE, 1.1)
            else:
                put_text(image, "RED LIGHT", (350, 431), RED, 1.6)
                put_text(image, f"{player.plate}   +{RED_LIGHT_PENALTY:.0f} s", (365, 475), WHITE, 1.1)
        elif noticed:
            # Two players: one line for each player with a notice, in the colour of their car
            draw_panel(image)
            rows = (443,) if len(noticed) == 1 else (423, 463)   # one line in the middle, or one under the other
            for y, player in zip(rows, noticed):
                if player.notice == "LAP TIME":
                    text, x = f"{player.plate}  LAP TIME {format_time(player.last_lap)}", 351
                elif player.notice == "STOP":
                    text, x = f"{player.plate}  STOP +{STOP_PENALTY:.0f} s", 379
                else:
                    text, x = f"{player.plate}  RED LIGHT +{RED_LIGHT_PENALTY:.0f} s", 350
                put_text(image, text, (x, y), player.color, 0.7)
        return image
