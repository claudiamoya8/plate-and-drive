"""The game: a time trial of one lap, with a countdown, checkpoints and a timer.

What happens:
1. Countdown "3, 2, 1" with the car stopped on the start. If the wheel cannot be seen
   (Controls.visible is False), the countdown waits and asks to show it; with the keyboard it
   never waits. When the countdown ends, started() becomes True: that is the moment when
   main.py takes the neutral position of the wheels (WheelTracker.set_neutral).
2. "GO!": the timer starts and the car is driven with the Controls of every frame.
3. The lap only counts if the car goes through the four invisible checkpoints in order
   (C1, C2, C3, C4) and then crosses the finish line. The game reads the pixel of the zones
   mask of the track (see track.py) at every position the car went through in the frame, so
   a line is never missed.
4. When the lap is complete the timer stops, the car brakes and finished() becomes True;
   times() gives the final time of the plate.

The STOP sign, the traffic light and the leaderboard come in the next steps: for now the
traffic light stays green and there are no penalties.

On screen (draw(): 1280 x 720, without the FPS): the track and the car, the scoreboard at the
top ("Plate ... Time ... Penalties ...") and, in a dark panel drawn with rectangle on the grass
inside the loop, the countdown (or the message to show the wheel), "GO!" and the final time.
All texts are written with put_text of src/camera.py, at fixed positions. The two top corners
are left free for the FPS and the small camera view.

The lap time is game time: each frame adds its dt, limited to MAX_DT. At the normal frame
rate (about 30 FPS with the camera) this is the real time; if the program freezes for a
moment, the car does not jump and the time does not run away.
"""
import cv2
import numpy as np

from src.camera import put_text
from src.controls import Controls
from src.game.car import Car
from src.game.track import ZONE_CHECKPOINTS, ZONE_FINISH, Track

COUNTDOWN = 3.0   # s of countdown before the start
GO_TIME = 0.8     # s the "GO!" stays on screen
MAX_DT = 0.05     # s: a longer frame counts as this

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
PANEL_COLOR = (40, 40, 40)
PANEL_TOP_LEFT = (330, 360)       # the panel goes on the grass inside the loop,
PANEL_BOTTOM_RIGHT = (630, 510)   # clear of the road, the walls and the traffic light
SCOREBOARD = (285, 32)            # top line, between the FPS corner and the camera corner


def format_time(seconds):
    """Time as minutes:seconds.tenths, for example 0:12.4."""
    tenths = round(seconds * 10)   # rounded first, so 59.96 s is 1:00.0 and not 0:60.0
    return f"{tenths // 600}:{tenths % 600 / 10:04.1f}"


def draw_panel(image):
    """Dark panel with a white border, drawn with rectangle (Lab 5)."""
    cv2.rectangle(image, PANEL_TOP_LEFT, PANEL_BOTTOM_RIGHT, PANEL_COLOR, -1)
    cv2.rectangle(image, PANEL_TOP_LEFT, PANEL_BOTTOM_RIGHT, WHITE, 2)


class Game:
    """One race of one lap. Interface of the README (Module interfaces), plus started() for main.py."""

    def __init__(self, plates):
        """
        Args:
            plates (list[str]): the license plate of the player (for now, one player).
        """
        if len(plates) != 1:
            raise ValueError("For now the game is for one player: give one plate")
        self.track = Track()
        self.plate = plates[0]
        self.car = Car(self.track, self.plate)
        self.countdown = COUNTDOWN    # seconds left before the start
        self.waiting_wheel = False    # True while the countdown waits for the wheel to be seen
        self.race_time = 0.0          # seconds since the start (game time)
        self.penalties = 0.0          # seconds added for the STOP and the traffic light (next step)
        self.next_checkpoint = 0      # how many checkpoints, in order, the car has gone through
        self.done = False

    def started(self):
        """True once the countdown has ended (main.py takes the neutral position of the wheels then)."""
        return self.countdown <= 0

    def finished(self):
        """True when the lap is complete."""
        return self.done

    def times(self):
        """Final time of the plate (lap time + penalties), once the lap is complete: {plate: seconds}."""
        return {self.plate: self.race_time + self.penalties} if self.done else {}

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

        if self.done:
            self.car.update(Controls(0.0, -1.0, True), dt)   # after the finish the car brakes until it stops
            return

        self.race_time += dt
        path = self.car.update(control, dt)
        for k, (x, y) in enumerate(path):
            zone = self.track.zone_at(x, y)
            if self.next_checkpoint < len(ZONE_CHECKPOINTS):
                if zone == ZONE_CHECKPOINTS[self.next_checkpoint]:
                    self.next_checkpoint += 1
            elif zone == ZONE_FINISH:
                # Lap complete. The finish was reached at step k of this frame: count only that part of dt
                self.race_time -= dt * (len(path) - 1 - k) / len(path)
                self.done = True
                break

    def draw(self):
        """Image of this frame: 1280 x 720 BGR, without the FPS."""
        image = self.track.draw("green")
        self.car.draw(image)

        put_text(image, f"Plate {self.plate}   Time {format_time(self.race_time)}   "
                        f"Penalties +{self.penalties:.0f} s", SCOREBOARD, WHITE, 0.75)

        # The texts of the panel go at fixed positions, centred in it
        if not self.started():
            draw_panel(image)
            if self.waiting_wheel:
                put_text(image, "Show your wheel", (360, 427), WHITE, 0.9)
                put_text(image, "to the camera", (376, 463), WHITE, 0.9)
            else:
                put_text(image, str(int(np.ceil(self.countdown))), (454, 462), YELLOW, 2.5)
        elif self.done:
            draw_panel(image)
            put_text(image, "FINISHED", (369, 432), YELLOW, 1.6)
            put_text(image, f"Time {format_time(self.race_time + self.penalties)}", (385, 474), WHITE, 1.0)
        elif self.race_time < GO_TIME:
            draw_panel(image)
            put_text(image, "GO!", (413, 462), YELLOW, 2.5)
        return image
