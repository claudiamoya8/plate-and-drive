"""Keyboard demo of the game, without the camera.

The game as it will be played, but with the keyboard instead of the steering wheels: the
countdown and then as many laps as you want, with the STOP and the traffic light, each valid
lap saved in data/leaderboard.json. Key p changes between one and two players. Key e ends the
turn and shows the leaderboard, as main.py will do. Key m shows the masks of the track, to
check them.

Usage, from the repository root:
    python -m src.game.demo
Keys: W/S/A/D = drive (accelerate, brake, left, right), and I/K/J/L for player 2;
p = one or two players, e = end the turn, r = start again, m = masks on/off,
q / ESC (or closing the window) = quit.
With the keyboard, Windows only repeats the last key held down, so two players cannot hold
their keys at the same time (a known limitation of src/controls.py: the two-player mode is
really played with the wheels).
"""
import time

import cv2
import numpy as np

from src.camera import FpsCounter, draw_fps, put_text
from src.controls import KeyboardControls
from src.game.game import Game, draw_leaderboard, leaderboard
from src.game.track import (CAMERA_CORNER, FPS_CORNER, ZONE_CHECKPOINTS, ZONE_FINISH, ZONE_LIGHT_LINE,
                            ZONE_STOP_AREA, ZONE_STOP_LINE)

PLATES = ("2047", "1123")   # player 1 (the registered plate of data/plates.json) and player 2

# Colour of each zone in the masks view (BGR)
ZONE_COLORS = {ZONE_FINISH: (255, 255, 255), ZONE_STOP_AREA: (0, 140, 255), ZONE_STOP_LINE: (0, 0, 255),
               ZONE_LIGHT_LINE: (0, 255, 255)}
ZONE_COLORS.update({zone: (255, 120, 0) for zone in ZONE_CHECKPOINTS})


def draw_masks(track):
    """Image to check the masks: the road in grey, every zone in its colour and the free corners outlined.

    The road is kept with its mask (bitwise_and, as in Lab 1), and the pixels of each zone are
    painted in its colour, as the Harris corners are painted in red in Lab 4.
    """
    width, height = track.size
    grey = np.full((height, width, 3), 90, np.uint8)
    view = cv2.bitwise_and(grey, grey, mask=track.road_mask)
    for zone, color in ZONE_COLORS.items():
        view[track.zones == zone] = color

    for k, s in enumerate(track.layout["checkpoints"], 1):
        p, u = track.point_at(s)
        put_text(view, f"C{k}", (int(p[0] - 70 * u[1]) - 12, int(p[1] + 70 * u[0]) + 8), (255, 120, 0), 0.6)
    for (x0, y0, x1, y1) in (FPS_CORNER, CAMERA_CORNER):
        cv2.rectangle(view, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
    legend = [("finish line", ZONE_FINISH), ("checkpoints C1-C4", ZONE_CHECKPOINTS[0]),
              ("STOP area", ZONE_STOP_AREA), ("STOP line", ZONE_STOP_LINE), ("traffic light line", ZONE_LIGHT_LINE)]
    for k, (name, zone) in enumerate(legend):
        put_text(view, name, (470, 430 + 28 * k), ZONE_COLORS[zone], 0.6)
    return view


def main():
    players = 1
    game = Game(list(PLATES[:players]))
    masks_view = draw_masks(game.track)   # the masks never change: drawn once
    keyboard = KeyboardControls(players)
    fps_counter = FpsCounter()
    window_name = "Plate and Drive - demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)   # resizable, as in the course's test.py
    show_masks = False
    turn_over = False   # True after key e: the game stops and the leaderboard is shown
    rows = []
    key = 255
    t_prev = time.perf_counter()
    while True:
        t = time.perf_counter()
        dt, t_prev = t - t_prev, t   # real dt between frames
        if not turn_over:
            game.update(keyboard.update(key, dt), dt)

        if turn_over:
            frame = game.track.draw("green")   # the track and the cars, without the scoreboard and the panels
            for player in game.players:
                player.car.draw(frame)
            draw_leaderboard(frame, rows)
        elif show_masks:
            frame = masks_view.copy()
            for player in game.players:
                player.car.draw(frame)
        else:
            frame = game.draw()
        put_text(frame, "W/S/A/D (player 2: I/K/J/L): drive   p: 1 or 2 players   e: end the turn   r: start again   "
                        "m: masks   q: quit", (20, 710), (255, 255, 255), 0.55)
        draw_fps(frame, fps_counter.tick())   # the FPS are written here, not inside Game.draw()
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("m"), ord("M")):
            show_masks = not show_masks
        if key in (ord("e"), ord("E")) and not turn_over:
            turn_over = True
            rows = leaderboard()   # the file is read once, when the turn ends
        if key in (ord("p"), ord("P")):
            players = 3 - players   # 1 -> 2 and 2 -> 1
        if key in (ord("r"), ord("R"), ord("p"), ord("P")):   # start again (with the new number of players)
            game = Game(list(PLATES[:players]))
            keyboard = KeyboardControls(players)
            turn_over = False
        # Closing the window with its X also quits, as in Lab 1 (otherwise imshow would open it again)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
