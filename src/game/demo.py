"""Keyboard demo of the game, without the camera.

Step by step this will become the playable demo of the game. For now there is the track
and one car driven with the keyboard, and (key m) the masks of the track, to check them.

Usage, from the repository root:
    python -m src.game.demo
Keys: W/S/A/D = drive (accelerate, brake, left, right), r = back to the start,
m = masks on/off, t = change the colour of the traffic light, q / ESC (or closing the window) = quit.
"""
import time

import cv2
import numpy as np

from src.camera import FpsCounter, draw_fps, put_text
from src.controls import KeyboardControls
from src.game.car import Car
from src.game.track import (CAMERA_CORNER, FPS_CORNER, LIGHT_COLORS, ZONE_CHECKPOINTS, ZONE_FINISH,
                            ZONE_LIGHT_LINE, ZONE_STOP_AREA, ZONE_STOP_LINE, Track)

# Colour of each zone in the masks view (BGR)
ZONE_COLORS = {ZONE_FINISH: (255, 255, 255), ZONE_STOP_AREA: (0, 140, 255), ZONE_STOP_LINE: (0, 0, 255),
               ZONE_LIGHT_LINE: (0, 255, 255)}
ZONE_COLORS.update({zone: (255, 120, 0) for zone in ZONE_CHECKPOINTS})


def draw_masks(track):
    """Image to check the masks: the road in grey, every zone in its colour and the free corners outlined.

    The two images are put together with masks, like in Lab 1: bitwise_and keeps the pixels
    inside a mask and bitwise_or joins the two parts.
    """
    width, height = track.size
    grey = np.full((height, width, 3), 90, np.uint8)
    road = cv2.bitwise_and(grey, grey, mask=track.road_mask)
    palette = np.zeros((256, 3), np.uint8)   # zone value -> colour
    for zone, color in ZONE_COLORS.items():
        palette[zone] = color
    zone_colors = palette[track.zones]
    zone_mask = np.where(track.zones > 0, 255, 0).astype(np.uint8)   # binarization, as in Lab 3
    view = cv2.bitwise_or(cv2.bitwise_and(road, road, mask=255 - zone_mask),
                          cv2.bitwise_and(zone_colors, zone_colors, mask=zone_mask))

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
    track = Track()
    masks_view = draw_masks(track)   # the masks never change: drawn once
    car = Car(track, "2047")
    keyboard = KeyboardControls(players=1)
    fps_counter = FpsCounter()
    window_name = "Plate and Drive - demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)   # resizable, as in the course's test.py
    show_masks = False
    light = 0
    key = 255
    t_prev = time.perf_counter()
    while True:
        t = time.perf_counter()
        dt, t_prev = t - t_prev, t   # real dt between frames, for the physics
        controls = keyboard.update(key, dt)
        car.update(controls[0], dt)

        frame = masks_view.copy() if show_masks else track.draw(LIGHT_COLORS[light])
        car.draw(frame)
        put_text(frame, "W/S/A/D: drive   r: restart   q: quit", (285, 30), (255, 255, 255), 0.6)
        wall = "   WALL" if car.touching_wall else ""
        put_text(frame, f"m: masks   t: light   Speed: {car.speed:3.0f} px/s{wall}", (285, 58), (255, 255, 255), 0.6)
        draw_fps(frame, fps_counter.tick())
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("m"), ord("M")):
            show_masks = not show_masks
        if key in (ord("t"), ord("T")):
            light = (light + 1) % len(LIGHT_COLORS)
        if key in (ord("r"), ord("R")):
            car.reset()
        # Closing the window with its X also quits, as in Lab 1 (otherwise imshow would open it again)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
