"""Color tuning for the wheel marks: the trackbar tool of Lab 1 (get_hsv_color_ranges), but live.

The six trackbars of Lab 1 (HMin, SMin, VMin, HMax, SMax, VMax) set the HSV range of one mark
at a time. The window shows the camera on the left and, on the right, only the pixels the wheel
tracker will use for that mark: the same mask as the tracker (inRange of Lab 1, then erosion and
dilation of Lab 3), applied with bitwise_and as in Lab 1. A range is good when the whole mark
shows up and nothing else in the room does.

How to tune a mark: hold the wheel in front of the camera, raise SMin and VMin until the
background disappears, then lower them until the mark is complete, and close HMin and HMax
around its color. Do it with the light of the place where you are going to play.

From the repository root:
    python -m src.tracker.tune
Keys: 1 = tune the top mark, 2 = tune the bottom mark, s = save both in data/wheel_colors.json,
      q / ESC (or closing the window) = quit.
"""
import json

import cv2
import numpy as np

from src.camera import FpsCounter, draw_fps, load_undistort_maps, open_camera, put_text, undistort
from src.tracker.wheel import COLORS_FILE, MARKS, color_mask, load_colors

# The trackbars of Lab 1. OpenCV keeps the hue between 0 and 179, so H goes up to 179
TRACKBARS = [("HMin", 179), ("SMin", 255), ("VMin", 255), ("HMax", 179), ("SMax", 255), ("VMax", 255)]
PREVIEW_SIZE = (640, 360)   # each half of the window (camera | mask), so it fits on screen with the trackbars


def nothing(x):
    pass   # the trackbars are read in every frame, as in Lab 1


def set_trackbars(window_name, color):
    """Moves the trackbars to the range of one mark."""
    for (name, _), value in zip(TRACKBARS, color["lower"] + color["upper"]):
        cv2.setTrackbarPos(name, window_name, value)


def read_trackbars(window_name):
    """The range set by the trackbars, as {"lower": [h, s, v], "upper": [h, s, v]}."""
    values = [cv2.getTrackbarPos(name, window_name) for name, _ in TRACKBARS]
    return {"lower": values[:3], "upper": values[3:]}


def save_colors(colors, path=COLORS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(colors, f, indent=2)


def main():
    cap = open_camera()
    if cap is None:
        return
    ret, frame = cap.read()
    if not ret:
        print("Error: the camera opened but returned no frames")
        cap.release()
        return
    h, w = frame.shape[:2]
    maps = load_undistort_maps((w, h))

    colors = load_colors()
    editing = "top"
    window_name = "Wheel colors - 1: top mark, 2: bottom mark, s: save, q: quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)   # resizable, as in test.py
    for name, maximum in TRACKBARS:
        cv2.createTrackbar(name, window_name, 0, maximum, nothing)
    set_trackbars(window_name, colors[editing])
    fps_counter = FpsCounter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected)")
                break
            view = cv2.flip(undistort(frame, maps), 1)   # mirrored, as the tracker sees it
            colors[editing] = read_trackbars(window_name)
            mask = color_mask(cv2.cvtColor(view, cv2.COLOR_BGR2HSV), colors[editing])
            segmented = cv2.bitwise_and(view, view, mask=mask)   # only the pixels inside the range, as in Lab 1

            output = np.hstack([cv2.resize(view, PREVIEW_SIZE), cv2.resize(segmented, PREVIEW_SIZE)])
            put_text(output, f"Tuning the {editing} mark ({cv2.countNonZero(mask)} pixels)",
                     (20, PREVIEW_SIZE[1] - 20), (255, 255, 255), 0.7)
            draw_fps(output, fps_counter.tick())
            cv2.imshow(window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key in (ord("1"), ord("2")):
                editing = MARKS[key - ord("1")]
                set_trackbars(window_name, colors[editing])
            if key == ord("s"):
                save_colors(colors)
                print(f"Saved in {COLORS_FILE}:")
                for mark in MARKS:
                    print(f"  {mark}: lower = {colors[mark]['lower']}, upper = {colors[mark]['upper']}")
            # Closing the window with its X also quits, as in Lab 1 (otherwise imshow would open it again)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
