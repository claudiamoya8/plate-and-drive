"""Steering wheel tracker: follows the two color marks of each wheel and turns them into Controls.

Each wheel is a white plate with two marks of different colors, one at the top and one at
the bottom. Steps, and the lab each one comes from:
1. HSV image and one mask per mark color with inRange (Lab 1). The color ranges are chosen
   with the trackbar tool of Lab 1 (python -m src.tracker.tune) and saved in data/wheel_colors.json.
2. Erosion and then dilation of each mask with a 3x3 kernel (Lab 3): small specks of the
   same color disappear and the marks keep their size.
3. Finding a wheel. While a wheel is not being followed (at the start, or after losing it),
   findContours finds every blob of each color, as in the card detector, and the wheel is the
   pair of blobs, one of each color, placed most like the marks on the plate: about the same
   size and about 1.8 diameters apart (9 cm and 5 cm). Because every blob is looked at, another
   object of the same color in the room does not stop the wheel from being found.
4. Following it. Each mark is then followed with meanShift and a Kalman filter, as in Lab 5
   (part D). The Kalman filter predicts where the mark will be, meanShift looks for it there in
   the mask, and the centre it finds corrects the filter. If a hand covers a mark for a moment,
   its predicted position is used for a few frames; after that the wheel is lost and it is
   searched for again as in step 3.
5. With the two centres: the angle of the line that joins them gives the steering, and their
   distance in pixels, which grows when the wheel comes closer to the camera, gives the throttle.
   Both are measured from the neutral position (wheel straight, at the distance where the car
   keeps its speed), which is taken with set_neutral().

The wheels are tracked on the mirrored image, the one the players see. Mirroring does not change
the colors, and this way left and right mean the same for the players and for the program.
With two players, each one has half of the image: player 1 on the left, player 2 on the right.

Demo, from the repository root (hold the wheel in front of the camera):
    python -m src.tracker.wheel
Keys: n = neutral (hold the wheel straight, at a comfortable distance, and press it),
      p = 1 or 2 players, m = masks on/off, q / ESC (or closing the window) = quit.
"""
import copy
import json
import os

import cv2
import numpy as np

from src.camera import FpsCounter, draw_fps, load_undistort_maps, open_camera, put_text, undistort
from src.controls import Controls

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS_FILE = os.path.normpath(os.path.join(HERE, "..", "..", "data", "wheel_colors.json"))
MARKS = ("top", "bottom")
# HSV ranges (OpenCV: H from 0 to 179, S and V from 0 to 255), used until tune.py saves the real ones:
# green at the top and magenta at the bottom, like the printed marks of docs/wheel_marks.pdf
DEFAULT_COLORS = {
    "top": {"lower": [35, 80, 80], "upper": [85, 255, 255]},
    "bottom": {"lower": [140, 80, 80], "upper": [175, 255, 255]},
}

KERNEL = np.ones((3, 3), np.uint8)   # 3x3 structuring element, as in Lab 3
MIN_PIXELS = 30      # fewer pixels of a color than this = no mark. The erosion already removes specks
                     # smaller than 5x5 = 25 pixels, and a 5 cm mark 4 m away still has about 90
MAX_MISSED = 10      # frames a covered mark is kept with the Kalman prediction (1/3 s at 30 FPS)
TERM_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)   # when meanShift stops, as in Lab 5
# Distance between the marks / their diameter: 9 cm / 5 cm = 1.8 on the plate. A pair of blobs
# can be a wheel between 1.2 and 3.0, which allows for a tilted wheel or a badly lit mark
MARK_RATIO = 1.8
SHAPE_RATIOS = (1.2, 3.0)

DEFAULT_SEPARATION = 75.0   # pixels between the marks with the wheel about 1 m away, until set_neutral()
MAX_ANGLE = 60.0     # degrees: turning the wheel this much (or more) = full steering
DEAD_ANGLE = 5.0     # degrees around the neutral angle that still count as straight
# Distance between the marks divided by the neutral one -> throttle. Far brakes (down to -1),
# the middle keeps the speed (0) and near accelerates (up to +1); np.interp joins the points with straight lines
THROTTLE_RATIOS = [0.80, 0.92, 1.08, 1.25]
THROTTLE_VALUES = [-1.0, 0.0, 0.0, 1.0]
BOX_SCALE = 1.25     # half side of the box around the wheel, in mark distances (plate radius 11 cm, marks 9 cm apart)
PLAYER_COLORS = [(255, 255, 0), (0, 165, 255)]   # BGR: cyan for player 1, orange for player 2


def load_colors(path=COLORS_FILE):
    """HSV ranges of the two marks, saved by tune.py. The default ones if the file does not exist."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found, using the default colors (tune them with python -m src.tracker.tune)")
        return copy.deepcopy(DEFAULT_COLORS)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def color_mask(hsv, color):
    """Mask of one mark color: inRange (Lab 1), then erosion and dilation with a 3x3 kernel (Lab 3).

    cv2.erode and cv2.dilate do the same as the erosion and dilation written by hand in Lab 3,
    whose pixel-by-pixel Python loops would be far too slow for live video.
    """
    mask = cv2.inRange(hsv, tuple(color["lower"]), tuple(color["upper"]))
    mask = cv2.erode(mask, KERNEL, iterations=2)    # removes specks of a few pixels...
    return cv2.dilate(mask, KERNEL, iterations=2)   # ...and gives the marks back their size


def search_window(cx, cy, side, shape):
    """Square window (x, y, w, h) for meanShift, centred on (cx, cy) and kept inside the image."""
    h, w = shape[:2]
    side = int(np.clip(side, 20, min(w, h)))
    x = int(np.clip(cx - side / 2, 0, w - side))
    y = int(np.clip(cy - side / 2, 0, h - side))
    return x, y, side, side


def find_blobs(mask):
    """Centre (x, y) and diameter of every blob of one color, found with findContours as in the card detector.

    The centre is the centre of the blob's bounding box, like the centre of the meanShift window.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        if cv2.contourArea(contour) >= MIN_PIXELS:
            x, y, w, h = cv2.boundingRect(contour)
            blobs.append((x + w / 2, y + h / 2, (w + h) / 2))
    return blobs


def find_wheel(top_mask, bottom_mask):
    """Finds a wheel: the pair of blobs, one of each color, placed most like the marks on the plate.

    Two blobs can be the marks of a wheel if they have a similar size and their distance divided
    by their diameter is within SHAPE_RATIOS. Of those pairs, the one closest to MARK_RATIO is taken.

    Returns:
        tuple: ((x, y) of the top mark, (x, y) of the bottom mark), or None if no pair fits.
    """
    best, best_error = None, None
    for tx, ty, top_diameter in find_blobs(top_mask):
        for bx, by, bottom_diameter in find_blobs(bottom_mask):
            if not 0.5 <= top_diameter / bottom_diameter <= 2:   # on the plate both marks are the same size
                continue
            ratio = np.hypot(tx - bx, ty - by) / ((top_diameter + bottom_diameter) / 2)
            error = abs(ratio - MARK_RATIO)
            if SHAPE_RATIOS[0] <= ratio <= SHAPE_RATIOS[1] and (best is None or error < best_error):
                best, best_error = ((tx, ty), (bx, by)), error
    return best


class MarkTracker:
    """Follows one color mark with meanShift and a Kalman filter, as in Lab 5 (part D)."""

    def __init__(self):
        # Constant velocity model with the matrices of Lab 5: state (x, y, vx, vy), measurement (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.missed = 0      # frames in a row in which the mark was not seen

    def start(self, x, y):
        """Starts following the mark at (x, y): the filter starts there with no speed, as in Lab 5."""
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.missed = 0

    def update(self, mask, side):
        """Follows the mark in this frame: Kalman prediction, then meanShift around it.

        Args:
            mask (np.ndarray): mask of the mark's color in this player's part of the image.
            side (int): side of the meanShift search window, in pixels.

        Returns:
            tuple: filtered (x, y) of the mark, or None once it has been lost.
        """
        prediction = self.kf.predict()   # where the mark should be now
        window = search_window(prediction[0, 0], prediction[1, 0], side, mask.shape)
        _, (x, y, w, h) = cv2.meanShift(mask, window, TERM_CRIT)
        if cv2.countNonZero(mask[y:y + h, x:x + w]) >= MIN_PIXELS:
            self.kf.correct(np.array([[x + w / 2], [y + h / 2]], np.float32))
            self.missed = 0
        else:
            self.missed += 1   # covered for a moment (a hand, a reflection): the prediction is kept
            if self.missed > MAX_MISSED:
                return None
        return float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])


class WheelTracker:
    """Tracks the wheel of each player and gives one Controls per player in every frame."""

    def __init__(self, players=1, colors_file=COLORS_FILE):
        """
        Args:
            players (int): 1 or 2. With 2, each player has half of the image.
            colors_file (str): HSV ranges of the marks, saved by tune.py.
        """
        if players not in (1, 2):
            raise ValueError("players must be 1 or 2")
        self.players = players
        self.colors = load_colors(colors_file)
        self.trackers = [{mark: MarkTracker() for mark in MARKS} for _ in range(players)]
        self.neutral = [(0.0, DEFAULT_SEPARATION) for _ in range(players)]   # (angle, separation)
        self.separation = [DEFAULT_SEPARATION] * players   # last distance between the marks
        self.wheels = [None] * players   # last (top, bottom, angle, separation) of each wheel, None if lost
        self.masks = None                # masks of the last frame, to show them

    def update(self, view):
        """Call it once per frame with the mirrored camera image.

        Returns:
            list[Controls]: one per player, player 1 first.
        """
        hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)   # Lab 1
        self.masks = {mark: color_mask(hsv, self.colors[mark]) for mark in MARKS}
        width = view.shape[1] // self.players
        controls = []
        for p in range(self.players):
            x0 = p * width
            masks = {mark: self.masks[mark][:, x0:x0 + width] for mark in MARKS}
            trackers = self.trackers[p]
            if self.wheels[p] is None:
                # Not followed: look for a pair of blobs placed like the marks on the plate
                pair = find_wheel(masks["top"], masks["bottom"])
                found = None if pair is None else dict(zip(MARKS, pair))
                if found is not None:
                    for mark in MARKS:
                        trackers[mark].start(*found[mark])
            else:
                # Followed: each mark with Kalman + meanShift. The search window is about as big
                # as the distance between the marks, so it is bigger than a mark
                found = {mark: trackers[mark].update(masks[mark], self.separation[p]) for mark in MARKS}
                if None in found.values():
                    found = None   # a mark has been lost: the wheel is searched for again in the next frame

            self.wheels[p] = None
            if found is not None:
                top = (found["top"][0] + x0, found["top"][1])
                bottom = (found["bottom"][0] + x0, found["bottom"][1])
                dx, dy = top[0] - bottom[0], top[1] - bottom[1]
                angle = float(np.degrees(np.arctan2(dx, -dy)))   # 0 = straight, > 0 = turned to the right
                separation = float(np.hypot(dx, dy))
                self.wheels[p] = (top, bottom, angle, separation)
                self.separation[p] = separation

            if self.wheels[p] is None:
                controls.append(Controls(0.0, -1.0, False))   # wheel lost: the game brakes the car
            else:
                controls.append(self.to_controls(p))
        return controls

    def to_controls(self, p):
        """Steering and throttle of player p, measured from their neutral position."""
        _, _, angle, separation = self.wheels[p]
        neutral_angle, neutral_separation = self.neutral[p]
        steering = np.interp(angle - neutral_angle, [-MAX_ANGLE, -DEAD_ANGLE, DEAD_ANGLE, MAX_ANGLE],
                             [-1.0, 0.0, 0.0, 1.0])
        throttle = np.interp(separation / neutral_separation, THROTTLE_RATIOS, THROTTLE_VALUES)
        return Controls(float(steering), float(throttle), True)

    def set_neutral(self):
        """Takes the current position of every wheel in sight as straight and 'keep the speed'."""
        for p, wheel in enumerate(self.wheels):
            if wheel is not None:
                self.neutral[p] = (wheel[2], wheel[3])

    def draw(self, view):
        """Draws the box around each wheel (Lab 5), the line between its marks and their centres."""
        h, w = view.shape[:2]
        width = w // self.players
        if self.players == 2:
            cv2.line(view, (width, 0), (width, h), (255, 255, 255), 2)   # border between the two players
        for p, wheel in enumerate(self.wheels):
            color = PLAYER_COLORS[p]
            if wheel is None:
                # Pixels of each mark color in this player's half: 0 means that color is not seen at all
                x0 = p * width
                top_px, bottom_px = (cv2.countNonZero(self.masks[mark][:, x0:x0 + width]) for mark in MARKS)
                put_text(view, f"P{p + 1}: wheel not found (top {top_px} px, bottom {bottom_px} px)",
                         (x0 + 20, h - 30), color, 0.7)
                continue
            top, bottom, _, separation = wheel
            cx, cy = (top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2
            r = BOX_SCALE * separation
            cv2.rectangle(view, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)), color, 3)
            cv2.line(view, (int(top[0]), int(top[1])), (int(bottom[0]), int(bottom[1])), color, 2)
            cv2.circle(view, (int(top[0]), int(top[1])), 6, color, -1)
            cv2.circle(view, (int(bottom[0]), int(bottom[1])), 6, color, -1)
            put_text(view, f"P{p + 1}", (int(cx - r), max(int(cy - r) - 10, 20)), color, 0.8)
        return view


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

    players = 1
    tracker = WheelTracker(players)
    fps_counter = FpsCounter()
    show_masks = False
    window_name = "Wheel tracker - n: neutral, p: players, m: masks, q: quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected)")
                break
            view = cv2.flip(undistort(frame, maps), 1)   # mirrored: tracked as the players see it
            controls = tracker.update(view)

            if show_masks:   # only the pixels of the two mark colors, joined as in Lab 1
                both = cv2.bitwise_or(tracker.masks["top"], tracker.masks["bottom"])
                view = cv2.bitwise_and(view, view, mask=both)
            tracker.draw(view)
            for p, c in enumerate(controls):
                wheel = tracker.wheels[p]
                if wheel is None:
                    continue
                x = 20 + p * (w // players)
                put_text(view, f"P{p + 1} steering {c.steering:+.2f}  throttle {c.throttle:+.2f}",
                         (x, 90), PLAYER_COLORS[p], 0.7)
                put_text(view, f"angle {wheel[2]:+.0f} deg  distance {wheel[3]:.0f} px "
                               f"(x{wheel[3] / tracker.neutral[p][1]:.2f})", (x, 125), PLAYER_COLORS[p], 0.7)
            draw_fps(view, fps_counter.tick())
            cv2.imshow(window_name, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("n"):
                tracker.set_neutral()
            if key == ord("p"):
                players = 3 - players   # 1 <-> 2
                tracker = WheelTracker(players)
            if key == ord("m"):
                show_masks = not show_masks
            # Closing the window with its X also quits, as in Lab 1 (otherwise imshow would open it again)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
