"""Seven-segment card detector: finds the cards in a camera frame and reads their digit.

Steps, and the lab each one comes from:
1. Grey image and Gaussian blur (Lab 3).
2. Binarization, pixel darker than a threshold (Lab 3). The threshold is chosen in every
   frame with Otsu's method, because the light changes in live video and a fixed 127
   would fail. The black ink of the card becomes white (255) in the binary image.
3. Quadrilaterals (the black frame) with findContours + approxPolyDP, helped by
   contourArea, arcLength and isContourConvex. These are not in the labs: they are
   the simple way to find a four-sided shape.
4. Homography from the card layout to the image with findHomography (Lab 4).
5. Points inside each zone of the layout are taken to the image with perspectiveTransform
   (Lab 4) and read in the binary image: first the frame and the orientation mark (to know
   it is a card and which side is up), then the seven segments.

The card layout (frame, mark and segment boxes) comes from src/security/cards.py.

Demo, from the repository root (show a card to the camera):
    python -m src.security.detector
Keys: b = binary image on/off, q / ESC = quit.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.camera import FpsCounter, draw_fps, load_undistort_maps, open_camera, put_text, undistort
from src.security import cards

MIN_AREA = 2000   # pixels: smaller quadrilaterals are ignored (a card 1.5 m away is still bigger)

# Corners of the black frame in card units, clockwise from the top-left one (y grows downwards)
fx0, fy0, fx1, fy1 = cards.FRAME_OUTER
ix0, iy0, ix1, iy1 = cards.FRAME_INNER
FRAME_CORNERS = np.float32([[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1]])
# The four sides of the black frame (top, bottom, left, right)
FRAME_BARS = [(fx0, fy0, fx1, iy0), (fx0, iy1, fx1, fy1), (fx0, iy0, ix0, iy1), (ix1, iy0, fx1, iy1)]
# Segment code -> digit, e.g. frozenset("bc") -> 1
CODE_TO_DIGIT = {frozenset(segments): digit for digit, segments in cards.DIGIT_SEGMENTS.items()}


@dataclass
class Card:
    digit: Optional[int]   # 0..9, or None if the card was found but could not be read with certainty
    corners: np.ndarray    # 4 x 2 frame corners in the image: top-left, top-right, bottom-right, bottom-left


def binarize(frame):
    """Grey + Gaussian blur + binarization (Lab 3), threshold chosen by Otsu. Black ink -> 255."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def find_quadrilaterals(binary):
    """Returns the four-sided shapes of the binary image, each as 4 corners in clockwise order."""
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = approx.reshape(4, 2).astype(np.float32)
            centre = points.mean(axis=0)
            angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
            quads.append(points[np.argsort(angles)])   # clockwise on screen
    return quads


def black_fraction(binary, H, box):
    """Fraction of black ink in a box of the card (card units), using H = card -> image.

    A 5 x 5 grid of points in the middle of the box is taken to the image with
    perspectiveTransform (Lab 4), and each point is read in the binary image.
    """
    x0, y0, x1, y1 = box
    xs = np.linspace(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0), 5)
    ys = np.linspace(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0), 5)
    grid = np.float32([[x, y] for y in ys for x in xs]).reshape(-1, 1, 2)
    points = cv2.perspectiveTransform(grid, H).reshape(-1, 2)
    cols = np.clip(np.round(points[:, 0]).astype(int), 0, binary.shape[1] - 1)
    rows = np.clip(np.round(points[:, 1]).astype(int), 0, binary.shape[0] - 1)
    return float(np.mean(binary[rows, cols] > 0))


def read_card(binary, corners):
    """Checks whether a quadrilateral is one of our cards and reads its digit.

    Each of the 4 corners is tried as the top-left one: the right one is the one where
    the frame is black and the orientation mark is in the top-left corner.

    Returns:
        Card, or None if the quadrilateral is not a card.
    """
    best = None
    for k in range(4):
        rolled = np.roll(corners, -k, axis=0)
        H, _ = cv2.findHomography(FRAME_CORNERS, rolled)   # card units -> image (Lab 4)
        if H is None:
            continue
        if min(black_fraction(binary, H, bar) for bar in FRAME_BARS) < 0.85:
            continue   # no black frame here
        mark = black_fraction(binary, H, cards.CORNER_BOXES["top_left"])
        other_corners = max(black_fraction(binary, H, cards.CORNER_BOXES[name])
                            for name in ("top_right", "bottom_right", "bottom_left"))
        if mark > 0.75 and other_corners < 0.25 and (best is None or mark > best[0]):
            best = (mark, H, rolled)
    if best is None:
        return None

    _, H, rolled = best
    fractions = {name: black_fraction(binary, H, box) for name, box in cards.SEGMENT_BOXES.items()}
    if any(0.25 < f < 0.75 for f in fractions.values()):
        digit = None   # a segment is half covered or blurred: better not to guess
    else:
        lit = frozenset(name for name, f in fractions.items() if f >= 0.75)
        digit = CODE_TO_DIGIT.get(lit)   # None if the segments do not form a digit
    return Card(digit, rolled)


def detect_cards(frame):
    """Finds every card in a BGR frame. Returns a list of Card (empty if there is none)."""
    binary = binarize(frame)
    found = []
    for quad in find_quadrilaterals(binary):
        card = read_card(binary, quad)
        if card is not None:
            found.append(card)
    return found


def draw_cards(frame, found):
    """Returns a mirrored copy of the frame with every card outlined and its digit written.

    Green: card read. Orange: card found but not readable ("?"). The dot marks the top-left corner.
    """
    w = frame.shape[1]
    view = frame.copy()
    colors = [(0, 255, 0) if card.digit is not None else (0, 165, 255) for card in found]
    for card, color in zip(found, colors):
        cv2.polylines(view, [card.corners.astype(np.int32)], True, color, 3)   # as in Lab 4
        cv2.circle(view, tuple(int(v) for v in card.corners[0]), 8, color, -1)  # top-left corner
    view = cv2.flip(view, 1)   # mirror only for display; texts are written after flipping
    for card, color in zip(found, colors):
        x = int(w - 1 - card.corners[:, 0].max())   # left edge of the card in the mirrored view
        y = int(card.corners[:, 1].min()) - 15
        put_text(view, str(card.digit) if card.digit is not None else "?", (x, max(y, 30)), color, 1.5)
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

    fps_counter = FpsCounter()
    show_binary = False
    window_name = "Card detector - b: binary on/off, q: quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected)")
                break
            frame = undistort(frame, maps)
            found = detect_cards(frame)

            shown = cv2.cvtColor(binarize(frame), cv2.COLOR_GRAY2BGR) if show_binary else frame
            view = draw_cards(shown, found)
            draw_fps(view, fps_counter.tick())
            cv2.imshow(window_name, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("b"):
                show_binary = not show_binary
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
