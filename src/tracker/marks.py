"""Printable marks for the steering wheels: solid green and magenta circles, 5 cm wide.

Printed colors are much more saturated than highlighter or marker on paper. That keeps the
marks far from the colors of the room, and they do not fade when the wheel comes closer or
goes away and the camera changes its exposure.

Usage, from the repository root:
    python -m src.tracker.marks        # writes docs/wheel_marks.pdf (one A4 page)

Print it in color, at 100 % (no "fit to page"), on plain matte paper: photo paper shines.
Cut the circles out and glue them flat on the plate, without tape over them: green at the
top and magenta at the bottom, with their centres 9 cm apart. The page has three of each:
two for the two wheels and one spare.
"""
import os

import cv2
import numpy as np

from src.security.cards import A4_PIXELS, DOCS, DPI, save_pdf

MARK_DIAMETER_CM = 5.0
COLORS = [(0, 255, 0), (255, 0, 255)]   # BGR: green (top mark) and magenta (bottom mark)
ROWS = 3                                # rows of one green and one magenta circle


def make_page():
    """One A4 page at 300 dpi with ROWS rows of a green and a magenta circle, drawn with cv2.circle (Lab 5)."""
    width, height = A4_PIXELS
    page = np.full((height, width, 3), 255, np.uint8)
    radius = int(round(MARK_DIAMETER_CM / 2.54 * DPI / 2))   # 5 cm at 300 dpi: 591 px wide
    for row in range(ROWS):
        y = int((row + 0.5) * height / ROWS)
        for col, color in enumerate(COLORS):
            x = int((col + 0.5) * width / len(COLORS))
            cv2.circle(page, (x, y), radius, color, -1, cv2.LINE_AA)
    return page


def main():
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, "wheel_marks.pdf")
    page = cv2.cvtColor(make_page(), cv2.COLOR_BGR2RGB)   # plt.imshow expects RGB
    save_pdf([page], path)
    print(f"Saved the wheel marks in {path}")
    print("Print it in color at 100 % (no 'fit to page') on plain matte paper.")


if __name__ == "__main__":
    main()
