"""Seven-segment cards for the security system: their layout and a printable PDF.

Each card shows one digit (0-9) the way a digital clock does: seven black bars
(segments) on a white background. The digit is surrounded by a thick black frame,
so the card is easy to find in the image, with a white margin outside the frame.
A small black square in the top-left corner inside the frame tells which side is up.

The segments are named like on any seven-segment display:

     a
   f   b
     g
   e   c
     d

The layout is defined here only once, in "card units" (the card is 40 x 60 units),
and it is shared by this script, which draws and prints the cards, and by the
detector, which uses the same boxes to know where to look for each segment.

Usage, from the repository root:
    python -m src.security.cards        # writes docs/cards.pdf (10 cards on 3 A4 pages)

Print the PDF at 100 % on matte paper and glue the cards on cardboard so they stay flat.
The exact printed size does not matter, the detector only uses the proportions.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.normpath(os.path.join(HERE, "..", "..", "docs"))

# ---- Card layout, in card units. Every box is (x0, y0, x1, y1) ----
CARD_SIZE = (40, 60)                # (width, height): portrait card, white margin of 4 units
FRAME_OUTER = (4, 4, 36, 56)        # outer border of the black frame: the detector finds this
FRAME_INNER = (7, 7, 33, 53)        # inner border of the black frame (3 units thick)

# 4 x 4 squares in the four corners inside the frame. Only the top-left one is printed
# (the orientation mark); the other three are always white, whatever the digit.
CORNER_BOXES = {
    "top_left": (9, 9, 13, 13),
    "top_right": (27, 9, 31, 13),
    "bottom_right": (27, 47, 31, 51),
    "bottom_left": (9, 47, 13, 51),
}

# The digit fills the box (12, 16, 28, 44): 16 x 28 units, segments 4 units thick,
# with a small gap of 0.5 units where two segments meet.
SEGMENT_BOXES = {
    "a": (16.5, 16, 23.5, 20),
    "b": (24, 20.5, 28, 27.5),
    "c": (24, 32.5, 28, 39.5),
    "d": (16.5, 40, 23.5, 44),
    "e": (12, 32.5, 16, 39.5),
    "f": (12, 20.5, 16, 27.5),
    "g": (16.5, 28, 23.5, 32),
}

# Segments that are on for each digit
DIGIT_SEGMENTS = {
    0: "abcdef",
    1: "bc",
    2: "abdeg",
    3: "abcdg",
    4: "bcfg",
    5: "acdfg",
    6: "acdefg",
    7: "abc",
    8: "abcdefg",
    9: "abcdfg",
}

# ---- Printing ----
DPI = 300
A4_PIXELS = (2480, 3508)   # (width, height) of an A4 page at 300 dpi
PRINT_SCALE = 26           # pixels per card unit: each card is 1040 x 1560 px (88 x 132 mm)
GAP = 80                   # pixels between two cards on the page (about 7 mm)


def fill_box(image, box, value, scale):
    """Paints a box given in card units with cv2.rectangle, the drawing function of Lab 5 (thickness=-1 fills it).

    Args:
        image (np.array): grayscale image to paint on.
        box (tuple): (x0, y0, x1, y1) in card units.
        value (int): grey level, 0 = black, 255 = white.
        scale (int): pixels per card unit.
    """
    x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), value, thickness=-1)


def draw_card(digit, scale=PRINT_SCALE):
    """Draws the card of one digit.

    Args:
        digit (int): digit from 0 to 9.
        scale (int): pixels per card unit. Use an even number so every box lands on whole pixels.

    Returns:
        np.array: grayscale image of the card (255 = white, 0 = black).
    """
    width, height = CARD_SIZE
    card = np.full((height * scale, width * scale), 255, np.uint8)
    fill_box(card, FRAME_OUTER, 0, scale)                   # black frame...
    fill_box(card, FRAME_INNER, 255, scale)                 # ...white inside it
    fill_box(card, CORNER_BOXES["top_left"], 0, scale)      # orientation mark
    for segment in DIGIT_SEGMENTS[digit]:
        fill_box(card, SEGMENT_BOXES[segment], 0, scale)
    # Light grey line on the edge of the card, to know where to cut
    cv2.rectangle(card, (0, 0), (width * scale - 1, height * scale - 1), 200, 2)
    return card


def make_pages():
    """Places the 10 cards on A4 pages, 4 per page (2 columns x 2 rows).

    Returns:
        list: one grayscale image per page (A4 at 300 dpi).
    """
    cards = [draw_card(digit) for digit in range(10)]
    card_h, card_w = cards[0].shape
    x_start = (A4_PIXELS[0] - (2 * card_w + GAP)) // 2
    y_start = (A4_PIXELS[1] - (2 * card_h + GAP)) // 2
    pages = []
    for first in range(0, len(cards), 4):
        page = np.full((A4_PIXELS[1], A4_PIXELS[0]), 255, np.uint8)
        for i, card in enumerate(cards[first:first + 4]):
            row, col = divmod(i, 2)
            x = x_start + col * (card_w + GAP)
            y = y_start + row * (card_h + GAP)
            page[y:y + card_h, x:x + card_w] = card
        pages.append(page)
    return pages


def save_pdf(pages, path):
    """Saves the pages in one PDF, each page at its real size (A4 at 300 dpi).

    Each page is shown with plt.imshow, as in the labs, and PdfPages puts the pages in one file.
    matplotlib is imported here, and not at the top, so that the detector, which only needs
    the layout above, does not load it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for page in pages:
            plt.figure(figsize=(page.shape[1] / DPI, page.shape[0] / DPI))
            plt.axes([0, 0, 1, 1])  # the image fills the whole page
            plt.imshow(page, cmap="gray", vmin=0, vmax=255, interpolation="none")
            plt.axis("off")
            pdf.savefig()
            plt.close()


def main():
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, "cards.pdf")
    pages = make_pages()
    save_pdf(pages, path)
    print(f"Saved {len(pages)} pages with the 10 cards in {path}")
    print("Print it at 100 % (no 'fit to page') and glue the cards on cardboard.")


if __name__ == "__main__":
    main()
