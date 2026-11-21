"""Sequence decoder of the security system: reads a 4-digit license plate, card by card.

It is a small state machine fed with the cards the detector finds in every frame:
- A digit is accepted when the same card is read for HOLD_TIME seconds in a row,
  so a single wrong reading never counts.
- Before the next digit the card has to be taken away (no card for REMOVE_TIME seconds).
  That is what makes repeated digits work: 1123 is entered with one card of each number.
- When 4 digits have been entered, the plate is looked up in data/plates.json: if it is
  there, access is granted; if not, the sequence is cleared and the user starts again.
- An unfinished sequence is forgotten after FORGET_TIME seconds without any card,
  and it can be reset by hand at any moment.
- Extension: in "new car" mode the 4 digits are saved as a new plate instead.

Demo, from the repository root (show the cards one by one to the camera):
    python -m src.security.decoder
Keys: n = register a new car, r = reset, q / ESC = quit.
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2

from src.camera import FpsCounter, draw_fps, load_undistort_maps, open_camera, put_text, undistort
from src.security.detector import detect_cards, draw_cards

HERE = os.path.dirname(os.path.abspath(__file__))
PLATES_FILE = os.path.normpath(os.path.join(HERE, "..", "..", "data", "plates.json"))

PLATE_LENGTH = 4
HOLD_TIME = 0.5     # seconds a digit has to be read the same to be accepted
REMOVE_TIME = 0.3   # seconds without any card to count as "card taken away" (a missed frame is not enough)
FORGET_TIME = 7.0   # seconds without any card to forget an unfinished sequence


@dataclass
class SecurityState:
    digits: str              # digits entered so far, e.g. "20"
    granted: Optional[str]   # the plate once access has been granted, None until then
    message: str             # what to tell the user on screen


def load_plates(path=PLATES_FILE):
    """Returns the registered plates as a list of strings (empty if the file does not exist)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_plates(plates, path=PLATES_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(plates), f, indent=2)


class SequenceDecoder:
    def __init__(self, plates_file=PLATES_FILE):
        self.plates_file = plates_file
        self.plates = load_plates(plates_file)
        self.registering = False
        self.reset("Show the first digit of your plate")

    def reset(self, message="Sequence reset: show the first digit"):
        """Forgets the digits entered so far."""
        self.digits = ""
        self.granted = None
        self.candidate = None          # digit being read, waiting for HOLD_TIME
        self.candidate_since = 0.0
        self.waiting_removal = False   # a digit was just accepted: wait until the card is taken away
        self.last_card_time = None     # when a card was last seen
        self.message = message

    def start_registration(self):
        """Extension: the next 4 digits are saved as a new plate."""
        self.registering = True
        self.reset("New car: show the 4 digits of the new plate")

    def update(self, found, now=None):
        """Updates the sequence with the cards found in one frame.

        Args:
            found (list): cards from detector.detect_cards().
            now (float): current time in seconds (time.perf_counter() if not given).

        Returns:
            SecurityState
        """
        now = time.perf_counter() if now is None else now
        if self.granted is not None:
            return self.state()        # access already granted: nothing else to read

        if not found:
            if self.last_card_time is not None:
                away = now - self.last_card_time
                if away >= REMOVE_TIME:
                    self.waiting_removal = False
                    self.candidate = None
                if self.digits and away >= FORGET_TIME:
                    self.reset("Too long without a card: sequence forgotten")
            return self.state()

        self.last_card_time = now
        if len(found) > 1:
            self.candidate = None
            self.message = "Show only one card at a time"
        elif self.waiting_removal:
            if self.digits:   # in the middle of a plate (after the 4th digit, keep the result message)
                self.message = "Take the card away and show the next digit"
        elif found[0].digit is None:
            self.candidate = None
            self.message = "Card not readable: hold it still and avoid reflections"
        elif found[0].digit != self.candidate:
            self.candidate, self.candidate_since = found[0].digit, now
            self.message = f"Reading {self.candidate}... hold it still"
        elif now - self.candidate_since >= HOLD_TIME:
            self.accept(self.candidate)
        return self.state()

    def accept(self, digit):
        """Adds an accepted digit and, with 4 digits, checks or registers the plate."""
        self.digits += str(digit)
        self.candidate = None
        self.waiting_removal = True
        if len(self.digits) < PLATE_LENGTH:
            self.message = f"Digit {digit} accepted. Take the card away"
            return

        plate, self.digits = self.digits, ""
        if self.registering:
            self.registering = False
            if plate in self.plates:
                self.message = f"Plate {plate} was already registered"
            else:
                self.plates.append(plate)
                save_plates(self.plates, self.plates_file)
                self.message = f"New car registered: {plate}"
        elif plate in self.plates:
            self.digits = plate
            self.granted = plate
            self.message = f"Access granted: {plate}"
        else:
            self.message = f"Plate {plate} is not registered. Try again"

    def state(self):
        return SecurityState(self.digits, self.granted, self.message)


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

    decoder = SequenceDecoder()
    fps_counter = FpsCounter()
    window_name = "Security - n: new car, r: reset, q: quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected)")
                break
            frame = undistort(frame, maps)
            found = detect_cards(frame)
            state = decoder.update(found)

            view = draw_cards(frame, found)
            slots = " ".join(state.digits.ljust(PLATE_LENGTH, "_"))
            title = "NEW CAR" if decoder.registering else "PLATE"
            put_text(view, f"{title}: {slots}", (20, 90), (255, 255, 255), 1.2)
            put_text(view, state.message, (20, h - 30), (0, 255, 0) if state.granted else (0, 255, 255), 0.9)
            draw_fps(view, fps_counter.tick())
            cv2.imshow(window_name, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("r"):
                decoder.registering = False
                decoder.reset()
            if key == ord("n"):
                decoder.start_registration()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
