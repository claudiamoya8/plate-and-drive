"""Plate and Drive: the main program.

One loop and one window (see "Design rules" in the README). Every frame the camera image is
undistorted and handed to the current phase:

1. Access: the player shows the 4 digits of the car's license plate with the cards
   (src/security). The cards are read on the original frame and the image is only mirrored
   to show it. With a registered plate, access is granted and, after a moment, the game starts.
2. Game: the wheel tracker (src/tracker) works on the mirrored image, the one the player
   sees, and turns the plate into Controls for the game (src/game). During the countdown of
   the game the player holds the wheel straight, at a comfortable distance; when it ends,
   that position is taken as the neutral one (set_neutral). The player drives as many laps
   as they want, and the game saves every valid lap in data/leaderboard.json. On screen: the
   game, the small camera view with the tracker box in the top-right corner and the FPS in
   the top-left one. The turn ends with e.
3. Leaderboard: the best lap of each plate and the best lap of this turn, for a few seconds.
   Then the program goes back to the access for the next car.

Usage, from the repository root:
    python main.py
Keys: access: n = register a new car, r = start the plate again;
      game: e = end the turn; q / ESC (or closing the window) = quit.
"""
import time

import cv2

from src.camera import FpsCounter, draw_fps, load_undistort_maps, open_camera, put_text, undistort
from src.game.game import Game, draw_leaderboard, format_time, leaderboard
from src.game.track import CAMERA_CORNER
from src.security.decoder import PLATE_LENGTH, SequenceDecoder
from src.security.detector import detect_cards, draw_cards
from src.tracker.wheel import WheelTracker

WINDOW_NAME = "Plate and Drive"
GRANTED_TIME = 1.5       # s the "Access granted" message stays on screen before the game starts
LEADERBOARD_TIME = 8.0   # s the leaderboard stays on screen after a turn

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
KEYS_POSITION = (1030, 710)      # bottom right, under the track: the keys of each phase
MESSAGE_POSITION = (20, 710)     # bottom left, under the track: what to do with the wheel


def draw_access(image, state, registering):
    """Texts of the access phase on the mirrored camera view: the plate so far, the message and the keys."""
    h = image.shape[0]
    slots = " ".join(state.digits.ljust(PLATE_LENGTH, "_"))
    put_text(image, f"{'NEW CAR' if registering else 'PLATE'}: {slots}", (20, 90), WHITE, 1.2)
    put_text(image, state.message, (20, h - 60), GREEN if state.granted else YELLOW, 0.9)
    put_text(image, "n: new car   r: start again   q: quit", (20, h - 20), WHITE, 0.55)


def paste_camera(image, view):
    """Pastes the camera view, small, in the corner the track keeps free for it."""
    x0, y0, x1, y1 = CAMERA_CORNER
    image[y0:y1, x0:x1] = cv2.resize(view, (x1 - x0, y1 - y0))
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), WHITE, 2)


def draw_game_texts(image, game, controls):
    """Texts that main.py adds to the game, under the track: what to do with the wheel and the keys."""
    if not game.started() and controls[0].visible:
        put_text(image, "Hold your wheel straight, at a comfortable distance", MESSAGE_POSITION, YELLOW, 0.7)
    elif game.started() and not controls[0].visible:
        put_text(image, "WHEEL LOST: braking", MESSAGE_POSITION, RED, 0.7)
    put_text(image, "e: end your turn   q: quit", KEYS_POSITION, WHITE, 0.55)


def draw_turn_over(game, plate, rows, seconds_left):
    """Image of the end of a turn: the track and the car, the leaderboard and the best lap of this turn."""
    image = game.track.draw("green")
    game.car.draw(image)
    draw_leaderboard(image, rows)
    best = game.best_laps().get(plate)
    result = f"best lap of this turn {format_time(best)}" if best is not None else "no complete lap this turn"
    put_text(image, f"Plate {plate}: {result}", (285, 32), WHITE, 0.75)
    put_text(image, f"Next car in {seconds_left:.0f} s   q: quit", KEYS_POSITION, WHITE, 0.55)
    return image


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
    phase = "access"
    granted_at = None       # when access was granted (the game starts GRANTED_TIME later)
    game = tracker = plate = None
    neutral_taken = False   # set_neutral() is called once per turn, when the countdown ends
    rows, turn_over_at = [], None   # leaderboard read at the end of the turn, and when the turn ended
    fps_counter = FpsCounter()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected)")
                break
            t = time.perf_counter()
            dt, t_prev = t - t_prev, t   # real dt between frames, for the game physics
            frame = undistort(frame, maps)

            if phase == "access":
                found = detect_cards(frame)   # on the original frame: mirrored, a 2 would be read as a 5
                state = decoder.update(found)
                image = draw_cards(frame, found)   # mirrored copy with the cards outlined
                draw_access(image, state, decoder.registering)
                if state.granted is None:
                    granted_at = None
                elif granted_at is None:
                    granted_at = t
                elif t - granted_at >= GRANTED_TIME:   # the plate was shown on screen long enough: play
                    plate = state.granted
                    game = Game([plate])
                    tracker = WheelTracker(players=1)
                    neutral_taken = False
                    phase = "game"
            elif phase == "game":
                view = cv2.flip(frame, 1)   # mirrored: the image the player sees
                controls = tracker.update(view)
                game.update(controls, dt)
                if game.started() and not neutral_taken:   # the countdown has just ended
                    tracker.set_neutral()
                    neutral_taken = True
                image = game.draw()
                tracker.draw(view)
                paste_camera(image, view)
                draw_game_texts(image, game, controls)
            else:   # leaderboard after a turn
                seconds_left = LEADERBOARD_TIME - (t - turn_over_at)
                image = draw_turn_over(game, plate, rows, seconds_left)
                if seconds_left <= 0:   # back to the access for the next car
                    decoder.reset("Turn over. Next car: show the first digit of your plate")
                    granted_at = None
                    phase = "access"

            draw_fps(image, fps_counter.tick())
            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if phase == "access":
                if key == ord("n"):
                    decoder.start_registration()
                if key == ord("r"):
                    decoder.registering = False
                    decoder.reset()
            elif phase == "game" and key == ord("e"):   # end of the turn: show the leaderboard
                rows = leaderboard()   # read once: the laps of this turn are already saved
                turn_over_at = t
                phase = "leaderboard"
            # Closing the window with its X also quits (otherwise imshow would open it again)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
