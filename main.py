"""Plate and Drive: the main program.

One loop and one window (see "Design rules" in the README). Every frame the camera image is
undistorted and handed to the current phase:

1. Access: the player shows the 4 digits of the car's license plate with the cards
   (src/security). The cards are read on the original frame and the image is only mirrored
   to show it. With a registered plate, access is granted and, after a moment, the game starts.
   With p the turn is for two players: player 1 shows their plate and then player 2.
2. Game: the wheel tracker (src/tracker) works on the mirrored image, the one the players
   see (with two players, player 1 on the left half and player 2 on the right one), and turns
   each wheel into Controls for the game (src/game). During the countdown of the game the
   players hold their wheels straight, at a comfortable distance; when it ends, that position
   is taken as the neutral one (set_neutral). They drive as many laps as they want, and the
   game saves every valid lap in data/leaderboard.json. On screen: the game, the small camera
   view with the tracker boxes in the top-right corner and the FPS in the top-left one. The
   turn ends with e.
3. Leaderboard: the best lap of each plate and the best lap of each player in this turn, for
   a few seconds. Then the program goes back to the access for the next turn.

Usage, from the repository root:
    python main.py
Keys: access: n = register a new car, p = one or two players, r = start the plates again;
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
GRANTED_TIME = 1.5       # s the "Access granted" message stays on screen before going on
LEADERBOARD_TIME = 8.0   # s the leaderboard stays on screen after a turn

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
KEYS_POSITION = (1030, 710)      # bottom right, under the track: the keys of each phase
MESSAGE_POSITION = (20, 710)     # bottom left, under the track: what to do with the wheels
TURN_LINES = ((285, 32), (285, 62))   # end of a turn: one line per player, where the scoreboard was


def first_message(players):
    """What the access asks for at the start of a turn."""
    if players == 1:
        return "Show the first digit of your plate"
    return "Two players. Player 1: show the first digit of your plate"


def ask_next_plate(decoder, message, now):
    """Asks for another plate, for player 2 or after the same plate twice.

    The card of the last digit may still be in view: the decoder waits until it is taken
    away, as between two digits, so it is not read as the first digit of the next plate.
    """
    decoder.reset(message)
    decoder.waiting_removal = True
    decoder.last_card_time = now


def draw_access(image, state, registering, players, plates):
    """Texts of the access phase on the mirrored camera view: the plate so far, the message and the keys."""
    h = image.shape[0]
    slots = " ".join(state.digits.ljust(PLATE_LENGTH, "_"))
    if registering:
        title = "NEW CAR"
    elif players == 2:
        title = f"PLAYER {len(plates) + 1}"
    else:
        title = "PLATE"
    put_text(image, f"{title}: {slots}", (20, 90), WHITE, 1.2)
    if players == 2:
        done = f"   (player 1: {plates[0]})" if plates else ""
        put_text(image, f"Two players: player 1 on the left, player 2 on the right{done}", (20, 135), WHITE, 0.7)
    put_text(image, state.message, (20, h - 60), GREEN if state.granted else YELLOW, 0.9)
    put_text(image, "n: new car   p: one or two players   r: start again   q: quit", (20, h - 20), WHITE, 0.55)


def paste_camera(image, view):
    """Pastes the camera view, small, in the corner the track keeps free for it."""
    x0, y0, x1, y1 = CAMERA_CORNER
    image[y0:y1, x0:x1] = cv2.resize(view, (x1 - x0, y1 - y0))
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), WHITE, 2)


def draw_game_texts(image, game, controls):
    """Texts that main.py adds to the game, under the track: what to do with the wheels and the keys."""
    players = len(game.players)
    missing = [k + 1 for k, control in enumerate(controls) if not control.visible]
    if not game.started():
        if not missing:
            hint = "" if players == 1 else " (player 1 on the left)"
            put_text(image, f"Hold your wheel straight, at a comfortable distance{hint}", MESSAGE_POSITION, YELLOW, 0.7)
        elif players == 2:   # the game panel asks to show the wheel: say which one
            side = "left" if missing[0] == 1 else "right"
            put_text(image, f"Player {missing[0]}: show your wheel on the {side} half", MESSAGE_POSITION, YELLOW, 0.7)
    elif missing:
        lost = ", ".join(game.players[k - 1].plate for k in missing)
        put_text(image, f"WHEEL LOST ({lost}): braking", MESSAGE_POSITION, RED, 0.7)
    put_text(image, "e: end your turn   q: quit", KEYS_POSITION, WHITE, 0.55)


def draw_turn_over(game, rows, seconds_left):
    """Image of the end of a turn: the track and the cars, the leaderboard and the best lap of each player."""
    image = game.track.draw("green")
    for player in game.players:
        player.car.draw(image)
    draw_leaderboard(image, rows)
    best = game.best_laps()
    for (x, y), player in zip(TURN_LINES, game.players):
        if len(game.players) == 2:   # a square of the colour of the car, as in the scoreboard of the game
            cv2.rectangle(image, (x, y - 13), (x + 13, y), player.color, -1)
            cv2.rectangle(image, (x, y - 13), (x + 13, y), WHITE, 1)
            x += 22
        result = f"best lap of this turn {format_time(best[player.plate])}" if player.plate in best \
            else "no complete lap this turn"
        put_text(image, f"Plate {player.plate}: {result}", (x, y), WHITE, 0.75)
    put_text(image, f"Next turn in {seconds_left:.0f} s   q: quit", KEYS_POSITION, WHITE, 0.55)
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
    players = 1             # 1 or 2, changed with p in the access
    plates = []             # plates that already got access in this turn (player 1 first)
    granted_at = None       # when the plate being shown got access (the game goes on GRANTED_TIME later)
    game = tracker = None
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
                if state.granted in plates:   # the same car cannot be both players
                    ask_next_plate(decoder, f"Plate {state.granted} is player 1. Player 2: show another plate", t)
                    state = decoder.state()
                image = draw_cards(frame, found)   # mirrored copy with the cards outlined
                draw_access(image, state, decoder.registering, players, plates)
                if state.granted is None:
                    granted_at = None
                elif granted_at is None:
                    granted_at = t
                elif t - granted_at >= GRANTED_TIME:   # the plate was shown on screen long enough
                    plates.append(state.granted)
                    granted_at = None
                    if len(plates) < players:   # player 2 still has to show their plate
                        ask_next_plate(decoder, "Player 2: show the first digit of your plate", t)
                    else:   # play
                        game = Game(plates)
                        tracker = WheelTracker(players=players)
                        neutral_taken = False
                        phase = "game"
            elif phase == "game":
                view = cv2.flip(frame, 1)   # mirrored: the image the players see
                controls = tracker.update(view)   # player 1 (left half) first, as the game expects
                game.update(controls, dt)
                if game.started() and not neutral_taken:   # the countdown has just ended: both wheels at once
                    tracker.set_neutral()
                    neutral_taken = True
                image = game.draw()
                tracker.draw(view)
                paste_camera(image, view)
                draw_game_texts(image, game, controls)
            else:   # leaderboard after a turn
                seconds_left = LEADERBOARD_TIME - (t - turn_over_at)
                image = draw_turn_over(game, rows, seconds_left)
                if seconds_left <= 0:   # back to the access for the next turn
                    plates = []
                    who = "Next car" if players == 1 else "Player 1"
                    decoder.reset(f"Turn over. {who}: show the first digit of your plate")
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
                if key in (ord("r"), ord("p")):   # start the plates again (p: with the other number of players)
                    if key == ord("p"):
                        players = 3 - players   # 1 -> 2 and 2 -> 1
                    plates = []
                    granted_at = None
                    decoder.registering = False
                    decoder.reset(first_message(players))
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
