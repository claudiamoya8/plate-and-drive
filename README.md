# Plate and Drive

Final project for **Computer Vision I** (3rd year, Mathematical Engineering, Universidad Pontificia Comillas ICAI, 2026/27).

The idea: you unlock your car by showing the webcam its license plate, one printed card per digit, and then you drive it around a race track by moving a steering wheel in front of the camera. The system has to run in real time and use only classical computer vision with OpenCV.

> **Work in progress.** The whole system is built: the camera calibration, the shared camera module (`src/camera.py`), the security system, the steering wheel tracker, the game and the main program (`main.py`), also for two players. Each part has its own section below. We are now measuring the results for the report and preparing the demo video. We update this README as the project evolves.

**Key dates (group B):** idea approval on 20 November 2026; final delivery and presentation on 4 December 2026.

## What the course asks for

The assignment is a real-time **classical** computer vision system that uses a camera as its input. It has two main blocks:

1. **Security system:** the user identifies themselves by showing visual patterns to the camera. A decoder remembers up to 4 patterns in a row and grants or blocks access depending on whether they come in the right order. It must be possible to reset the sequence.
2. **Proposed system (free topic):** once the right sequence is entered, a **tracker** starts. It draws a bounding box around the region of interest and follows it as it moves.

On top of that, the camera has to be **calibrated** with a chessboard pattern. Calibration is done *offline*, before running the system, and the report must include the camera matrix K, the distortion coefficients and the RMS error.

**No deep learning.** The course is about classical computer vision, so we won't use MediaPipe, neural networks or YOLO. Only classical OpenCV techniques such as color segmentation, contours, morphology, homographies and Kalman filtering.

Grading is `N × F`:

| Module | Points |
|---|---|
| Camera calibration | 1 |
| Pattern detector | 2 |
| Information extraction (sequence decoder) | 2 |
| Tracker | 2 |
| Real-time video output | 1 |
| Extensions | 2 |

`F` = report (0.5) + repository (0.25) + video (0.25).

## The idea

This is the plan we are starting from.

### 1. Access: your car's license plate

- Every car has a **4-digit license plate**, numbers only (for example `2047`).
- To get in, you show the camera **4 printed cards**, one per digit, in order.
- Each card is a **seven-segment digit**, like the ones on a digital clock: black bars on a white background, inside a **thick black frame**, with a **small mark in the top-left corner** to tell the orientation and a white margin around everything.

This part is already built: how the cards are read and how the decoder works is explained in [Security system (access)](#security-system-access).

### 2. The game: time trial

- A **fixed track seen from above** (top-down view) that fits on a single screen, with no scrolling.
- You can drive **as many laps as you want**: every valid lap is timed and saved, and the leaderboard keeps the best lap of each license plate. **There are no other cars.**
- **Stop signs and traffic lights** are painted on the road so they can be seen from above:
  - **Stop:** you must come to a full stop inside a zone before the line. Crossing it without stopping **adds time** (5 s to start with, to be tuned while playing).
  - **Traffic light:** it changes color on a timer. Running a red light **adds time**.
- There are invisible **checkpoints** so you can't cut corners.
- There are walls on both sides of the road: the car cannot go through them, and it does not get stuck when it hits one.
- The car's **license plate is shown on top of it**.
- The screen also shows a small view of the camera with the tracker's box, and the **FPS are always visible**.

### 3. Driving with the steering wheel (tracker)

- The wheel is a **white plastic plate with two color marks**, one at the top and one at the bottom, each a different color.
- The tracker follows both marks by color, draws the **bounding box around the wheel** and keeps going through short occlusions. How it works is explained in [Steering wheel tracker](#steering-wheel-tracker).
- **Steering:** the angle of the line joining the two marks.
- **Throttle and brake:** the distance in pixels between the two marks, which grows as you bring the wheel closer to the camera. There are three zones:
  - close → accelerate;
  - middle → keep the current speed;
  - far → brake until the car stops completely.

  That is how you stop at a stop sign. **There are no hand gestures.**
- If the wheel disappears from the image, the car brakes on its own.
- A small tool tunes the color ranges to the lighting of the room.

### 4. Extensions

1. **Register a new car:** from the start screen anyone can choose "new car", show 4 cards, and that license plate is saved.
2. **Two players at once on the same track**, each with their own wheel and a car of a different color with its license plate. Both wheels use the same colors; the system tells them apart by the **half of the image** they are in (left player and right player).
3. **Leaderboard:** the best lap for each license plate is saved.

### Ideas we ruled out

While shaping the idea we also considered these options and decided **not** to do them:

- Stopping the car by showing the palm of your hand.
- QR-style codes for the security system.
- Reading handwritten numbers, or using a neural network to read them.
- Measuring the real distance of the wheel using the calibration.
- Split screen.
- Computer-controlled cars.

## System overview

A rough picture of how the pieces fit together, based on the plan above:

```
OFFLINE   chessboard photos ──► calibration ──► data/calibration.npz (K, dist, RMS)

LIVE      every frame:  webcam ──► undistort (remap) ──► current phase ──► output window + FPS

          Phase 1 · Security   card detector ──► sequence decoder ──► registered plate? ──yes──► Phase 2

          Phase 2 · Game       tracker (HSV + meanShift + Kalman) ──► controls ──► game update (dt) ──► track
                               shown together with the small camera view and the tracker box ──key e──► Phase 3

          Phase 3 · Leaderboard  best lap of each license plate, for 8 s ──► back to Phase 1
```

## Design rules

These are the rules we agreed on so that the modules fit together:

- **One loop and one OpenCV window.** Each frame, the camera is read, processed and everything (game, small camera view, tracker box, FPS) is drawn into a single image shown with `imshow`. No Tkinter, no pygame, no extra threads.
- The game is drawn with the drawing functions used in the course labs (`line`, `circle`, `rectangle`): the road is a chain of very thick `line` segments, so `fillPoly` is not needed. Texts (FPS, license plates) use `putText` through `put_text` and `draw_fps` in `src/camera.py`. A separate one-channel mask, drawn with the same thick lines, tells which pixels are road.
- The game physics use the **real `dt`** between frames, so the car's speed does not depend on the FPS.
- Registered license plates are stored in `data/plates.json` and the leaderboard in `data/leaderboard.json`, which is not uploaded: every computer keeps its own laps.
- Calibration parameters are stored in `data/calibration.npz` (`K`, `dist`, `rms`). The main program only loads them. To undistort in real time, `initUndistortRectifyMap` is called once and `remap` on every frame, instead of `cv2.undistort`.
- The code has to be fast: no per-pixel loops in Python, and working on downscaled images whenever possible.

### Module interfaces

These planned interfaces let us work on different modules in parallel without getting in each other's way. All code identifiers are in English.

```python
# What the tracker (or the keyboard, at first) gives the game on every frame, per player
@dataclass
class Controls:
    steering: float  # -1 (full left) .. +1 (full right)
    throttle: float  # -1 (full brake) .. 0 (keep speed) .. +1 (full throttle)
    visible: bool    # False if the wheel is lost -> the game brakes the car

# Game (free laps: main.py ends the turn with a key)
class Game:
    def __init__(self, plates: list[str]): ...  # 1 or 2 players: player 1 (left half of the camera) first
    players: list[Player]  # in the same order; each one has its .plate and its .car
    def started(self) -> bool: ...  # the countdown has ended: main.py calls tracker.set_neutral() then
    def draw_cars(self, image) -> None: ...  # all the cars and then all their plates (for the end-of-turn screen)
    def update(self, controls: list[Controls], dt: float) -> None: ...  # every valid lap is saved in data/leaderboard.json
    def draw(self) -> np.ndarray: ...  # BGR image of the track, to paste into the window
    def best_laps(self) -> dict[str, float]: ...  # license plate -> best lap of this turn (with penalties)
def leaderboard(n: int = 5) -> list[tuple[str, float]]: ...  # best laps ever, one per license plate, fastest first
def draw_leaderboard(image, rows: list[tuple[str, float]]) -> None: ...  # leaderboard panel for the end of a turn

# Security (already built, in src/security/)
def detect_cards(frame_bgr) -> list[Card]: ...   # Card(digit, corners): digit 0-9, or None if it can't be read
class SequenceDecoder:
    def update(self, cards: list[Card]) -> SecurityState: ...  # SecurityState(digits, granted, message)
    def reset(self) -> None: ...
    def start_registration(self) -> None: ...   # the next 4 digits are saved as a new car
# SecurityState.granted is the plate once access is granted, and None until then

# Wheel tracker (already built, in src/tracker/)
class WheelTracker:
    def __init__(self, players: int = 1): ...         # with 2 players, each one has half of the image
    def update(self, view_bgr) -> list[Controls]: ...  # view = the mirrored camera image, as the players see it
    def set_neutral(self) -> None: ...                 # wheels straight, at a comfortable distance
    def draw(self, view_bgr) -> np.ndarray: ...        # box, marks and line of every wheel
```

### Building on the course labs

Wherever it fits, we build on the code and methods of the course labs. This is what we take from each one:

- **Lab 1:** HSV color segmentation with `inRange`, the trackbar tool to find color ranges, rotations with `warpAffine`, and combining images with masks (`bitwise_and`, `bitwise_or`).
- **Lab 2:** calibration with `findChessboardCorners`, `cornerSubPix` and `calibrateCamera`, and undistortion with `initUndistortRectifyMap` + `remap`.
- **Lab 3:** Gaussian blur and binarization for the cards, and erosion and dilation for the wheel marks.
- **Lab 4:** the homography from SIFT matching (`findHomography` + `perspectiveTransform`), which we use to read the cards.
- **Lab 5:** the live video loop, the Kalman filter and meanShift for the wheel tracker, and the drawing functions (`line`, `circle`, `rectangle`).

## Camera calibration

Done offline, once, before running the system. It follows the method from Lab 2, turned into two scripts in `src/calibration/`:

```bash
conda activate voi-lab
python -m src.calibration.capture     # live view; SPACE saves a photo (only if the whole chessboard is visible), q quits
python -m src.calibration.calibrate   # computes K, the distortion coefficients and the RMS error
```

- **Chessboard:** `damero_8x6_25mm_A4.pdf`, 9×7 squares (8×6 inner corners), stuck to a rigid folder. Our printer shrank it slightly: seven squares measure 170 mm, so we use **24.29 mm** per square instead of 25 mm.
- **Photos:** 28 photos at 1280×720 with the laptop's webcam, at different distances and angles. `capture.py` paints in green the part of the image the photos already cover, which is how we noticed the corners of the image were missing in the first 20 photos.
- **Method:** `findChessboardCorners`, `cornerSubPix` with an (11, 11) window, 3D points on the plane Z = 0 and `calibrateCamera`.
- The photos are **not uploaded**: a person appears in them. They stay in `data/chessboard_photos/`, which is in `.gitignore`. Only the results are uploaded: `data/calibration.npz` (used by the program) and `data/calibration.txt` (readable summary).

**Result** (webcam at 1280×720):

| fx, fy (px) | cx, cy (px) | k1, k2, k3 | p1, p2 | RMS (px) |
|---|---|---|---|---|
| 825.35, 824.40 | 646.95, 351.14 | 0.0141, −0.1609, 0.1466 | 0.0027, 0.0004 | 0.384 |

On average, the corners predicted by the model land 0.38 px away from the detected ones. Calibrating again without a quarter of the photos and measuring the error on those gives 0.39 px, so the model is not just fitting the photos it has seen. The calibration is only valid for this camera at 1280×720.

## Security system (access)

To unlock a car, you show the webcam the four digits of its license plate, one card at a time. Everything is in `src/security/`:

```bash
python -m src.security.cards      # writes docs/cards.pdf: the 10 cards, ready to print
python -m src.security.detector   # live view: outlines every card and writes its digit (b = binary image, q = quit)
python -m src.security.decoder    # the whole access (n = register a new car, r = reset, q = quit)
```

**The cards.** Each card draws one digit the way a digital clock does: seven black bars on white, inside a thick black frame, with a small square in the top-left corner so we know which way up the card is. The layout is defined only once, in `cards.py`, and the detector reads the same numbers, so the two can never get out of step. The printed size does not matter, only the proportions. The cards can also be shown on a tablet screen, which is how we tested them before printing.

**Reading a card** (`detector.py`):

1. Grey image, Gaussian blur and binarization. Instead of the fixed threshold of 127 we used in Lab 3, the threshold is chosen in every frame with Otsu's method, because the light changes in live video and a dimly lit card would disappear.
2. `findContours` + `approxPolyDP` keep the four-sided shapes. These are not in the labs; they are the simple way to find a quadrilateral.
3. For each one, `findHomography` maps the card layout onto the image, and `perspectiveTransform` takes a grid of points from each zone of the card to the image, where we check whether they fall on black. The card is never straightened.
4. It only counts as one of our cards if the four sides of the frame are black and the mark is in one corner with the other three corners white. Trying each of the four corners as the top-left one tells us how the card is rotated, so a 6 upside down is not read as a 9.
5. A segment is on if at least 75 % of its points are black, and off if at most 25 % are. If one is in between (a finger, a reflection), the card is marked as unreadable instead of guessing, because some digits differ in a single segment, like 8 and 0.

Cards are always read on the original frame. The image is only mirrored when it is shown on screen; otherwise a 2 would be read as a 5.

**The decoder** (`decoder.py`) is a small state machine:

- A digit is accepted once it has been read the same way for 0.5 s, so one wrong reading never counts.
- Between two digits the card has to leave the image for 0.3 s. That is what makes repeated digits work (`1123` needs only one card of each number), and a single missed frame is not mistaken for the card being taken away.
- With 4 digits, the plate is looked up in `data/plates.json`. If it is there, access is granted; if not, the screen says so and the sequence starts again.
- An unfinished plate is forgotten after 7 s without any card, and `r` resets it at any time.
- Extension: after pressing `n`, the next 4 digits are saved as a new car.

**Results so far.** On the laptop, the whole access loop runs at about 30 FPS, which is the camera's limit; reading the cards takes about 3 ms per frame. With simulated camera views (the ten digits at six angles, with perspective, blur and noise), the detector read 60 cards out of 60. It never gave a wrong digit when a segment was half covered, and it found no false cards in 40 cluttered scenes or in 150 real frames of the room.

## Steering wheel tracker

Each player drives with a white plastic plate held like a steering wheel, with the hands on the sides. On its flat part there are two colored circles about 5 cm wide, one at the top and one at the bottom, 9 cm apart: green at the top and magenta at the bottom, printed from `docs/wheel_marks.pdf`. Everything is in `src/tracker/`:

```bash
python -m src.tracker.marks   # writes docs/wheel_marks.pdf: the colored marks, ready to print
python -m src.tracker.tune    # tunes the colors of the marks (1 = top mark, 2 = bottom mark, s = save, q = quit)
python -m src.tracker.wheel   # live demo (n = neutral position, p = 1 or 2 players, m = masks, q = quit)
```

**Tuning the colors** (`tune.py`). It is the trackbar tool we used in Lab 1, working on the live camera. Six trackbars set the HSV range of one mark, and the window shows the camera next to the pixels the tracker will take as that mark. The ranges are saved in `data/wheel_colors.json`. They depend on the light, so they have to be tuned again wherever we play. Our first marks, painted with highlighters, were too pale: under the warm light of our room the green looked lime (hue 29–35 instead of the usual 35–85), the pink had a saturation of only about 75 out of 255, and the marks disappeared when the wheel came close or went far, because the camera changes its exposure. That is why we now print them: printed colors are much more saturated.

**Finding and following the marks** (`wheel.py`):

1. HSV image and one mask per mark color with `inRange`.
2. Erosion and then dilation with a 3×3 kernel, so specks of a few pixels disappear and the marks keep their size. We use `cv2.erode` and `cv2.dilate` instead of the hand-written loops of Lab 3, which would be far too slow for live video.
3. While a wheel is not being followed (at the start, or after losing it), `findContours`, which we also use for the cards, finds every blob of each color. The wheel is the pair of blobs, one green and one magenta, placed most like the marks on the plate: about the same size and about 1.8 diameters apart. Because every blob is looked at, other green or magenta things in the room do not stop the wheel from being found.
4. Each mark is then followed with a Kalman filter and `meanShift`. The filter, with a constant velocity model, predicts where the mark will be; `meanShift` looks for it there; and the centre it finds corrects the filter. In Lab 5 `meanShift` worked on the back projection of a hue histogram; here it works on the color mask, because the range of the mark is already known. If a hand covers a mark for a moment, its predicted position is used. After 10 frames without it (a third of a second), the wheel counts as lost and is searched for again as in step 3.
5. The box around the wheel is drawn with `rectangle`, in the color of the player's car in the game: red for player 1 and blue for player 2.

The wheels are tracked on the mirrored image, the one the players see, so left and right mean the same for them and for the program. With two players, each one has half of the image: player 1 on the left, player 2 on the right.

**From the wheel to the car.** Everything is measured from a neutral position: the wheel straight and at a comfortable distance. In the demo it is taken with `n`; in [the main program](#main-program) it is taken when the countdown before the start ends.

- **Steering:** the angle of the line between the two marks. Up to 5° it counts as straight, and at 60° the steering is full.
- **Throttle:** the distance in pixels between the marks, compared with the neutral one. Between 92 % and 108 % of it the car keeps its speed. Closer, it accelerates, with full throttle at 125 %; farther, it brakes, with a full brake at 80 %. The values grow gradually instead of jumping, so the small shaking of the hands does not make the car jerk.
- If a wheel is lost, the game gets `Controls(0, -1, False)` and the car brakes.

**Results so far.** With the real wheels the demo runs at about 30 FPS, and the tracker takes about 3.5 ms per frame. With simulated camera views (blur, noise, darker and brighter light, and a background with skin, wood and denim colors), the angle error stayed under 1.5° and the distance error under 3 % between 0.5 and 2 m, and a wheel moving away from the camera was followed up to 3.5 m. A mark covered for 8 frames was kept with an error of 0.6°. Other green or magenta objects in the room, even a sticker 16 pixels wide, neither stopped the wheel from being found nor were followed instead of a mark. When a finger covers half of a mark, the angle moves by about 7.5°, because the centre of the mark moves to its visible half.

## Game

A time trial on a single track seen from above, drawn with the drawing functions of the labs. It is for one or two players. It can be played with the keyboard, without a camera, and [the main program](#main-program) plays it with the wheels. Everything is in `src/game/`, plus `src/controls.py`:

```bash
python -m src.game.demo   # the game with the keyboard (W/S/A/D = drive, p = 1 or 2 players, e = end the turn, r = start again, m = masks, q = quit)
python -m src.controls    # test window for the keyboard controls
```

**The track** (`track.py`). One track of 1280×720 px that fits on the screen, with the top-left and top-right corners kept free for the FPS and the small camera view.

1. The centre line is a closed polygon of 13 points placed by hand, smoothed with Chaikin's algorithm (Chaikin, 1974): every side is replaced by two points at 1/4 and 3/4 of it, and after 4 rounds the corners have become round curves. This is not in the labs; it is a simple way to get a smooth track from a few points. The lap starts at the finish line and is about 2,540 px long.
2. The road is that line drawn as a chain of very thick `line` segments (110 px), so `fillPoly` is not needed. Red and white walls of 12 px run along both edges, and there are a dashed centre line, a checkered finish line, the STOP and the traffic light, all drawn with `line`, `circle` and `rectangle`. Everything except the traffic light never changes, so this background is drawn only once; every frame copies it and draws on top only the traffic light, the car and the texts.
3. Three one-channel masks are drawn apart with the same thick lines, never taken from the colors of the image: the road; where the centre of the car can be (the road 12 px narrower on each side, so the car never goes into the walls); and the zones, where every pixel holds the number of its zone (the finish line, the four checkpoints, the STOP area and its line, and the traffic light line). The zones are kept inside the road with `bitwise_and`, as in Lab 1. To know where the car is, the game reads a single pixel of a mask.

**The car** (`car.py`). A car of 40×22 px, red for player 1 and blue for player 2, drawn once with `rectangle`, with its license plate written above it in the same color.

- **Values:** top speed 260 px/s, acceleration 180 px/s² with full throttle, braking 340 px/s² with full brake (from full speed to 0 in about 0.8 s) and a turn rate of 130°/s with full steering, also when stopped. There is no reverse gear. With a throttle of 0 it keeps its speed, like the middle zone of the wheel, and if the wheel is lost it brakes with the wheels straight. We tuned the braking by playing with the keyboard, and the top speed and the turn rate by playing with the wheel.
- It moves with the real `dt` between frames, in steps of at most 4 px, so it never jumps over a line of the zones mask.
- **Walls:** a step that would take the car out of its mask slides along the wall instead. Only the part of the step along the road is kept (its projection on the direction of the closest point of the centre line), with a push of 1 or 2 px towards the middle on the curves. When the car hits a wall it keeps only its speed along the wall (the speed times the cosine of the angle with the wall), and while it keeps pushing against it, it slows down to 100 px/s. It never gets stuck, and it can always turn away.
- Every frame it is rotated with `getRotationMatrix2D` + `warpAffine` and pasted on the track with a mask, using `bitwise_and` and `bitwise_or`, as in Lab 1. The rotation uses `INTER_NEAREST`, so the mask keeps only 0 and 255 and the edges of the car stay clean.

**Keyboard controls** (`src/controls.py`). The game receives `Controls(steering, throttle, visible)` on every frame, the same from the keyboard as from the tracker. `cv2.waitKey` only reports key presses, so every press of W/S/A/D (I/K/J/L for player 2) adds 0.2 in its direction, and the opposite key drops the value to 0 at once. A key counts as released when no press of it has arrived for 0.6 s (Windows repeats a held key only after about 0.5 s), and then the value goes back to 0 little by little. Windows only repeats the last key held down, so with the keyboard two players cannot hold their keys at the same time; the two-player mode is really played with the wheels.

**Free laps and the leaderboard** (`game.py`):

1. A 3 s countdown with the car stopped on the start. It waits while the wheel cannot be seen, and when it ends, `main.py` takes the neutral position of the wheel.
2. A lap is valid if the car goes through four invisible checkpoints in order and then crosses the finish line, so corners cannot be cut. The zones mask is read at every step the car took in the frame, so no line is missed even if the FPS drop.
3. Every valid lap is saved at once in `data/leaderboard.json`, with the license plate, the time with its penalties and the penalties, so nothing is lost if the program is closed. The file is created with the first lap and is not uploaded: every computer keeps its own laps. Then the timer, the checkpoints and the penalties start again from 0, and the next lap goes on without stopping. A lap ends at the step where the car touched the finish line; the rest of that frame already counts for the next lap.
4. The player drives as many laps as they want. The scoreboard at the top shows the lap, its time and penalties, the last lap and the best one, and for 2 s after each lap a panel on the grass shows its time, with the license plate of the car. When the turn ends, the leaderboard shows the five fastest license plates, each with its best lap.

The lap time is game time: each frame adds its `dt`, limited to 0.05 s, so if the program freezes for a moment the car does not jump and the time does not run away. All the texts are written with `put_text`, at fixed positions.

**The STOP and the traffic light.** There is one of each, and breaking either rule adds 5 s to the lap in which it happens:

- **STOP:** the car has to stop completely (below 5 px/s) inside the area before the thick white line. Crossing the line without having stopped there adds the penalty. The area is about 90 px long and braking from full speed takes about 100 px, so you have to start braking a little before it, as at a real stop sign.
- **Traffic light:** green for 4 s, yellow for 1 s and red for 3 s, over and over, from the "GO!". Crossing its line while it is red adds the penalty; on yellow it does not.
- Each line is checked the first time the car touches it in a lap, so a penalty is never counted twice. For 2 s the panel shows "STOP" or "RED LIGHT", with the license plate of the car and "+5 s".

**Two players.** `Game([plate_1, plate_2])`: player 1 is the one on the left half of the camera image, and `controls[0]` is theirs.

- The two cars start side by side, 80 px apart, player 1 on the left of the road. They share the track, the screen and the traffic light, but they do not crash into each other: they go through each other, like ghosts, and each one does its own time trial, with its own laps, penalties and timer. Split screen is still ruled out.
- The countdown waits while either wheel cannot be seen, and when it ends `started()` becomes True once, so `main.py` takes the neutral position of both wheels at the same time.
- Each car saves its laps with its own license plate, and `best_laps()` gives the best lap of each player.
- The scoreboard has one line per player, after a small square of the color of their car.
- All the cars are drawn first and then all the plates, so when the cars cross each other, a car never hides the plate of the other.
- In the demo, `p` changes between one and two players.

**Results so far.** On the laptop of the demo, updating and drawing the game takes about 3 ms per frame with one player and about 4 ms with two, so the game does not limit the FPS. In simulation, a driver who follows the road at full speed and never stops takes 9.9 s per lap, plus its penalties (always the STOP, and the traffic light when it is red), and one who stops at the STOP and waits at the red light takes between 11 and 13 s, with no penalties. With the real tracker following a drawn wheel moved by a simulated player, a lap took 13.6 s (before the STOP and the traffic light were added).

## Main program

`main.py` puts everything together in one loop and one window:

```bash
python -m src.tracker.tune   # first, wherever you are going to play: tune the colors of the wheel marks to the light
python main.py               # the whole program (n = new car, p = one or two players, r = start the plates again, e = end the turn, q = quit)
```

Every frame the camera image is undistorted and goes to the current phase:

1. **Access.** The player shows the four digits of the license plate, one card at a time, as in `python -m src.security.decoder`. The cards are read on the original frame, and the image is only mirrored to show it. With a registered plate the screen says "Access granted" and the game starts 1.5 s later. `n` registers a new car and `r` starts the plate again.
   - **Two players** (`p`): player 1 shows their plate and then player 2. Before player 2's first digit, the last card has to be taken away, as between two digits, so it is not read twice. The same plate cannot be both players. Player 1 stands on the left half of the image and player 2 on the right one.
2. **Game.** The tracker works on the mirrored image and turns each wheel into `Controls` for the game, player 1 first. During the countdown the players hold their wheels straight, at a comfortable distance; when it ends (`Game.started()`), that position becomes the neutral one of every wheel at once (`set_neutral`). The players drive as many laps as they want, and every valid lap is saved in `data/leaderboard.json`. On screen: the game, the small camera view with the tracker boxes in the top-right corner, the FPS in the top-left one and, under the track, what to do with the wheels (which wheel is missing, or lost) and the keys. `e` ends the turn.
3. **Leaderboard.** For 8 s, the best lap of each license plate and the best lap of each player in this turn. Then the program goes back to the access for the next turn, with the same number of players.

Every turn starts with a new tracker, so one player's neutral position is never used for the next one. `q`, `ESC` or closing the window quits at any moment.

## Hardware and environment

- **Camera:** the built-in webcam of Rodrigo's laptop (Windows, index 0, `cv2.CAP_DSHOW`, 1280×720, about 30 FPS measured). It is used for calibration, for tuning the wheel colors and for recording the demo.
- **The game has to be able to run on its own, without a camera,** using a keyboard test mode that goes through the same `Controls` interface. This lets us develop the game on a computer without a webcam.
- **Python 3.9** with the course's conda environment, `voi-lab`: `opencv-python==4.8.0.76`, `numpy==1.26.0`, `matplotlib`, `scipy` and `scikit-image`. We won't add other dependencies without agreeing on it first.

```bash
conda activate voi-lab
```

`requirements.txt` lists the exact libraries the code uses so far. We will add instructions on how to run each part as they are built.

## Repository structure (planned)

This is the structure we plan to follow. It may change as the project grows.

```
.
├── README.md
├── requirements.txt
├── main.py                  # main loop: access -> game -> leaderboard
├── src/
│   ├── camera.py            # open the camera, FPS, undistort
│   ├── controls.py          # Controls class and keyboard control (test mode without camera)
│   ├── calibration/
│   │   ├── capture.py       # saves chessboard photos with a key press
│   │   └── calibrate.py     # computes K, dist, rms -> data/calibration.npz
│   ├── security/
│   │   ├── cards.py         # generates the PDF with the 10 cards
│   │   ├── detector.py      # frame + homography + 7 segments
│   │   └── decoder.py       # sequence, registration, reset
│   ├── tracker/
│   │   ├── marks.py         # generates the PDF with the colored marks of the wheels
│   │   ├── tune.py          # tunes the colors of the marks with trackbars
│   │   └── wheel.py         # HSV + meanShift + Kalman -> Controls
│   └── game/
│       ├── track.py         # drawing, road mask, stops, traffic lights, checkpoints
│       ├── car.py           # simple physics
│       ├── game.py          # Game class, timer, penalties, leaderboard
│       └── demo.py          # playable keyboard demo without camera: python -m src.game.demo
├── data/                    # calibration.npz, plates.json, wheel_colors.json, leaderboard.json (chessboard photos and leaderboard.json stay local)
└── docs/                    # cards.pdf, wheel_marks.pdf, report figures, block diagram
```

## Team and work split

This is our initial split. We may change it along the way.

| Who | Modules | Branches |
|---|---|---|
| **Rodrigo** | Calibration, shared camera module, seven-segment cards (design and printing), digit detector, decoder, registering new plates | `calibration`, `camera`, `security` |
| **Claudia** | The whole game: track, car, stops, traffic lights, checkpoints, timer and leaderboard. At first it is controlled with the **keyboard** so it doesn't depend on the wheel. | `game` |
| **Both** | Wheel tracker, integration in `main.py`, two-player mode, report, video and README | `tracker`, `integration` |

### Roadmap

1. Camera calibration.
2. In parallel: Claudia builds the game with keyboard controls and Rodrigo builds the access system.
3. Wheel tracker, which replaces the keyboard.
4. Stop signs, traffic lights and leaderboard.
5. Two players.
6. The last 4 or 5 days, only the report and the video.

### How we work with Git

- One branch per module (`calibration`, `camera`, `security`, `tracker`, `game`, `integration`), merged into `main` once it works.
- Every script is run from the repository root with `python -m`, for example `python -m src.camera`.
- Always `git pull` before starting.
- Nobody edits another module's files at the same time.
- Small commits from both of us, with clear messages.
- Code, comments, this README and commit messages are in English.
- We don't upload the professors' PDFs and slides, the demo video (it will be linked here), `data/leaderboard.json` or `__pycache__`.

## Deliverables

- **This GitHub repository**, public, with a complete and well-structured README.
- **A demo video** supported by slides or a poster. The **FPS are shown on screen at all times** and everything is processed in real time.
- **A report** of at most **5 pages**, including the link to this repository: introduction, methodology (calibration, block diagram, sequence of image transformations, security system, proposed system), results and future work, with a bibliography.
- Everything packed in `LabProject_Rodrigo_Claudia.zip`.

## Demo

Video: *coming soon*.

## Authors

- Rodrigo Sicilia
- Claudia Moya
