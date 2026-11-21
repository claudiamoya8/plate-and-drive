# Plate and Drive

Final project for **Computer Vision I** (3rd year, Mathematical Engineering, Universidad Pontificia Comillas ICAI, 2026/27).

The idea: you unlock your car by showing the webcam its license plate, one printed card per digit, and then you drive it around a race track by moving a steering wheel in front of the camera. The system has to run in real time and use only classical computer vision with OpenCV.

> **Work in progress.** The camera calibration, the shared camera module (`src/camera.py`) and the security system are done; the calibration and the security system have their own sections below. The game and the steering wheel tracker are still being built, so those parts describe the plan and may change. We update this README as the project evolves.

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
- The goal is to **complete one lap as fast as possible**. **There are no other cars.**
- **Stop signs and traffic lights** are painted on the road so they can be seen from above:
  - **Stop:** you must come to a full stop inside a zone before the line. Crossing it without stopping **adds time** (5 s to start with, to be tuned while playing).
  - **Traffic light:** it changes color on a timer. Running a red light **adds time**.
- There are invisible **checkpoints** so you can't cut corners.
- The car is slower when it leaves the road.
- The car's **license plate is shown on top of it**.
- The screen also shows a small view of the camera with the tracker's box, and the **FPS are always visible**.

### 3. Driving with the steering wheel (tracker)

- Our first idea for the wheel is a **cardboard plate with two color marks**, one at the top and one at the bottom, each a different color.
- The tracker segments both marks by color in **HSV**, finds their centers, draws the **bounding box around the wheel** and uses a **Kalman filter** to keep going through short occlusions.
- **Steering:** the angle of the line joining the two marks.
- **Throttle and brake:** the distance in pixels between the two marks, which grows as you bring the wheel closer to the camera. There are three zones:
  - close → accelerate;
  - middle → keep the current speed;
  - far → brake until the car stops completely.

  That is how you stop at a stop sign. **There are no hand gestures.**
- If the wheel disappears from the image, the car brakes on its own.
- We also want a small tool to tune the color ranges to the lighting at each moment.

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

          Phase 2 · Game       tracker (HSV + Kalman) ──► controls ──► game update (dt) ──► track
                               shown together with the small camera view and the tracker box
```

## Design rules

These are the rules we agreed on so that the modules fit together:

- **One loop and one OpenCV window.** Each frame, the camera is read, processed and everything (game, small camera view, tracker box, FPS) is drawn into a single image shown with `imshow`. No Tkinter, no pygame, no extra threads.
- The game is drawn with OpenCV primitives (`rectangle`, `circle`, `fillPoly`, `putText`) on top of a background image of the track. A **mask** of the track tells which pixels are road.
- The game physics use the **real `dt`** between frames, so the car's speed does not depend on the FPS.
- Registered license plates are stored in `data/plates.json` and the leaderboard in `data/leaderboard.json`.
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

# Game
class Game:
    def __init__(self, plates: list[str]): ...  # 1 or 2 players
    def update(self, controls: list[Controls], dt: float) -> None: ...
    def draw(self) -> np.ndarray: ...  # BGR image of the track, to paste into the window
    def finished(self) -> bool: ...
    def times(self) -> dict[str, float]: ...  # license plate -> final time (with penalties)

# Security (already built, in src/security/)
def detect_cards(frame_bgr) -> list[Card]: ...   # Card(digit, corners): digit 0-9, or None if it can't be read
class SequenceDecoder:
    def update(self, cards: list[Card]) -> SecurityState: ...  # SecurityState(digits, granted, message)
    def reset(self) -> None: ...
    def start_registration(self) -> None: ...   # the next 4 digits are saved as a new car
# SecurityState.granted is the plate once access is granted, and None until then
```

### Building on the course labs

We plan to reuse the code and approach from the course labs wherever it fits:

- **Lab 1:** HSV color segmentation with `inRange`, masks and rotations with `warpAffine`.
- **Lab 2:** calibration with `findChessboardCorners`, `cornerSubPix` and `calibrateCamera`.
- **Lab 3:** Gaussian blur, Sobel, Canny, binarization, dilation and erosion.
- **Lab 4:** edges, corners, Hough transform, and the homography from SIFT matching (`findHomography` + `perspectiveTransform`), which we use to read the cards.
- **Lab 5:** background subtraction, optical flow, Kalman filter and meanShift.

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

1. Grey image, Gaussian blur and binarization, as in Lab 3. Instead of a fixed threshold of 127, the threshold is chosen in every frame with Otsu's method, because the light changes in live video and a dimly lit card would disappear.
2. `findContours` + `approxPolyDP` keep the four-sided shapes. These are not in the labs; they are the simple way to find a quadrilateral.
3. For each one, `findHomography` (Lab 4) maps the card layout onto the image, and `perspectiveTransform` (Lab 4) takes a grid of points from each zone of the card to the image, where we check whether they fall on black. The card is never straightened.
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
├── main.py                  # main loop: security -> game
├── src/
│   ├── camera.py            # open the camera, FPS, undistort
│   ├── calibration/
│   │   ├── capture.py       # saves chessboard photos with a key press
│   │   └── calibrate.py     # computes K, dist, rms -> data/calibration.npz
│   ├── security/
│   │   ├── cards.py         # generates the PDF with the 10 cards
│   │   ├── detector.py      # frame + homography + 7 segments
│   │   └── decoder.py       # sequence, registration, reset
│   ├── tracker/
│   │   └── wheel.py         # HSV + Kalman -> Controls
│   └── game/
│       ├── track.py         # drawing, road mask, stops, traffic lights, checkpoints
│       ├── car.py           # simple physics
│       └── game.py          # Game class, timer, penalties, leaderboard
├── data/                    # calibration.npz, plates.json, leaderboard.json (chessboard photos stay local)
└── docs/                    # report figures, block diagram
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
- We don't upload the professors' PDFs and slides, the demo video (it will be linked here) or `__pycache__`.

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
