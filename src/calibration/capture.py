"""Capture chessboard photos to calibrate the camera.

Opens the webcam and shows live whether the chessboard is visible. Pressing
SPACE saves the original frame, without mirroring or drawings, but only if the
whole chessboard is detected. The parts of the image already covered by the
saved photos are painted green, so the photos can be spread over the whole image.

Keys:
    SPACE    save photo (only if the whole chessboard is visible)
    q / ESC  quit

Photos are saved in data/chessboard_photos/, which is not uploaded to GitHub
(it is in .gitignore because people appear in the photos).

Usage, from the repository root:
    python src/calibration/capture.py                      # camera 0, 1280x720
    python src/calibration/capture.py --camera 1 --width 1280 --height 720
"""
import argparse
import glob
import os
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
PATTERN = (8, 6)  # inner corners of the chessboard (columns, rows)
PREFIX = "chessboard_"  # photos are saved as chessboard_000.png, chessboard_001.png, ...


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW: Windows only
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def detect_fast(frame, scale=0.5):
    """Looks for the chessboard on a downscaled copy so the FPS don't drop.

    Returns the corners in coordinates of the original image, or None.
    """
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not ok:
        return None
    return corners / scale


class BackgroundDetector:
    """Looks for the chessboard in a separate thread so the live view keeps its FPS.

    findChessboardCorners takes between 50 and 180 ms per frame on this laptop;
    done in the main loop, the live view dropped to about 6 FPS.
    """

    def __init__(self):
        self._frame = None
        self._corners = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._frame = frame

    def result(self):
        with self._lock:
            return self._corners

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)

    def _loop(self):
        while self._running:
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                time.sleep(0.005)
                continue
            corners = detect_fast(frame)
            with self._lock:
                self._corners = corners


def next_filename(folder):
    existing = glob.glob(os.path.join(folder, PREFIX + "*.png"))
    start, end = len(PREFIX), len(PREFIX) + 3
    numbers = [int(os.path.basename(p)[start:end]) for p in existing if os.path.basename(p)[start:end].isdigit()]
    n = max(numbers) + 1 if numbers else 0
    return os.path.join(folder, f"{PREFIX}{n:03d}.png"), len(existing)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--folder", default=os.path.join(DATA, "chessboard_photos"))
    args = ap.parse_args()

    os.makedirs(args.folder, exist_ok=True)
    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        print(f"Could not open camera {args.camera}. Try --camera 1.")
        return

    ok, frame = cap.read()
    if not ok:
        print("The camera opens but returns no frames.")
        return
    h, w = frame.shape[:2]
    print(f"Camera {args.camera} opened at {w}x{h}. Photos in: {args.folder}")

    # Areas already covered by saved photos (including those from previous sessions)
    coverage = np.zeros((h, w), np.uint8)
    for path in sorted(glob.glob(os.path.join(args.folder, PREFIX + "*.png"))):
        img = cv2.imread(path)
        if img is not None and img.shape[:2] == (h, w):
            c = detect_fast(img)
            if c is not None:
                cv2.fillConvexPoly(coverage, cv2.convexHull(c.astype(np.int32)), 255)

    window = "Calibration - SPACE saves, q quits"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    t_prev = time.perf_counter()
    fps = 0.0
    notice, t_notice = "", 0.0
    detector = BackgroundDetector()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stopped receiving frames.")
            break

        detector.submit(frame)
        corners = detector.result()  # result from a very recent frame

        view = frame.copy()
        green = np.zeros_like(view)
        green[coverage > 0] = (0, 160, 0)
        view = cv2.addWeighted(view, 1.0, green, 0.35, 0)
        if corners is not None:
            cv2.drawChessboardCorners(view, PATTERN, corners.astype(np.float32), True)

        t = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t - t_prev, 1e-6))
        t_prev = t
        _, n_photos = next_filename(args.folder)

        status = "CHESSBOARD OK - press SPACE" if corners is not None else "Whole chessboard not visible"
        color = (0, 200, 0) if corners is not None else (0, 0, 255)
        cv2.putText(view, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(view, f"Photos: {n_photos}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(view, status, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if notice and t - t_notice < 1.5:
            cv2.putText(view, notice, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow(window, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            photo_corners = detect_fast(frame)  # check the exact frame that is about to be saved
            if photo_corners is None:
                notice, t_notice = "Not saved: chessboard not fully visible or moving", t
            else:
                path, _ = next_filename(args.folder)
                cv2.imwrite(path, frame)  # original frame, without drawings
                cv2.fillConvexPoly(coverage, cv2.convexHull(photo_corners.astype(np.int32)), 255)
                notice, t_notice = f"Saved {os.path.basename(path)}", t
                print(notice)

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
