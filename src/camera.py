"""Shared webcam module: open the camera, remove lens distortion and measure the FPS.

Every live part of the project (security system, wheel tracker, main program) gets
its frames through these functions, so all of them use the same camera settings
and the same calibration.

- The camera is opened like in the course's test.py: DirectShow backend, 1280x720.
- Lens distortion is removed like in Lab 2: the undistortion maps are computed once
  with initUndistortRectifyMap, and every frame only goes through remap.
- The FPS are a moving average over the last frames, so the number is readable.

Frames are NOT mirrored here. The cards must be read on the original image
(a mirrored 2 looks like a 5), so mirroring is only done when a frame is shown.

Demo, from the repository root:
    python -m src.camera
Keys: u = undistortion on/off (to compare), q / ESC = quit.
"""
import os
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.normpath(os.path.join(HERE, "..", "data", "calibration.npz"))


def open_camera(index=0, width=1280, height=720):
    """Opens the webcam and asks for the given resolution.

    Args:
        index (int): camera index (0 is the laptop's built-in webcam).
        width (int): requested frame width. The calibration was done at 1280x720.
        height (int): requested frame height.

    Returns:
        cv2.VideoCapture: the opened camera, or None if it could not be opened.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW: Windows backend, as in test.py
    if not cap.isOpened():
        print(f"Error: could not open camera {index}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def load_undistort_maps(frame_size, calibration_file=CALIBRATION_FILE):
    """Builds the undistortion maps from the calibration. Done only once.

    Args:
        frame_size (tuple): (width, height) of the camera frames.
        calibration_file (str): path to calibration.npz.

    Returns:
        tuple: (map1, map2) for cv2.remap, or None if there is no calibration
        for this frame size (then the frames are used as they come).
    """
    if not os.path.exists(calibration_file):
        print(f"Warning: {calibration_file} not found, frames will not be undistorted")
        return None
    with np.load(calibration_file) as data:
        K, dist = data["K"], data["dist"]
        calib_size = tuple(int(v) for v in data["image_size"])
    if calib_size != tuple(frame_size):
        print(f"Warning: the calibration was made at {calib_size[0]}x{calib_size[1]} but the camera "
              f"gives {frame_size[0]}x{frame_size[1]}, frames will not be undistorted")
        return None
    # K is also the new camera matrix, so the undistorted image keeps the same size and scale
    return cv2.initUndistortRectifyMap(K, dist, None, K, tuple(frame_size), cv2.CV_16SC2)


def undistort(frame, maps):
    """Removes the lens distortion of one frame. Returns the frame unchanged if maps is None."""
    if maps is None:
        return frame
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)


class FpsCounter:
    """Frames per second, as a moving average over the last `window` frames."""

    def __init__(self, window=30):
        self.times = deque(maxlen=window)

    def tick(self):
        """Call it once per frame. Returns the current FPS (0 until two frames have been seen)."""
        self.times.append(time.perf_counter())
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


def put_text(image, text, position, color=(0, 255, 255), scale=1.0):
    """Writes text with a black outline, so it can be read on any background."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_fps(image, fps):
    """Writes the FPS in the top-left corner of the image."""
    put_text(image, f"FPS: {fps:.1f}", (20, 40))


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
    print(f"Camera opened at {w}x{h}. Undistortion: {'on' if maps is not None else 'off'}")

    fps_counter = FpsCounter()
    use_undistort = maps is not None
    window_name = "Camera - u: undistortion on/off, q: quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected)")
                break
            if use_undistort:
                frame = undistort(frame, maps)
            fps = fps_counter.tick()

            view = cv2.flip(frame, 1)  # mirror only for display
            draw_fps(view, fps)
            put_text(view, "Undistorted" if use_undistort else "Original (distorted)", (20, 80), (255, 255, 255))
            cv2.imshow(window_name, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("u") and maps is not None:
                use_undistort = not use_undistort
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
