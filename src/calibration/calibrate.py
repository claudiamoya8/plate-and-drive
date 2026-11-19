"""Offline camera calibration from the chessboard photos.

It is the method from Lab 2 turned into a script:
findChessboardCorners -> cornerSubPix -> 3D chessboard points -> calibrateCamera.

Reads the photos in data/chessboard_photos/ and saves:
    data/calibration.npz       K, dist, rms, image size and square size
    data/calibration.txt       the same values as text, to copy into the report
    data/calibration_results/corners/          every photo with its corners drawn
    data/calibration_results/before_after.jpg  one photo, original and undistorted, side by side
The photos and data/calibration_results/ are not uploaded to GitHub (people appear in them).

Usage, from the repository root:
    python src/calibration/calibrate.py               # 8x6 chessboard, 24.29 mm squares (measured)
    python src/calibration/calibrate.py --square 25   # if another chessboard is printed with another size
"""
import argparse
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))


def chessboard_points(pattern, square_mm):
    """3D coordinates of the corners, in mm, with Z = 0 (flat chessboard)."""
    cols, rows = pattern
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=os.path.join(DATA, "chessboard_photos"))
    ap.add_argument("--output", default=DATA, help="folder where the results are saved")
    # Measured on the printed chessboard: 170 mm between the first and last inner corner of a row (7 squares)
    ap.add_argument("--square", type=float, default=170 / 7, help="square side in mm (default 24.29)")
    ap.add_argument("--cols", type=int, default=8, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--window", type=int, default=11, help="cornerSubPix half window size")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    paths = sorted(glob.glob(os.path.join(args.folder, "*.png")) + glob.glob(os.path.join(args.folder, "*.jpg")))
    if not paths:
        print(f"No photos in {args.folder}")
        return

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj = chessboard_points(pattern, args.square)
    corners_folder = os.path.join(args.output, "calibration_results", "corners")
    os.makedirs(corners_folder, exist_ok=True)

    points_3d, points_2d, valid, size = [], [], [], None
    for path in paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print(f"  {os.path.basename(path)}: different size {gray.shape[::-1]}, skipped")
            continue
        ok, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print(f"  {os.path.basename(path)}: chessboard not found, skipped")
            continue
        corners = cv2.cornerSubPix(gray, corners, (args.window, args.window), (-1, -1), criteria)
        points_3d.append(obj)
        points_2d.append(corners)
        valid.append(path)
        drawing = cv2.drawChessboardCorners(img.copy(), pattern, corners, True)
        cv2.imwrite(os.path.join(corners_folder, os.path.basename(path)), drawing)

    print(f"\nValid photos: {len(valid)} of {len(paths)}  (image {size[0]}x{size[1]})")
    if len(valid) < 10:
        print("Warning: with fewer than 10 valid photos the calibration is unreliable. Take more.")
    if len(valid) < 3:
        return

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(points_3d, points_2d, size, None, None)

    # Reprojection error of each photo, to find the bad ones
    errors = []
    for p3, p2, rv, tv in zip(points_3d, points_2d, rvecs, tvecs):
        proj, _ = cv2.projectPoints(p3, rv, tv, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((proj - p2) ** 2, axis=2)))))

    np.set_printoptions(precision=4, suppress=True)
    print("\nIntrinsic matrix K:\n", K)
    print("Distortion coefficients [k1 k2 p1 p2 k3]:\n", dist.ravel())
    print(f"Reprojection RMS: {rms:.4f} px")

    print("\nError per photo (px):")
    threshold = max(2 * rms, 0.5)
    for path, e in sorted(zip(valid, errors), key=lambda x: -x[1]):
        mark = "  <- check: moved or blurry?" if e > threshold else ""
        print(f"  {os.path.basename(path)}: {e:.3f}{mark}")

    if rms < 0.5:
        print("\nGood calibration (RMS < 0.5 px).")
    elif rms < 1.0:
        print("\nAcceptable calibration. Removing the marked photos and running it again may improve it.")
    else:
        print("\nBad calibration (RMS >= 1 px). Delete the marked photos, take more and run it again.")

    np.savez(os.path.join(args.output, "calibration.npz"), K=K, dist=dist, rms=rms,
             image_size=np.array(size), square_mm=args.square, pattern=np.array(pattern))
    with open(os.path.join(args.output, "calibration.txt"), "w", encoding="utf-8") as f:
        f.write(f"Image: {size[0]}x{size[1]} px\n")
        f.write(f"Chessboard: {pattern[0]}x{pattern[1]} inner corners, {args.square:.2f} mm squares\n")
        f.write(f"Photos used: {len(valid)}\n\n")
        f.write(f"fx = {K[0, 0]:.2f} px   fy = {K[1, 1]:.2f} px\n")
        f.write(f"cx = {K[0, 2]:.2f} px   cy = {K[1, 2]:.2f} px\n")
        k1, k2, p1, p2, k3 = dist.ravel()[:5]
        f.write(f"k1 = {k1:.4f}   k2 = {k2:.4f}   k3 = {k3:.4f}\n")
        f.write(f"p1 = {p1:.5f}   p2 = {p2:.5f}\n\n")
        f.write(f"Reprojection RMS = {rms:.4f} px\n")

    # Visual check: one photo before and after undistortion
    img = cv2.imread(valid[0])
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_16SC2)
    undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(args.output, "calibration_results", "before_after.jpg"), np.hstack([img, undistorted]))

    print(f"\nSaved: {os.path.join(args.output, 'calibration.npz')}")
    print(f"       {os.path.join(args.output, 'calibration.txt')}")
    print(f"       {os.path.join(args.output, 'calibration_results')}")


if __name__ == "__main__":
    main()
