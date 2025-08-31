#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Camera calibration script using 9x6 chessboard images with visualizations.

Author: Kaixu Chen <chenkaixusan@gmail.com>
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd


def prepare_object_points(board_size, square_size):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    return objp * square_size


def find_chessboard_corners(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not ret:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners_refined


def save_visualization(image, corners, board_size, save_path):
    img_vis = cv2.drawChessboardCorners(image.copy(), board_size, corners, True)
    cv2.imwrite(save_path, img_vis)


def save_undistortion_comparison(original_img, camera_matrix, dist_coeffs, save_path):
    undistorted = cv2.undistort(original_img, camera_matrix, dist_coeffs)
    concat = np.hstack((original_img, undistorted))
    cv2.imwrite(save_path, concat)


def calibrate_camera_from_images(
    image_dir,
    board_size=(8, 5),
    square_size=25.0,
    output_file=".calibration_parameters.npz",
    vis_dir="visualization_output",
):
    os.makedirs(vis_dir, exist_ok=True)
    objpoints, imgpoints, valid_images = [], [], []
    objp = prepare_object_points(board_size, square_size)

    images = sorted(glob.glob(os.path.join(image_dir, "*.JPG")))
    if not images:
        raise RuntimeError(f"No .JPG images found in directory: {image_dir}")

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARNING] Failed to read image: {fname}")
            continue

        corners = find_chessboard_corners(img, board_size)
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_images.append(fname)

            vis_path = os.path.join(vis_dir, f"corners_{i+1:02d}.jpg")
            save_visualization(img, corners, board_size, vis_path)
        else:
            print(f"[INFO] Chessboard not detected in image: {os.path.basename(fname)}")

    if not objpoints:
        print("❌ Calibration failed: No valid chessboard detections.")
        return {"status": "Failed", "reason": "No corners detected", "num_images": 0}

    h, w = cv2.imread(valid_images[0]).shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    # 保存标定参数
    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        image_size=(w, h),
    )

    # 保存所有成功图像的undistortion可视化
    for i, fname in enumerate(valid_images):
        img = cv2.imread(fname)
        undistort_path = os.path.join(vis_dir, f"undistort_{i+1:02d}.jpg")
        save_undistortion_comparison(img, camera_matrix, dist_coeffs, undistort_path)

    result = {
        "status": "Success" if ret else "Failed",
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "num_images": len(valid_images),
        "image_size": (w, h),
    }
    return result


if __name__ == "__main__":
    CHESSBOARD_SIZE = (8, 5)
    SQUARE_SIZE = 25.0
    IMAGE_DIR = "camera_calibration/chessboard_images_9x6"
    VIDEO_DIR = "/workspace/code/camera_calibration/chessboard/video/DJI_20250831173129_0014_D.MP4"
    OUTPUT_FILE = "camera_calibration/calibration_parameters.npz"
    VIS_DIR = "logs/calibration_vis"

    result = calibrate_camera_from_images(
        image_dir=IMAGE_DIR,
        board_size=CHESSBOARD_SIZE,
        square_size=SQUARE_SIZE,
        output_file=OUTPUT_FILE,
        vis_dir=VIS_DIR,
    )

    print("\n📌 Calibration Result Summary:")
    print(pd.DataFrame([result]))
