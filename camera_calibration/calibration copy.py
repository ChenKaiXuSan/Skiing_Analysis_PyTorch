#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Camera calibration script that auto-detects input type:
- If `input_path` is a directory -> load images.
- If `input_path` is a file -> treat as video and sample frames.

Author: Kaixu Chen <chenkaixusan@gmail.com>
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------- Utilities ----------------------
def is_video_file(path: str) -> bool:
    video_exts = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".m4v",
        ".MP4",
        ".MOV",
        ".AVI",
        ".MKV",
        ".M4V",
    }
    p = Path(path)
    return p.is_file() and (p.suffix in video_exts)


def is_image_dir(path: str) -> bool:
    p = Path(path)
    return p.is_dir()


def prepare_object_points(board_size, square_size):
    """
    board_size: (cols, rows) = inner corners! e.g. chessboard 9x6 => (9,6)
    square_size: size of one square in mm (or any length unit)
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    return objp * square_size


def find_chessboard_corners(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # flags 可以按需添加 cv2.CALIB_CB_FAST_CHECK 提升速度（略降召回）
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


# ---------------------- IO loaders ----------------------
def load_images_from_dir(
    image_dir,
    patterns=("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"),
):
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(image_dir, pat)))
    return sorted(paths)


def sample_frames_from_video(
    video_path, step=10, max_frames=None, warmup=0, vis_dir=None
):
    """
    step: sample every N frames
    max_frames: cap total sampled frames (None = unlimited)
    warmup: skip first `warmup` frames (e.g., motion blur)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    idx = -1
    kept = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx < warmup:
            continue
        if idx % step != 0:
            continue
        frames.append((idx, frame.copy()))
        kept += 1
        if max_frames is not None and kept >= max_frames:
            break
    cap.release()

    # 可选：保存抽帧预览
    if vis_dir:
        os.makedirs(os.path.join(vis_dir, "video_samples"), exist_ok=True)
        for k, (fi, fr) in enumerate(frames[:10]):  # 只存前10张预览
            cv2.imwrite(
                os.path.join(vis_dir, "video_samples", f"sample_{fi:06d}.jpg"), fr
            )
    return frames


# ---------------------- Calibration core ----------------------
def calibrate_camera(
    images_or_frames,
    board_size=(9, 6),
    square_size=25.0,
    output_file=".calibration_parameters.npz",
    vis_dir="visualization_output",
    tag_prefix="img",
):
    """
    images_or_frames: list of tuples (name, image ndarray) OR (frame_idx, image)
    """
    ouptut_dir = os.path.join(vis_dir, "video_corners")
    os.makedirs(ouptut_dir, exist_ok=True)

    objpoints, imgpoints, used_names = [], [], []
    objp = prepare_object_points(board_size, square_size)

    for i, (name, img) in enumerate(images_or_frames):
        if img is None:
            print(f"[WARN] empty image at {name}")
            continue
        corners = find_chessboard_corners(img, board_size)
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
            used_names.append(name)
            vis_path = os.path.join(ouptut_dir, f"{tag_prefix}_corners_{i+1:03d}.jpg")
            save_visualization(img, corners, board_size, vis_path)
        else:
            print(f"[INFO] Chessboard not detected: {name}")

    if not objpoints:
        print("❌ Calibration failed: No valid chessboard detections.")
        return {"status": "Failed", "reason": "No corners detected", "num_images": 0}

    # use the first valid image size
    h, w = images_or_frames[0][1].shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    # save parameters
    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        image_size=(w, h),
        used=used_names,
    )

    # undistort visualization for all used images (or subset if too many)
    limit = min(len(used_names), 30)
    for i, name in enumerate(used_names[:limit]):
        # find the already loaded image by name
        img = None
        for nm, im in images_or_frames:
            if nm == name:
                img = im
                break
        if img is not None:
            undistort_path = os.path.join(
                vis_dir, f"{tag_prefix}_undistort_{i+1:03d}.jpg"
            )
            save_undistortion_comparison(
                img, camera_matrix, dist_coeffs, undistort_path
            )

    result = {
        "status": "Success" if ret else "Failed",
        "rms_reproj_error": float(ret),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "num_images": len(used_names),
        "image_size": (w, h),
        "used_names": used_names,
    }
    return result


def calibrate_from_input(
    input_path,
    board_size=(9, 6),
    square_size=25.0,
    output_file="calibration_parameters.npz",
    vis_dir="calibration_vis",
    # video sampling params
    video_step=8,
    video_max_frames=300,
    video_warmup=0,
):
    """
    Auto-detect input:
    - Directory -> load images
    - File -> treat as video and sample frames
    """
    os.makedirs(vis_dir, exist_ok=True)

    if is_image_dir(input_path):
        img_paths = load_images_from_dir(input_path)
        if not img_paths:
            raise RuntimeError(f"No images found under: {input_path}")
        images = []
        for p in img_paths:
            im = cv2.imread(p)
            if im is None:
                print(f"[WARN] cannot read image: {p}")
                continue
            images.append((p, im))
        tag = "img"
        result = calibrate_camera(
            images_or_frames=images,
            board_size=board_size,
            square_size=square_size,
            output_file=output_file,
            vis_dir=vis_dir,
            tag_prefix=tag,
        )
        return result

    elif is_video_file(input_path):
        frames = sample_frames_from_video(
            input_path,
            step=video_step,
            max_frames=video_max_frames,
            warmup=video_warmup,
            vis_dir=vis_dir,
        )
        if not frames:
            raise RuntimeError(f"No frames sampled from video: {input_path}")
        # wrap to (name,image)
        frames_named = [(f"frame_{idx:06d}", frame) for idx, frame in frames]
        tag = "vid"
        result = calibrate_camera(
            images_or_frames=frames_named,
            board_size=board_size,
            square_size=square_size,
            output_file=output_file,
            vis_dir=vis_dir,
            tag_prefix=tag,
        )
        return result

    else:
        raise RuntimeError(
            f"Invalid input_path. Not a directory or a known video file: {input_path}"
        )


# ---------------------- Main ----------------------
if __name__ == "__main__":
    # ⚠️ board_size 是“内角点数” (cols, rows)，确保与你打印的棋盘一致
    CHESSBOARD_SIZE = (9, 6)  # e.g., 9x6 内角点就填 (9,6)
    SQUARE_SIZE = 25.0  # mm 或 cm，保持单位一致
    OUTPUT_FILE = "camera_calibration/calibration_parameters.npz"
    VIS_DIR = "logs/calibration_vis"

    # 如果你要用视频：把 INPUT_PATH 改成视频路径即可
    # INPUT_PATH = "camera_calibration/chessboard_images_9x6"  # 可填目录或视频文件
    INPUT_PATH = "/workspace/code/camera_calibration/chessboard/video/DJI_20250831173129_0014_D.MP4"

    result = calibrate_from_input(
        input_path=INPUT_PATH,
        board_size=CHESSBOARD_SIZE,
        square_size=SQUARE_SIZE,
        output_file=OUTPUT_FILE,
        vis_dir=VIS_DIR,
        video_step=1,  # 每 8 帧取一帧；可根据棋盘出现频率调整
        video_max_frames=500,  # 最多取 300 帧；避免过多重复视角
        video_warmup=0,
    )

    print("\n📌 Calibration Result Summary:")
    df = pd.DataFrame(
        [
            {
                "status": result.get("status"),
                "rms_reproj_error": result.get("rms_reproj_error"),
                "num_used_images": result.get("num_images"),
                "image_size": result.get("image_size"),
            }
        ]
    )
    print(df)
