#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torchvision.io import read_video


def _load_pt(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    print(f"[INFO] Loading: {file_path}")
    return torch.load(file_path, map_location="cpu")


def _get_frames_and_shape(
    data: Dict[str, Any],
    *,
    want_frames: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[int], Optional[int]]:
    """
    返回 (frames, H, W)。若 want_frames=False，只返回 H/W（优先从 frames 或 img_shape 获取）。
    查找顺序：
      1) data["frames"]
      2) data["frames_path"]
      3) data["img_shape"]  # (H, W)
      4) data["video_path"] -> read_video
    """
    frames = None
    H = W = None

    # 1) 直接在 pt 里
    if "frames" in data and isinstance(data["frames"], torch.Tensor):
        frames = data["frames"]
        if frames.dim() == 4:
            H, W = int(frames.shape[1]), int(frames.shape[2])

    # 2) 外置路径
    if frames is None and isinstance(data.get("frames_path"), str):
        try:
            frames = torch.load(data["frames_path"], map_location="cpu")
            if frames.dim() == 4:
                H, W = int(frames.shape[1]), int(frames.shape[2])
        except Exception as e:
            print(f"[WARN] Failed to load frames_path: {data['frames_path']} ({e})")

    # 3) 元数据里的 img_shape
    if (H is None or W is None) and isinstance(data.get("img_shape"), (tuple, list)):
        try:
            H, W = int(data["img_shape"][0]), int(data["img_shape"][1])
        except Exception:
            pass

    # 4) 退回视频路径解码（仅在需要帧或还没有 H/W 时）
    if (want_frames and frames is None) or (H is None or W is None):
        vp = data.get("video_path", None)
        if (
            want_frames
            and frames is None
            and isinstance(vp, str)
            and os.path.exists(vp)
        ):
            frames = read_video(vp, pts_unit="sec", output_format="THWC")[0]
            H, W = int(frames.shape[1]), int(frames.shape[2])
        elif (H is None or W is None) and isinstance(vp, str) and os.path.exists(vp):
            # 只需要 H/W 也可以读，但这里避免再次解码整视频；保持 None
            pass

    return frames, H, W


def _to_numpy_xy(
    kpts: torch.Tensor,
) -> np.ndarray:
    """
    接受 (T,K,2|3|>=2) 的张量，返回 numpy (T,K,2) 的 x,y。
    """
    if not isinstance(kpts, torch.Tensor):
        kpts = torch.as_tensor(kpts)
    if kpts.dim() < 3 or kpts.shape[-1] < 2:
        raise ValueError(
            f"Invalid keypoints shape: {tuple(kpts.shape)} (expect (T,K,≥2))"
        )
    xy = kpts[..., :2].cpu().numpy()
    return xy


def _maybe_denorm_xy(
    xy: np.ndarray,
    H: Optional[int],
    W: Optional[int],
    assume_normalized: Optional[bool] = None,
) -> np.ndarray:
    """
    若认为 xy 是归一化（[0,1]），且提供了 H/W，则转为像素坐标。
    assume_normalized:
      - True: 强制按归一化处理
      - False: 强制当像素处理
      - None: 自动判断（max<=1.5 视为归一化）
    """
    if H is None or W is None:
        return xy  # 无法反归一化，原样返回

    if assume_normalized is None:
        m = float(np.nanmax(xy)) if xy.size else 0.0
        is_norm = m <= 1.5  # 宽松判断
    else:
        is_norm = assume_normalized

    if is_norm:
        xy = xy.copy()
        xy[..., 0] *= W
        xy[..., 1] *= H
    return xy


def _extract_scores(
    data_block: Dict[str, Any],
    fallback_kpts: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    优先使用 data_block['keypoints_score'] -> (T,K)。
    若不存在且 fallback_kpts[...,2] 存在，则取其为分数；否则全 1。
    """
    ks = data_block.get("keypoints_score", None)
    if isinstance(ks, torch.Tensor):
        return ks.cpu().numpy()
    if isinstance(ks, np.ndarray):
        return ks

    if (
        isinstance(fallback_kpts, torch.Tensor)
        and fallback_kpts.dim() >= 3
        and fallback_kpts.shape[-1] >= 3
    ):
        return fallback_kpts[..., 2].cpu().numpy()

    # 默认全 1
    if isinstance(fallback_kpts, torch.Tensor):
        T, K = int(fallback_kpts.shape[0]), int(fallback_kpts.shape[1])
    else:
        # 不知道 T,K 时返回空；上层可据此决定
        return np.array([])
    return np.ones((T, K), dtype=np.float32)


# ------------------ 对外 API：分别加载 YOLO / Detectron2 ------------------ #


def load_keypoints_from_yolo_pt(
    file_path: str,
    *,
    return_frames: bool = False,
    assume_normalized: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    """
    从 pt_info 中读取 YOLO 关键点，返回 (keypoints_xy, keypoints_score, frames)。
      - keypoints_xy: (T,K,2) 的 numpy，若可判定为归一化且提供 H/W，则已转像素坐标
      - keypoints_score: (T,K) 的 numpy；若 pt 内没有，则尝试从 kpts[...,2]；再不行则全 1
      - frames: 若 return_frames=True 则返回 (T,H,W,C) 的 torch.uint8，否则 None
    """
    data = _load_pt(file_path)
    if "YOLO" not in data or "keypoints" not in data["YOLO"]:
        raise KeyError(f"pt file missing YOLO.keypoints: {file_path}")

    frames, H, W = _get_frames_and_shape(data, want_frames=return_frames)
    kpts_t: torch.Tensor = data["YOLO"]["keypoints"]
    xy = _to_numpy_xy(kpts_t)
    xy = _maybe_denorm_xy(xy, H, W, assume_normalized=assume_normalized)

    scores = _extract_scores(data["YOLO"], fallback_kpts=kpts_t)

    # 基本形状检查
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError(f"Invalid YOLO keypoints shape after processing: {xy.shape}")
    if scores.size and (
        scores.shape[0] != xy.shape[0] or scores.shape[1] != xy.shape[1]
    ):
        raise ValueError(
            f"YOLO keypoints_score shape {scores.shape} mismatches keypoints {xy.shape}"
        )

    return xy, scores, frames


def load_keypoints_from_d2_pt(
    file_path: str,
    *,
    return_frames: bool = False,
    assume_normalized: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    """
    从 pt_info 中读取 Detectron2 关键点，返回 (keypoints_xy, keypoints_score, frames)。
      - keypoints_xy: (T,K,2) 的 numpy，若可判定为归一化且提供 H/W，则已转像素坐标
      - keypoints_score: (T,K) 的 numpy；若 pt 内没有，则尝试从 kpts[...,2]；再不行则全 1
      - frames: 若 return_frames=True 则返回 (T,H,W,C) 的 torch.uint8，否则 None
    """
    data = _load_pt(file_path)
    if "detectron2" not in data or "keypoints" not in data["detectron2"]:
        raise KeyError(f"pt file missing detectron2.keypoints: {file_path}")

    frames, H, W = _get_frames_and_shape(data, want_frames=return_frames)
    kpts_t: torch.Tensor = data["detectron2"]["keypoints"]
    xy = _to_numpy_xy(kpts_t)
    xy = _maybe_denorm_xy(xy, H, W, assume_normalized=assume_normalized)

    scores = _extract_scores(data["detectron2"], fallback_kpts=kpts_t)

    # 基本形状检查
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError(f"Invalid D2 keypoints shape after processing: {xy.shape}")
    if scores.size and (
        scores.shape[0] != xy.shape[0] or scores.shape[1] != xy.shape[1]
    ):
        raise ValueError(
            f"D2 keypoints_score shape {scores.shape} mismatches keypoints {xy.shape}"
        )

    return xy, scores, frames
