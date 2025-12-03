#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/image_edit/load.py
Project: /workspace/code/image_edit
Created Date: Wednesday December 3rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday June 6th 2025 2:41:25 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/load_fn.py
Project: /workspace/code/vggt
Created Date: Friday November 21st 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday November 21st 2025 2:42:14 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np

from typing import Optional, Tuple, Dict, Any

from torchvision.io import read_video

import logging

logger = logging.getLogger(__name__)


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        # img = Image.open(image_path)
        img = image_path.numpy()
        img = Image.fromarray(img)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (new_width / width) / 14) * 14
                )  # Make divisible by 14
            else:
                new_height = target_size
                new_width = (
                    round(width * (new_height / height) / 14) * 14
                )  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def _load_pt(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    logger.info(f"Loading: {file_path}")
    return torch.load(file_path, map_location="cpu")


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


def load_info(
    pt_file_path: str,
    video_file_path: str,
    assume_normalized: Optional[bool] = None,
    clip_bbox_to_image: bool = True,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    """
    从 Detectron2 的 pt_info 中读取关键点与 bbox。
    返回: (keypoints_xy, keypoints_score, bboxes_xyxy, bbox_scores, frames)

    - keypoints_xy: (T, K, 2)  像素坐标 (若可判定为归一化且有 H/W，则已反归一化)
    - keypoints_score: (T, K)  若无则从 kpts[...,2] 回退；再无则全 1
    - bboxes_xyxy: (T, N, 4)   每行为 [x1, y1, x2, y2]（同样会按需反归一化）
    - bbox_scores: (T, N)      若无则全 1
    - frames: (T, H, W, C) 的 torch.uint8，若 return_frames=False 则为 None
    """
    data = _load_pt(pt_file_path)
    # --------- frames / shape ----------
    # FIXME: 虽然 pt 里可能有 frames，但在prepare_dataset的时候，保存的frame是有问题的，在后面vggt的推理里面会有问题，所以选择直接从视频读取
    frames = read_video(video_file_path, pts_unit="sec", output_format="THWC")[0]

    return frames
