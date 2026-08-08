#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def save_figure(image: Any, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.savefig(save_path, dpi=300, facecolor="white", edgecolor="white")
    # close matplotlib figure
    try:
        import matplotlib.pyplot as plt

        plt.close(image)
    except Exception:
        pass


def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Got None image")

    img = np.asarray(img)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported channel number: {img.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def _read_image_rgb(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img_pil:
        rgb_img = img_pil.convert("RGB")
        return np.asarray(rgb_img, dtype=np.uint8)


def merge_frame_to_video(save_path: Path, flag: str, fps: int = 30) -> Path:
    frame_dir = save_path / flag
    out_path = save_path / "video"
    out_path.mkdir(exist_ok=True, parents=True)

    frames = sorted(
        frame_dir.glob("*"),
        key=lambda x: int(x.stem.split("_")[0]),
    )
    if not frames:
        raise RuntimeError(f"No frames found in {frame_dir}")

    output_file = out_path / f"{flag}.mp4"

    first_img = _ensure_rgb_uint8(_read_image_rgb(frames[0]))
    H, W = first_img.shape[:2]

    video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc")
    video_writer_cls = getattr(cv2, "VideoWriter")
    cvt_color = getattr(cv2, "cvtColor")
    color_rgb2bgr = getattr(cv2, "COLOR_RGB2BGR")

    # cv2 编码器回退，尽量避免单一 codec 失败
    writer = None
    for codec in ["mp4v", "avc1", "XVID", "MJPG"]:
        fourcc = video_writer_fourcc(*codec)
        candidate = video_writer_cls(
            str(output_file),
            fourcc,
            float(fps),
            (W, H),
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(
            "Failed to create cv2 VideoWriter with known codecs"
        )

    writer.write(
        cvt_color(first_img, color_rgb2bgr)
    )

    for f in tqdm(frames[1:], desc=f"Save {flag}"):
        img = _ensure_rgb_uint8(_read_image_rgb(f))

        if img.shape[0] != H or img.shape[1] != W:
            img_pil = Image.fromarray(img)
            # PIL resampling compatibility
            resample_enum = getattr(Image, "Resampling", Image)
            resample = resample_enum.BILINEAR  # type: ignore[attr-defined]
            img_pil = img_pil.resize((W, H), resample)
            img = np.asarray(img_pil)

        writer.write(
            cvt_color(img, color_rgb2bgr)
        )

    writer.release()
    return output_file
