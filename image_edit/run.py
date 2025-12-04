#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
简单命令行版本：读取文件，调用 Qwen-Image-Edit + multiple-angles LoRA 做“镜头变换”，
并把结果保存到指定位置。

用法示例：
    python run_camera_edit.py \
        --input input.jpg \
        --output output.jpg \
        --rotate_deg 45 \
        --move_forward 5 \
        --vertical_tilt 0 \
        --wideangle

也可以：
    python run_camera_edit.py \
        --input input.jpg \
        --output output.jpg
    # 不设任何控制，相当于原图（因为 prompt = "no camera movement" 时直接返回原图）
"""

import logging
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import torch
from torchvision.io import read_video

from image_edit.qwen_image_edit import CameraEditor
logger = logging.getLogger(__name__)


def process_one_video(
    video_path: Path,
    out_dir: Path,
    flag: str,
    cfg: DictConfig,
):
    """处理单个视频文件的镜头编辑。"""

    subject = video_path.parent.name or "default"

    out_dir = out_dir / subject / flag
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = read_video(video_path.as_posix(), pts_unit="sec", output_format="THWC")[0]

    pipe = CameraEditor(cfg=cfg)

    for idx in tqdm(range(0, frames.shape[0]), desc="Processing frames"):
        if idx > 5:
            break
        for rotate_deg in [-90, -45, 0, 45, 90]:
            result_img, used_seed, prompt = pipe.infer_camera_edit(
                image=frames[idx],
                rotate_deg=rotate_deg,
                move_forward=0.0,
                vertical_tilt=0.0,
                wideangle=False,
            )

            out_path = out_dir / f"frame_{idx:04d}" / f"edited_{rotate_deg}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result_img.save(out_path)

            logger.info(f"[Saved] {out_path}")
            logger.info(f"[Used seed] {used_seed}")
            logger.info(f"[Prompt] {prompt}")

    # final
    torch.cuda.empty_cache()
    del pipe
    logger.info(f"Finished processing {video_path}")

    return out_dir
