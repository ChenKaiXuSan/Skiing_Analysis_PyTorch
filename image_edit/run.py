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

from pathlib import Path

from image_edit.qwen_image_edit import CameraEditor

from omegaconf import DictConfig, OmegaConf
import logging

from tqdm import tqdm
from image_edit.load import load_info

logger = logging.getLogger(__name__)


def process_one_video(
    video_path: Path,
    pt_path: Path,
    out_dir: Path,
    inference_output_path: Path,
    cfg: DictConfig,
):
    """处理单个视频文件的镜头编辑。"""

    subject = video_path.parent.name or "default"

    out_dir = out_dir / "multi_view" / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Run-MV] {video_path} & {pt_path} → {out_dir} | ")

    # * load info from pt and video
    *_, frames = load_info(
        video_file_path=video_path.as_posix(),
        pt_file_path=pt_path.as_posix(),
        assume_normalized=False,
    )

    pipe = CameraEditor()

    for idx in tqdm(range(0, len(frames)), desc="Processing frames"):
        for rotate_deg in [-90, -45, 0, 45, 90]:
            # if idx > 10:
            #     break

            # save images
            # img_dir = out_dir / f"frame_{idx:04d}"
            # img_dir.mkdir(parents=True, exist_ok=True)

            result_img, used_seed, prompt = pipe.infer_camera_edit(
                image=frames[idx],
                rotate_deg=rotate_deg,
                move_forward=0.0,
                vertical_tilt=0.0,
                wideangle=0,
                # seed=args.seed,
                # randomize_seed=not args.no_random_seed,
                # true_guidance_scale=args.guidance,
                # num_inference_steps=args.steps,
                # height=args.height if args.height > 0 else None,
                # width=args.width if args.width > 0 else None,
            )

            out_path = out_dir / f"frame_{idx:04d}" / f"edited_{rotate_deg}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result_img.save(out_path)

            logger.info(f"[Saved] {out_path}")
            logger.info(f"[Used seed] {used_seed}")
            logger.info(f"[Prompt] {prompt}")

        return out_dir
