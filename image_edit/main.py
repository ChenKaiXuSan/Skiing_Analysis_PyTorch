#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/image_edit/main.py
Project: /workspace/code/image_edit
Created Date: Wednesday December 3rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday December 3rd 2025 5:07:17 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Batch process: run VGGT-based multi-view reconstruction for each subject.
(Single-thread version: no multithreading)

Author: Kaixu Chen
Last Modified: 2025-11-25
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from image_edit.run import process_one_video

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def find_files(
    subject_dir: Path,
    patterns: List[str],
    recursive: bool = False,
) -> List[Path]:
    """在 subject_dir 下按模式查找文件（视频或 pt）。"""
    files: List[Path] = []
    if recursive:
        for pat in patterns:
            files.extend(subject_dir.rglob(pat))
    else:
        for pat in patterns:
            files.extend(subject_dir.glob(pat))
    return sorted({f.resolve() for f in files})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(config_path="../configs", config_name="qwen_image_edit", version_base=None)
def main(cfg: DictConfig) -> None:
    # logging 设置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(logging.INFO)

    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    # 读取路径
    video_root = Path(cfg.paths.video_path).resolve()
    out_root = Path(cfg.paths.log_path).resolve()

    if not video_root.exists():
        raise FileNotFoundError(f"video_path not found: {video_root}")
    if not out_root.exists():
        raise FileNotFoundError(f"log_path not found: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    recursive = bool(cfg.dataset.get("recursive", False))

    # 搜索 patterns
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]

    # ---------------------------------------------------------------------- #
    # 扫描 video_root
    # ---------------------------------------------------------------------- #
    subjects_video = sorted([p for p in video_root.iterdir() if p.is_dir()])

    if not subjects_video:
        raise FileNotFoundError(f"No subject folders under: {video_root}")

    logger.info(f"Found {len(subjects_video)} subjects in: {video_root}")

    # { subject_name: [video files] }
    videos_map: Dict[str, List[Path]] = {}
    for subject_dir in subjects_video:
        vids = find_files(subject_dir, vid_patterns, recursive)
        if vids:
            videos_map[subject_dir.name] = vids
        else:
            logger.warning(f"[No video] {subject_dir}")

    # ---------------------------------------------------------------------- #
    # 构建 multi-view 任务（只保留多视角）
    # ---------------------------------------------------------------------- #
    multi_pairs: List[Tuple[str, Path, Path]] = []

    logger.info("Matching video & pt for each subject (multi-view only)...")

    subjects = sorted(set(videos_map.keys()))
    if not subjects:
        raise ValueError("没有任何 subject 同时包含 video 与 pt 文件")

    for subject_name in subjects:
        vids = videos_map[subject_name]

        # 多视角：至少 2 个 video
        # 约定：vids[1] 为 left，vids[0] 为 right
        if len(vids) >= 2:
            multi_pairs.append((subject_name, vids[1], vids[0]))
        else:
            logger.warning(f"[Skip] {subject_name}: need >=2 videos for multi-view")

    logger.info(f"Total matched subjects: {len(subjects)}")
    logger.info(f"Total multi-view pairs: {len(multi_pairs)}")

    if not multi_pairs:
        logger.info("No valid multi-view pairs found. EXIT.")
        logger.info("==== ALL DONE ====")
        return

    # ---------------------------------------------------------------------- #
    # 顺序执行（无多线程）
    # ---------------------------------------------------------------------- #
    for subject_name, left_v, right_v in multi_pairs:
        logger.info(f"[Subject: {subject_name}] START")

        out_dir = process_one_video(
            video_path=left_v,
            out_dir=out_root,
            flag="left",
            cfg=cfg,
        )

        out_dir = process_one_video(
            video_path=right_v,
            out_dir=out_root,
            flag="right",
            cfg=cfg,
        )

    logger.info("==== ALL DONE ====")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
