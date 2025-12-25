#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/front/main.py
Project: /workspace/code/front
Created Date: Friday December 12th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday December 12th 2025 3:20:37 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field

from .run import process_one_person

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# データ構造を定義
# --------------------------------------------------------------------------- #
@dataclass
class SubjectData:
    """各被写体のマルチビューデータを格納するデータクラス"""

    subject_name: str
    left_video: Path
    right_video: Path
    left_pt: Path
    right_pt: Path
    left_sam3d_body: Path
    right_sam3d_body: Path
    vggt_files: List[Path] = field(default_factory=list)
    videopose3d_files: List[Path] = field(default_factory=list)
    front_sam3_results: Path = field(default=None)


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def find_files(
    subject_dir: Path,
    patterns: List[str],
    recursive: bool = False,
) -> List[Path]:
    """在 subject_dir 下按模式查找文件（视频或 pt / npz）。

    Args:
        subject_dir: 被扫描的被试目录
        patterns: e.g. ["*.mp4", "*.mov"]
        recursive: 是否递归子目录

    Returns:
        排好序的绝对路径列表（去重）
    """
    files: List[Path] = []

    if recursive:
        for pat in patterns:
            files.extend(subject_dir.rglob(pat))
    else:
        for pat in patterns:
            files.extend(subject_dir.glob(pat))

    # 用 set 去重，再统一转成绝对路径 & 排序
    return sorted({f.resolve() for f in files})


def build_subject_map(
    root: Path,
    patterns: List[str],
    recursive: bool,
    name: str,
) -> Dict[str, List[Path]]:
    """从 root 下扫描所有 subject 文件夹并构建 {subject_name: [files]} 映射。"""
    if not root.exists():
        raise FileNotFoundError(f"{name} root not found: {root}")

    subjects = sorted([p for p in root.iterdir()])
    if not subjects:
        raise FileNotFoundError(f"No subject folders under: {root}")

    logger.info(f"[{name}] Found {len(subjects)} subjects in: {root}")

    subject_map: Dict[str, List[Path]] = {}
    for subject_dir in subjects:
        if subject_dir.is_dir():
            files = find_files(subject_dir, patterns, recursive)
            if files:
                subject_map[subject_dir.name] = files
            else:
                logger.warning(f"[{name}] [No files] {subject_dir}")
        elif subject_dir.is_file():
            subject_map[subject_dir.stem] = subject_dir.resolve()

    return subject_map


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(
    config_path="../configs",
    config_name="front_side",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # logging 设置（Hydra 也有自己的 logging 配置，这里做一个最简单的 fallback）

    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    # -------------------------- パスとパターンの設定 -------------------------- #
    # データソースの設定を辞書にまとめ、繰り返し処理を可能にする
    data_sources = [
        (
            "video",
            cfg.paths.video_path,
            ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"],
        ),
        ("pt", cfg.paths.pt_path, ["*.pt"]),
        ("vggt", cfg.paths.vggt_path, ["*.npz"]),
        ("videopose3d", cfg.paths.videopose3d_path, ["*.npz"]),
        ("sam3d_body", cfg.paths.sam3d_body_path, ["*.npz"]),
        ("front_sam3_results", cfg.paths.sam3_path, ["*.npy"]),
    ]

    subject_maps: Dict[str, Dict[str, List[Path]]] = {}
    all_subject_names = set()
    recursive = bool(cfg.dataset.get("recursive", False))
    out_root = Path(cfg.paths.log_path).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # -------------------------- 扫描各类文件 -------------------------- #
    for name, path_cfg, patterns in data_sources:
        root_path = Path(path_cfg).resolve()

        # subject_map を構築
        subject_map = build_subject_map(root_path, patterns, recursive, name=name)
        subject_maps[name] = subject_map

        # 最初のデータソースで全被写体リストを初期化、それ以降は交差演算
        if not all_subject_names:
            all_subject_names.update(subject_map.keys())
        else:
            all_subject_names.intersection_update(subject_map.keys())

    logger.info("Matching multi-view data for each subject...")

    subjects = sorted(list(all_subject_names))

    if not subjects:
        # エラーメッセージを短く、かつ情報量が多くなるように改善
        raise ValueError(
            "No subjects found that contain files for ALL required modalities: "
            f"{[name for name, _, _ in data_sources]}"
        )

    logger.info(f"Total matched subjects (with all modalities): {len(subjects)}")

    # -------------------------- 构建 multi-view 任务 -------------------------- #
    logger.info("Matching multi-view data for each subject...")

    # SubjectData クラスを使用してペアを構築
    multi_pairs: List[SubjectData] = []

    for subject_name in subjects:
        # すべての subject_map からデータを取得
        vids = subject_maps["video"].get(subject_name, [])
        pts = subject_maps["pt"].get(subject_name, [])
        sam3d_body_files = subject_maps["sam3d_body"].get(subject_name, [])
        vggt_files = subject_maps["vggt"].get(subject_name, [])
        videopose3d_files = subject_maps["videopose3d"].get(subject_name, [])
        front_sam3_results = subject_maps["front_sam3_results"].get(subject_name, [])

        # 约定：vids[1]/pts[1] 为 left，vids[0]/pts[0] 为 right
        multi_pairs.append(
            SubjectData(
                subject_name=subject_name,
                left_video=vids[1],
                right_video=vids[0],
                left_pt=pts[1],
                right_pt=pts[0],
                left_sam3d_body=sam3d_body_files[1],
                right_sam3d_body=sam3d_body_files[0],
                vggt_files=vggt_files,
                videopose3d_files=videopose3d_files,
                front_sam3_results=front_sam3_results[0]
                if front_sam3_results
                else None,
            )
        )

    if not multi_pairs:
        logger.info("No valid multi-view pairs found. EXIT.")
        logger.info("==== ALL DONE ====")
        return

    # -------------------------- 顺序执行 -------------------------- #

    for pair in multi_pairs:
        logger.info(
            f"[Subject: {pair.subject_name}] Multi-view START\n"
            f"  left_v : {pair.left_video}\n"
            f"  right_v: {pair.right_video}\n"
            f"  left_pt: {pair.left_pt}\n"
            f"  right_pt: {pair.right_pt}\n"
            f"  vggt_files: {pair.vggt_files}\n"
            f"  videopose3d_files: {pair.videopose3d_files}\n"
            f"  front_sam3_results: {pair.front_sam3_results}\n"
        )

        process_one_person(
            left_sam3d_body_path=pair.left_sam3d_body,
            right_sam3d_body_path=pair.right_sam3d_body,
            front_sam3_results=pair.front_sam3_results,
            output_dir=out_root / pair.subject_name,
        )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
