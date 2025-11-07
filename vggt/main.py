#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Batch process: extract frames from each video under subjects and run VGGT reconstruction.

Author: Kaixu Chen
Last Modified: 2025-11-06
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# 如果 vggt 是本仓库内模块，请确保可 import 到
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 你之前实现的可调用 API
from vggt.infer_video import reconstruct_from_video

logger = logging.getLogger("vggt.batch")


def find_videos(
    subject_dir: Path,
    patterns: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """在 subject_dir 下按模式查找视频文件。"""
    if not patterns:
        patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]
    videos: List[Path] = []
    if recursive:
        for pat in patterns:
            videos.extend(subject_dir.rglob(pat))
    else:
        for pat in patterns:
            videos.extend(subject_dir.glob(pat))
    # 去重 + 排序
    vids = sorted({v.resolve() for v in videos})
    return vids


def should_skip(out_dir: Path, skip_if_exists: bool) -> bool:
    """已存在 predictions/glb 就跳过。"""
    if not skip_if_exists:
        return False
    pred = out_dir / "predictions.npz"
    glb = next(out_dir.glob("scene_*.glb"), None)
    return pred.exists() and glb is not None


def process_one_video(
    video_path: Path,
    out_root: Path,
    cfg: DictConfig,
) -> Optional[Path]:
    """
    处理单个视频。返回输出目录；失败返回 None。
    输出目录结构：out_root/<subject>/<video_stem>/
    """
    # subject = 上一级文件夹名；若没有父级就用 "default"
    subject = video_path.parent.name or "default"
    video_stem = video_path.stem
    out_dir = out_root / subject / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if should_skip(out_dir, cfg.runtime.skip_if_exists):
        logger.info(f"[Skip] {video_path.name} 结果已存在，跳过。")
        return out_dir

    # 读取推理相关配置（提供默认值，避免 KeyError）
    infer = cfg.get("infer", {})
    mode = infer.get("mode", "uniform")  # "uniform" | "every_k" | "fps"
    fps = float(infer.get("fps", 1.0))
    every_k = infer.get("every_k", None)
    uniform_frames = infer.get("uniform_frames", 60)
    max_frames = infer.get("max_frames", None)
    max_long_edge = infer.get("max_long_edge", 1024)
    conf_thres = float(infer.get("conf_thres", 50.0))
    prediction_mode = infer.get("prediction_mode", "Depthmap and Camera Branch")
    keep_frames = bool(infer.get("keep_frames", True))
    export_ply = bool(infer.get("export_ply", False))
    voxel_size = float(infer.get("voxel_size", 0.0))
    random_sample = infer.get("random_sample", None)
    verbose = bool(cfg.runtime.get("verbose", True))

    logger.info(f"[Run] {video_path} → {out_dir} | mode={mode}")

    try:
        result = reconstruct_from_video(
            video_path=str(video_path),
            outdir=str(out_dir),
            mode=mode,
            fps=fps,
            every_k=every_k,
            uniform_frames=uniform_frames,
            max_frames=max_frames,
            max_long_edge=max_long_edge,
            conf_thres=conf_thres,
            voxel_size=voxel_size,
            random_sample=random_sample,
            export_ply=export_ply,
            prediction_mode=prediction_mode,
            keep_frames=keep_frames,
            verbose=verbose,
        )
        logger.info(
            f"[OK] {video_path.name} | frames={result['n_frames']} | "
            f"npz={Path(result['npz_path']).name} | glb={Path(result['glb_path']).name} | "
            f"time={result['time']:.2f}s"
        )
        return out_dir
    except Exception as e:
        logger.exception(f"[Failed] {video_path} | {e}")
        return None


@hydra.main(config_path="../configs", config_name="vggt", version_base=None)
def main(cfg: DictConfig):
    """
    期望的配置（vggt.yaml）结构示例：

    paths:
      video_path: /data/videos              # 根目录，下面每个子文件夹视为一个 subject
      log_path:   /data/outputs/vggt_logs   # 输出根目录

    dataset:
      recursive: false                      # 是否递归进入子子目录找视频
      patterns: ["*.mp4","*.mov"]           # 允许匹配的后缀
      subject_filter: []                    # 只处理特定 subject 文件夹名（留空表示全量）

    infer:
      mode: "uniform"                       # "uniform" | "every_k" | "fps"
      fps: 1.0
      every_k: null
      uniform_frames: 60
      max_frames: null
      max_long_edge: 1024
      conf_thres: 50.0
      prediction_mode: "Depthmap and Camera Branch"
      keep_frames: true
      export_ply: false
      voxel_size: 0.0
      random_sample: null

    runtime:
      skip_if_exists: true                  # 已有结果则跳过
      verbose: true
      dry_run: false                        # 仅列出将要处理的文件
    """
    # 清晰打印配置（Hydra 会把工作目录切换到 .hydra/ 旁的运行目录）
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    video_root = Path(cfg.paths.video_path).resolve()
    out_root = Path(cfg.paths.log_path).resolve()
    if not video_root.exists():
        raise FileNotFoundError(f"video_path not found: {video_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) 列出 subjects
    subjects = sorted([p for p in video_root.iterdir() if p.is_dir()])
    if cfg.dataset.get("subject_filter"):
        allow = set(cfg.dataset.subject_filter)
        subjects = [s for s in subjects if s.name in allow]
    if not subjects:
        raise FileNotFoundError(f"No subject folders under: {video_root}")

    logger.info(f"Found {len(subjects)} subjects in {video_root}")

    # 2) 遍历 subject → 视频
    total_videos = 0
    todo: List[Path] = []
    for subject_dir in subjects:
        vids = find_videos(
            subject_dir,
            patterns=cfg.dataset.get("patterns"),
            recursive=bool(cfg.dataset.get("recursive", False)),
        )
        if not vids:
            logger.info(f"[No video] {subject_dir}")
            continue
        todo.extend(vids)
        total_videos += len(vids)
    logger.info(f"Total videos to consider: {total_videos}")

    if bool(cfg.runtime.get("dry_run", False)):
        for v in todo:
            logger.info(f"[DRY] {v}")
        logger.info("Dry-run finished. No processing executed.")
        return

    # 3) 逐视频处理（如需并行，可用 joblib/multiprocessing 改造这里）
    ok, fail = 0, 0
    for v in todo:
        out_dir = process_one_video(v, out_root, cfg)
        if out_dir is None:
            fail += 1
        else:
            ok += 1

    logger.info(f"== Done | OK: {ok} | Failed: {fail} | Total: {ok+fail} ==")


if __name__ == "__main__":
    # 避免 Hydra 把栈追踪截断
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()