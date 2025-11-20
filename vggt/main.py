#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Batch process: extract frames from each video under subjects and run VGGT reconstruction.

Author: Kaixu Chen
Last Modified: 2025-11-20
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from vggt.single_view_infer import reconstruct_from_video

logger = logging.getLogger("vggt.batch")


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
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
    """如果已存在 predictions.npz 和 scene_*.glb 就跳过。"""
    if not skip_if_exists:
        return False
    pred = out_dir / "predictions.npz"
    glb = next(out_dir.glob("scene_*.glb"), None)
    return pred.exists() and glb is not None


def build_infer_kwargs(cfg: DictConfig) -> Dict:
    """
    从 cfg 中构建 reconstruct_from_video 所需的参数字典。
    统一处理默认值，避免 KeyError + 重复代码。
    """
    infer = cfg.get("infer", {})

    return dict(
        mode=infer.get("mode", "uniform"),  # "uniform" | "every_k" | "fps"
        fps=float(infer.get("fps", 1.0)),
        every_k=infer.get("every_k", None),
        uniform_frames=infer.get("uniform_frames", 60),
        max_frames=infer.get("max_frames", None),
        max_long_edge=infer.get("max_long_edge", 1024),
        conf_thres=float(infer.get("conf_thres", 50.0)),
        prediction_mode=infer.get("prediction_mode", "Depthmap and Camera Branch"),
        keep_frames=bool(infer.get("keep_frames", True)),
        export_ply=bool(infer.get("export_ply", False)),
        voxel_size=float(infer.get("voxel_size", 0.0)),
        random_sample=infer.get("random_sample", None),
        verbose=bool(cfg.runtime.get("verbose", True)),
        gpu=infer.get("gpu", 0),
    )


# --------------------------------------------------------------------------- #
# Processing functions
# --------------------------------------------------------------------------- #
def process_multi_view_video(
    left_video_path: Path,
    right_video_path: Path,
    out_root: Path,
    cfg: DictConfig,
) -> Optional[Path]:
    """
    处理双目视频。返回输出目录；失败返回 None。

    目前示例代码仍然只对 left_video_path 进行 VGGT 推理，
    主要提供一个“成对管理 + 输出目录区分”的框架。
    后续如果需要真正 multi-view 融合，可以在这里扩展。
    """
    subject = left_video_path.parent.name or "default"
    video_stem_left = left_video_path.stem
    video_stem_right = right_video_path.stem

    out_dir = out_root / subject / f"{video_stem_left}_and_{video_stem_right}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if should_skip(out_dir, cfg.runtime.skip_if_exists):
        logger.info(
            f"[Skip] {left_video_path.name} & {right_video_path.name} 结果已存在，跳过。"
        )
        return out_dir

    infer_kwargs = build_infer_kwargs(cfg)
    dry_run = bool(cfg.runtime.get("dry_run", False))

    logger.info(
        f"[Run-MV] {left_video_path} & {right_video_path} → {out_dir} | "
        f"mode={infer_kwargs['mode']}, fps={infer_kwargs['fps']}"
    )

    if dry_run:
        logger.info("[Dry-Run] 仅列出任务，不实际运行 VGGT。")
        return out_dir

    try:
        # TODO: 如果将来需要真正双目融合，可在这里扩展传参方式
        result = reconstruct_from_video(
            video_path=str(left_video_path),
            outdir=str(out_dir),
            **infer_kwargs,
        )

        logger.info(
            f"[OK-MV] {left_video_path.name} & {right_video_path.name} | "
            f"frames={result.get('n_frames', 'NA')} | "
            f"npz={Path(result['npz_path']).name if 'npz_path' in result else 'NA'} | "
            f"glb={Path(result['glb_path']).name if 'glb_path' in result else 'NA'} | "
            f"time={result.get('time', 0.0):.2f}s"
        )
        return out_dir
    except Exception as e:
        logger.exception(f"[Failed-MV] {left_video_path} & {right_video_path} | {e}")
        return None


def process_single_view_video(
    video_path: Path,
    out_root: Path,
    cfg: DictConfig,
) -> Optional[Path]:
    """
    处理单个视频。返回输出目录；失败返回 None。
    输出目录结构：out_root/<subject>/<video_stem>/
    """
    subject = video_path.parent.name or "default"
    video_stem = video_path.stem
    out_dir = out_root / subject / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if should_skip(out_dir, cfg.runtime.skip_if_exists):
        logger.info(f"[Skip] {video_path.name} 结果已存在，跳过。")
        return out_dir

    infer_kwargs = build_infer_kwargs(cfg)
    dry_run = bool(cfg.runtime.get("dry_run", False))

    logger.info(
        f"[Run-SV] {video_path} → {out_dir} | "
        f"mode={infer_kwargs['mode']}, fps={infer_kwargs['fps']}"
    )

    if dry_run:
        logger.info("[Dry-Run] 仅列出任务，不实际运行 VGGT。")
        return out_dir

    try:
        result = reconstruct_from_video(
            video_path=str(video_path),
            outdir=str(out_dir),
            **infer_kwargs,
        )

        logger.info(
            f"[OK-SV] {video_path.name} | "
            f"frames={result.get('n_frames', 'NA')} | "
            f"npz={Path(result['npz_path']).name if 'npz_path' in result else 'NA'} | "
            f"glb={Path(result['glb_path']).name if 'glb_path' in result else 'NA'} | "
            f"time={result.get('time', 0.0):.2f}s"
        )
        return out_dir
    except Exception as e:
        logger.exception(f"[Failed-SV] {video_path} | {e}")
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(config_path="../configs", config_name="vggt", version_base=None)
def main(cfg: DictConfig) -> None:
    # logging 设置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(logging.INFO)

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

    recursive = bool(cfg.dataset.get("recursive", False))
    patterns = cfg.dataset.get("patterns")

    # multi-view pair: (subject_name, left_path, right_path)
    multi_pairs: List[Tuple[str, Path, Path]] = []
    # 所有视频（用于单视角处理，可根据 config 决定是否启用）
    all_videos: List[Path] = []

    for subject_dir in subjects:
        vids = find_videos(subject_dir, patterns=patterns, recursive=recursive)
        if not vids:
            logger.info(f"[No video] {subject_dir}")
            continue

        vids = sorted(vids)
        all_videos.extend(vids)

        if len(vids) >= 2:
            # 这里简单取前两个作为左右视角，如有命名规范可在此改成更智能的匹配
            multi_pairs.append((subject_dir.name, vids[0], vids[1]))
        elif len(vids) == 1:
            logger.warning(
                f"[Single video] {subject_dir} 仅发现 1 个视频，将只参与单视角处理。"
            )

    run_multi_view = bool(cfg.runtime.get("run_multi_view", True))
    run_single_view = bool(cfg.runtime.get("run_single_view", True))

    # 2) 处理多视角
    if run_multi_view and multi_pairs:
        logger.info(f"Total {len(multi_pairs)} multi-view video pairs to process.")
        ok_mv, fail_mv = 0, 0
        for subject_name, left_v, right_v in multi_pairs:
            out_dir = process_multi_view_video(left_v, right_v, out_root, cfg)
            if out_dir is None:
                fail_mv += 1
            else:
                ok_mv += 1
        logger.info(
            f"== Multi-View Done | OK: {ok_mv} | Failed: {fail_mv} "
            f"| Total: {ok_mv + fail_mv} =="
        )
    else:
        logger.info("Skip multi-view processing (run_multi_view=False or no pairs).")

    # 3) 处理单视角
    if run_single_view and all_videos:
        logger.info(f"Total {len(all_videos)} single-view videos to process.")
        ok_sv, fail_sv = 0, 0
        for video_path in all_videos:
            out_dir = process_single_view_video(video_path, out_root, cfg)
            if out_dir is None:
                fail_sv += 1
            else:
                ok_sv += 1

        logger.info(
            f"== Single-View Done | OK: {ok_sv} | Failed: {fail_sv} "
            f"| Total: {ok_sv + fail_sv} =="
        )
    else:
        logger.info("Skip single-view processing (run_single_view=False or no videos).")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
