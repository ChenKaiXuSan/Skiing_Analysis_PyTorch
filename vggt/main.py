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
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig, OmegaConf

from vggt.infer import process_multi_view_video, process_single_view_video

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

    # 读取路径
    video_root = Path(cfg.paths.video_path).resolve()
    pt_root = Path(cfg.paths.pt_path).resolve()
    out_root = Path(cfg.paths.log_path).resolve()

    if not video_root.exists():
        raise FileNotFoundError(f"video_path not found: {video_root}")
    if not pt_root.exists():
        raise FileNotFoundError(f"pt_path not found: {pt_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    recursive = bool(cfg.dataset.get("recursive", False))
    subject_filter = set(cfg.dataset.get("subject_filter", []))

    # 并发线程数（可以在 cfg.runtime.num_workers 里改）
    num_workers = int(cfg.runtime.get("num_workers", 4))
    debug_mode = bool(cfg.runtime.get("debug", False)) or os.getenv("VGGT_DEBUG", "0") == "1"

    if debug_mode:
        logger.info("[Debug] debug_mode=True, 将使用单线程顺序执行，不启用多线程。")
        num_workers = 1


    # 搜索 patterns
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]
    pt_patterns = ["*.pt"]

    # ---------------------------------------------------------------------- #
    # 扫描 video_root
    # ---------------------------------------------------------------------- #
    subjects_video = sorted([p for p in video_root.iterdir() if p.is_dir()])
    if subject_filter:
        subjects_video = [p for p in subjects_video if p.name in subject_filter]

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
    # 扫描 pt_root
    # ---------------------------------------------------------------------- #
    subjects_pt = sorted([p for p in pt_root.iterdir() if p.is_dir()])
    if subject_filter:
        subjects_pt = [p for p in subjects_pt if p.name in subject_filter]

    if not subjects_pt:
        raise FileNotFoundError(f"No subject folders under: {pt_root}")

    logger.info(f"Found {len(subjects_pt)} subjects in: {pt_root}")

    # { subject_name: [pt files] }
    pts_map: Dict[str, List[Path]] = {}
    for subject_dir in subjects_pt:
        pts = find_files(subject_dir, pt_patterns, recursive)
        if pts:
            pts_map[subject_dir.name] = pts
        else:
            logger.warning(f"[No pt] {subject_dir}")

    # ---------------------------------------------------------------------- #
    # 构建 multi-view & single-view 任务
    # ---------------------------------------------------------------------- #
    multi_pairs: List[Tuple[str, Path, Path, Path, Path]] = []
    single_jobs: List[Tuple[Path, Path]] = []

    logger.info("Matching video & pt for each subject...")

    subjects = sorted(set(videos_map.keys()) & set(pts_map.keys()))
    if not subjects:
        raise ValueError("没有任何 subject 同时包含 video 与 pt 文件")

    for subject_name in subjects:
        vids = videos_map[subject_name]
        pts = pts_map[subject_name]

        # 多视角：至少 2 个 video + 2 个 pt
        if len(vids) >= 2 and len(pts) >= 2:
            multi_pairs.append((subject_name, vids[0], vids[1], pts[0], pts[1]))

            extra = list(zip(vids, pts))
            single_jobs.extend(extra)
        else:
            # 只有单视角
            logger.warning(f"[Single view only] {subject_name}")
            single_jobs.extend(zip(vids, pts))

    logger.info(f"Total matched subjects: {len(subjects)}")
    logger.info(f"Total multi-view pairs: {len(multi_pairs)}")
    logger.info(f"Total single-view jobs: {len(single_jobs)}")

    run_multi_view = bool(cfg.runtime.get("run_multi_view", True))
    run_single_view = bool(cfg.runtime.get("run_single_view", True))

    # ---------------------------------------------------------------------- #
    # 构造统一任务列表：multi + single 一起丢进线程池
    # ---------------------------------------------------------------------- #
    jobs = []

    if run_multi_view:
        for pair in multi_pairs:
            subject_name, left_v, right_v, left_pt, right_pt = pair
            jobs.append(
                (
                    "multi",
                    dict(
                        subject_name=subject_name,
                        left_v=left_v,
                        right_v=right_v,
                        left_pt=left_pt,
                        right_pt=right_pt,
                    ),
                )
            )

    if run_single_view:
        for v, p in single_jobs:
            jobs.append(("single", dict(video=v, pt=p)))

    if not jobs:
        logger.info("No jobs to run (multi_view / single_view disabled or empty).")
        logger.info("==== ALL DONE ====")
        return

    logger.info(
        f"Submitting {len(jobs)} jobs "
        f"({sum(1 for j in jobs if j[0] == 'multi')} multi-view, "
        f"{sum(1 for j in jobs if j[0] == 'single')} single-view) "
        f"with {num_workers} threads..."
    )

    ok_mv = fail_mv = ok_sv = fail_sv = 0

    def _run_job(job):
        job_type, payload = job
        if job_type == "multi":
            subject_name = payload["subject_name"]
            left_v = payload["left_v"]
            right_v = payload["right_v"]
            left_pt = payload["left_pt"]
            right_pt = payload["right_pt"]
            logger.info(f"[Subject: {subject_name}] Multi-view START")
            out_dir = process_multi_view_video(
                left_video_path=left_v,
                left_pt_path=left_pt,
                right_video_path=right_v,
                right_pt_path=right_pt,
                out_root=out_root,
                cfg=cfg,
            )
            return job_type, subject_name, left_v, out_dir
        else:  # "single"
            v = payload["video"]
            p = payload["pt"]
            logger.info(f"[Single-view] {v.name} START")
            out_dir = process_single_view_video(
                video_path=v,
                pt_path=p,
                out_root=out_root,
                cfg=cfg,
            )
            return job_type, v.name, v, out_dir

    # 统一线程池：multi + single 一起并行
    if debug_mode:
        # ---------------- 单线程顺序执行（方便下断点调试） ----------------
        for job in jobs:
            job_type, name, v_or_left, out_dir = _run_job(job)
            if job_type == "multi":
                if out_dir is None:
                    fail_mv += 1
                    logger.error(f"[Subject: {name}] Multi-view FAILED")
                else:
                    ok_mv += 1
                    logger.info(f"[Subject: {name}] Multi-view OK")
            else:
                if out_dir is None:
                    fail_sv += 1
                    logger.error(f"[Single-view] {name} FAILED")
                else:
                    ok_sv += 1
                    logger.info(f"[Single-view] {name} OK")
    else:
        # ---------------- 多线程并行执行（正常运行模式） ----------------
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_job = {ex.submit(_run_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                job_type, name, v_or_left, out_dir = future.result()
                if job_type == "multi":
                    if out_dir is None:
                        fail_mv += 1
                        logger.error(f"[Subject: {name}] Multi-view FAILED")
                    else:
                        ok_mv += 1
                        logger.info(f"[Subject: {name}] Multi-view OK")
                else:
                    if out_dir is None:
                        fail_sv += 1
                        logger.error(f"[Single-view] {name} FAILED")
                    else:
                        ok_sv += 1
                        logger.info(f"[Single-view] {name} OK")

    logger.info(
        f"== Multi-View Summary | OK: {ok_mv} | Failed: {fail_mv} | Total: {ok_mv + fail_mv} =="
    )
    logger.info(
        f"== Single-View Summary | OK: {ok_sv} | Failed: {fail_sv} | Total: {ok_sv + fail_sv} =="
    )
    logger.info("==== ALL DONE ====")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
