import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from .infer import process_one_video

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
@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    # logging 设置
    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    # 读取路径
    if cfg.infer.type == "video":
        video_root = Path(cfg.paths.video_path).resolve()
        pt_root = Path(cfg.paths.pt_path).resolve()
        out_root = Path(cfg.paths.log_path).resolve()
        inference_output_path = Path(cfg.paths.result_output_path).resolve()
        if not video_root.exists():
            raise FileNotFoundError(f"video_path not found: {video_root}")
        if not pt_root.exists():
            pt_root = None
    elif cfg.infer.type == "unity":
        video_root = Path(cfg.paths.unity.video_path).resolve()
        pt_root = None
        out_root = Path(cfg.paths.log_path).resolve()
        inference_output_path = Path(cfg.paths.result_output_path).resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    inference_output_path.mkdir(parents=True, exist_ok=True)

    recursive = bool(cfg.dataset.get("recursive", False))

    # 搜索 patterns
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]
    pt_patterns = ["*.pt"]

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
    # 扫描 pt_root
    # ---------------------------------------------------------------------- #
    if pt_root is not None:
        subjects_pt = sorted([p for p in pt_root.iterdir() if p.is_dir()])
        if not subjects_pt:
            raise FileNotFoundError(f"No subject folders under: {pt_root}")
        logger.info(f"Found {len(subjects_pt)} subjects in: {pt_root}")

        subjects_pt = [p.name for p in subjects_pt]
        for subject_name in subjects_pt:
            pt_files = find_files(pt_root / subject_name, pt_patterns, recursive)
            if not pt_files:
                logger.warning(f"[No pt] {subject_name} in {pt_root / subject_name}")
    else:
        if cfg.infer.type=="unity":
            subjects_pt = ["male"]
        elif cfg.infer.type=="video":
            subjects_pt = [k for k in videos_map.keys()]
            

    # ---------------------------------------------------------------------- #
    # 构建 multi-view 任务（只保留多视角）
    # ---------------------------------------------------------------------- #
    _pairs: List[Tuple[str, Path, Path]] = []

    logger.info("Matching video & pt for each subject (multi-view only)...")

    subjects = sorted(set(videos_map.keys()) & set(subjects_pt))
    if not subjects:
        raise ValueError("没有任何 subject 同时包含 video 与 pt 文件")

    for subject_name in subjects:
        vids = videos_map[subject_name]
        if pt_root is not None:
            pts = sorted(
                [p for p in (pt_root / subject_name).iterdir() if p.suffix == ".pt"]
            )
        else:
            pts = []

        if len(pts) != 0:
            pts_by_stem = {p.stem: p for p in pts}
            for vid in vids:
                pt = pts_by_stem.get(vid.stem)
                if vid.stem == "osmo_1":
                    _pairs.append(("right", subject_name, vid, pt))
                elif vid.stem == "osmo_2":
                    _pairs.append(("left", subject_name, vid, pt))
        else:
            for vid in vids:
                if "left" in vid.stem:
                    _pairs.append(("left", subject_name, vid, None))
                elif "right" in vid.stem:
                    _pairs.append(("right", subject_name, vid, None))

    logger.info(f"Total matched subjects: {len(subjects)}")

    # ---------------------------------------------------------------------- #
    # 顺序执行（无多线程）
    # ---------------------------------------------------------------------- #
    for flag, subject_name, vid, pt in _pairs:
        logger.info(f"{flag} {subject_name} START")

        out_dir = process_one_video(
            video_path=vid,
            pt_path=pt,
            out_dir=out_root / subject_name / flag,
            inference_output_path=inference_output_path / subject_name / flag,
            cfg=cfg,
        )

    logger.info("==== ALL DONE ====")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
