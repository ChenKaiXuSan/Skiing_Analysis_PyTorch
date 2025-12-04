import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from bundle_adjustment.run import process_one_person

logger = logging.getLogger(__name__)


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

    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subjects:
        raise FileNotFoundError(f"No subject folders under: {root}")

    logger.info(f"[{name}] Found {len(subjects)} subjects in: {root}")

    subject_map: Dict[str, List[Path]] = {}
    for subject_dir in subjects:
        files = find_files(subject_dir, patterns, recursive)
        if files:
            subject_map[subject_dir.name] = files
        else:
            logger.warning(f"[{name}] [No files] {subject_dir}")

    return subject_map


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(
    config_path="../configs",
    config_name="bundle_adjustment",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # logging 设置（Hydra 也有自己的 logging 配置，这里做一个最简单的 fallback）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(logging.INFO)

    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    # -------------------------- 读取路径 -------------------------- #
    video_root = Path(cfg.paths.video_path).resolve()
    pt_root = Path(cfg.paths.pt_path).resolve()
    vggt_root = Path(cfg.paths.vggt_path).resolve()
    videopose3d_root = Path(cfg.paths.videopose3d_path).resolve()
    out_root = Path(cfg.paths.log_path).resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    recursive = bool(cfg.dataset.get("recursive", False))

    # -------------------------- 扫描各类文件 -------------------------- #
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]
    pt_patterns = ["*.pt"]
    npz_patterns = ["*.npz"]

    # FIXME: vggt npy 和 videopose3d npy 的结构不一样，这里需要修改一下
    # video / pt / vggt / videopose3d 统一用一个工具函数构建映射
    videos_map = build_subject_map(video_root, vid_patterns, recursive, name="video")
    pts_map = build_subject_map(pt_root, pt_patterns, recursive, name="pt")
    vggt_map = build_subject_map(vggt_root, npz_patterns, recursive, name="vggt")
    videopose3d_map = build_subject_map(
        videopose3d_root, npz_patterns, recursive, name="videopose3d"
    )

    # -------------------------- 构建 multi-view 任务 -------------------------- #
    logger.info("Matching multi-view data for each subject...")

    # 如果你希望只有“video+pt+vggt+videopose3d 都存在”的 subject 才处理，
    # 就用四者交集；如果只要 video+pt 就行，可以改成前两者交集。
    subjects = sorted(
        set(videos_map.keys())
        & set(pts_map.keys())
        & set(vggt_map.keys())
        & set(videopose3d_map.keys())
    )

    if not subjects:
        raise ValueError(
            "没有任何 subject 同时包含 video / pt / vggt / videopose3d 文件"
        )

    logger.info(f"Total matched subjects (with all modalities): {len(subjects)}")

    # multi_pairs: 每个元素 = (subject_name, left_v, right_v, left_pt, right_pt)
    multi_pairs: List[Tuple[str, Path, Path, Path, Path, List[Path], List[Path]]] = []

    for subject_name in subjects:
        vids = videos_map[subject_name]
        pts = pts_map[subject_name]

        # 多视角：至少 2 个 video + 2 个 pt
        # 约定：vids[1]/pts[1] 为 left，vids[0]/pts[0] 为 right
        if len(vids) >= 2 and len(pts) >= 2:
            # 如果你希望更稳定一些，可以先按文件名排序（find_files 已经返回排序结果）
            left_v, right_v = vids[1], vids[0]
            left_pt, right_pt = pts[1], pts[0]
            vggt_files = vggt_map[subject_name]
            videopose3d_files = videopose3d_map[subject_name]

            multi_pairs.append(
                (
                    subject_name,
                    left_v,
                    right_v,
                    left_pt,
                    right_pt,
                    vggt_files,
                    videopose3d_files,
                )
            )
        else:
            logger.warning(
                f"[Skip] {subject_name}: need >= 2 videos and >= 2 pts for multi-view "
                f"(got {len(vids)} videos, {len(pts)} pts)"
            )

    logger.info(f"Total multi-view pairs: {len(multi_pairs)}")

    if not multi_pairs:
        logger.info("No valid multi-view pairs found. EXIT.")
        logger.info("==== ALL DONE ====")
        return

    # -------------------------- 顺序执行（无多线程） -------------------------- #
    ok_mv = 0
    fail_mv = 0

    for (
        subject_name,
        left_v,
        right_v,
        left_pt,
        right_pt,
        vggt_files,
        videopose3d_files,
    ) in multi_pairs:
        logger.info(
            f"[Subject: {subject_name}] Multi-view START\n"
            f"  left_v : {left_v}\n"
            f"  right_v: {right_v}\n"
            f"  left_pt: {left_pt}\n"
            f"  right_pt: {right_pt}\n"
            f"  vggt_files: {vggt_files}\n"
            f"  videopose3d_files: {videopose3d_files}"
        )
        try:
            out_dir = process_one_person(
                left_video_path=left_v,
                left_pt_path=left_pt,
                right_video_path=right_v,
                right_pt_path=right_pt,
                vggt_files=vggt_files,
                videopose3d_files=videopose3d_files,
                out_root=out_root,
                cfg=cfg,
            )
            if out_dir is None:
                fail_mv += 1
                logger.error(
                    f"[Subject: {subject_name}] Multi-view FAILED (None out_dir)"
                )
            else:
                ok_mv += 1
                logger.info(f"[Subject: {subject_name}] Multi-view OK -> {out_dir}")
        except Exception as e:
            fail_mv += 1
            logger.exception(
                f"[Subject: {subject_name}] Multi-view FAILED with exception: {e}"
            )

    logger.info(f"==== ALL DONE ====\n  Success: {ok_mv}\n  Failed : {fail_mv}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
