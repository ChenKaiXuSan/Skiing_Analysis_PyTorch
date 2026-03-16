#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

from .load import OnePersonInfo
from .visualize import run_visualization

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量可视化 fused/smoothed 3D 结果目录。"
    )
    parser.add_argument(
        "--fused-dir",
        type=Path,
        default=Path("/workspace/data/fused_smoothed_results"),
        help="包含 *_fused.npy 和 *_smoothed.npy 的目录",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/code/logs/3d_vis_batch"),
        help="批量输出目录",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("/workspace/data/dual_view_pose/side_raw_data"),
        help="视频文件所在目录，程序会尝试根据 3D 结果文件名匹配对应视频（如 xxx_fused.npy 会匹配 xxx_left.mp4 和 xxx_right.mp4）",
    )
    parser.add_argument(
        "--sam3d-results-dir",
        type=Path,
        default=Path("/workspace/data/dual_view_pose/sam3d_body_results/person"),
        help="包含 sam3d 结果的目录，程序会尝试根据 3D 结果文件名匹配对应 sam3d 结果（如 xxx_fused.npy 会匹配 xxx_sam3d.npy）",
    )
    parser.add_argument(
        "--suffix-fused",
        type=str,
        default="_fused.npy",
        help="原始融合结果文件后缀",
    )
    parser.add_argument(
        "--suffix-smoothed",
        type=str,
        default="_smoothed.npy",
        help="平滑结果文件后缀",
    )
    return parser.parse_args()


def find_fused_path(
    input_dir: Path, fused_suffix: str, smoothed_suffix: str
) -> dict[str, Tuple[Path, Path]]:
    """在 input_dir 下递归查找所有 fused 文件并尝试匹配对应的 smoothed 文件。

    返回 (unique_name, fused_path, smoothed_path) 列表，unique_name 用于输出子目录命名。
    """
    pairs: dict[str, Tuple[Path, Path]] = {}
    for fused_path in sorted(input_dir.rglob(f"*{fused_suffix}")):
        stem = fused_path.name[: -len(fused_suffix)]

        # 优先在同一目录寻找对应 smoothed 文件
        smoothed_candidate = fused_path.with_name(f"{stem}{smoothed_suffix}")
        if smoothed_candidate.exists():
            rel = fused_path.relative_to(input_dir)
            unique_name = str((rel.parent / stem).as_posix()).replace("/", "__")
            pairs[unique_name] = (fused_path, smoothed_candidate)
            continue

        # 否则尝试在整个 tree 中查找同名 smoothed
        found = list(input_dir.rglob(f"{stem}{smoothed_suffix}"))
        if found:
            rel = fused_path.relative_to(input_dir)
            unique_name = str((rel.parent / stem).as_posix()).replace("/", "__")
            pairs[unique_name] = (fused_path, found[0])

    return pairs


def find_sam3d_results_dir(sam3d_results_dir: Path) -> dict[str, dict[str, Path]]:
    """在 sam3d_results_dir 下查找包含 sam3d 结果的目录（根据文件名特征判断）。
    osmo_2 -> left, osmo_1 -> right
    """

    if not sam3d_results_dir.exists():
        return None

    person_dict: dict[str, dict[str, Path]] = {}

    for person in sam3d_results_dir.iterdir():
        person_name = person.stem

        person_dict[person_name] = {"left": None, "right": None}
        for subdir in person.iterdir():
            if "left" in subdir.stem or "osmo_2" in subdir.stem:
                person_dict[person_name]["left"] = subdir
            elif "right" in subdir.stem or "osmo_1" in subdir.stem:
                person_dict[person_name]["right"] = subdir
            else:
                raise ValueError(
                    f"无法根据文件名判断 SAM3D 结果视角（left/right），请检查文件名: {subdir}"
                )

    return person_dict


def find_video_dir(video_dir: Path) -> dict[str, dict[str, Path]]:
    """在 video_dir 下查找包含视频文件的目录（根据文件名特征判断）。
    osmo_2 -> left, osmo_1 -> right
    """
    if not video_dir.exists():
        return None

    res_dict = {}

    for person in video_dir.iterdir():
        if not person.is_dir():
            continue
        for subdir in person.iterdir():
            person_name = person.name
            if person_name not in res_dict:
                res_dict[person_name] = {"left": None, "right": None}
            if "left" in subdir.stem or "osmo_2" in subdir.stem:
                res_dict[person_name]["left"] = subdir
            elif "right" in subdir.stem or "osmo_1" in subdir.stem:
                res_dict[person_name]["right"] = subdir
            else:
                raise ValueError(
                    f"无法根据文件名判断视频视角（left/right），请检查文件名: {subdir}"
                )

    return res_dict


def load_person_infos_from_dirs(
    video_dir: Path,
    sam3d_results_dir: Path,
    fused_dir: Path,
    fused_suffix: str = "_fused.npy",
    smoothed_suffix: str = "_smoothed.npy",
) -> dict[str, OnePersonInfo]:
    """
    根据三个目录自动组装每个人的 OnePersonInfo。
    匹配规则：以 3D 结果（fused_dir）为主，按 stem 匹配 2D/视频。
    """
    fused_path = find_fused_path(fused_dir, fused_suffix, smoothed_suffix)
    sam3d_results_dir = find_sam3d_results_dir(sam3d_results_dir)
    video_dir = find_video_dir(video_dir)

    # find set person name
    person_names = set(sam3d_results_dir.keys() if sam3d_results_dir else []) | set(
        video_dir.keys() if video_dir else []
    )

    person_infos: dict[str, OnePersonInfo] = {}
    for person_name in sorted(person_names):
        person_infos[person_name] = OnePersonInfo(
            person_name=person_name,
            left_video_path=video_dir[person_name]["left"],
            right_video_path=video_dir[person_name]["right"],
            left_2d_kpt_path=sam3d_results_dir[person_name]["left"],
            right_2d_kpt_path=sam3d_results_dir[person_name]["right"],
            fused_3d_kpt_path=fused_path[person_name][0],
            fused_smoothed_3d_kpt_path=fused_path[person_name][1],
        )
    return person_infos


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    person_infos = load_person_infos_from_dirs(
        video_dir=args.video_dir.resolve(),
        sam3d_results_dir=args.sam3d_results_dir.resolve(),
        fused_dir=args.fused_dir.resolve(),
        fused_suffix=args.suffix_fused,
        smoothed_suffix=args.suffix_smoothed,
    )

    for person_name, person_info in person_infos.items():
        if person_name == "pro_1":
            continue

        logger.info(f"Processing person: {person_name}")

        output_dir = args.out_dir / person_name
        output_dir.mkdir(parents=True, exist_ok=True)

        run_visualization(
            person_info=person_info,
            out_dir=output_dir,
        )

    logger.info("[done] all pairs processed successfully")
