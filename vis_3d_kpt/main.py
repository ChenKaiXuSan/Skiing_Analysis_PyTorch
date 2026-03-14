#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from .visualize import run_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量可视化 fused/smoothed 3D 结果目录。"
    )
    parser.add_argument(
        "--input-dir",
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
        action="store_true",
        help="遇到单个 pair 处理失败时继续处理剩余 pairs",
    )
    parser.add_argument(
        "--center-mode",
        choices=("none", "mean", "pelvis"),
        default="none",
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


def find_pairs(
    input_dir: Path, fused_suffix: str, smoothed_suffix: str
) -> List[Tuple[str, Path, Path]]:
    """在 input_dir 下递归查找所有 fused 文件并尝试匹配对应的 smoothed 文件。

    返回 (unique_name, fused_path, smoothed_path) 列表，unique_name 用于输出子目录命名。
    """
    pairs: List[Tuple[str, Path, Path]] = []
    for fused_path in sorted(input_dir.rglob(f"*{fused_suffix}")):
        stem = fused_path.name[: -len(fused_suffix)]

        # 优先在同一目录寻找对应 smoothed 文件
        smoothed_candidate = fused_path.with_name(f"{stem}{smoothed_suffix}")
        if smoothed_candidate.exists():
            rel = fused_path.relative_to(input_dir)
            unique_name = str((rel.parent / stem).as_posix()).replace("/", "__")
            pairs.append((unique_name, fused_path, smoothed_candidate))
            continue

        # 否则尝试在整个 tree 中查找同名 smoothed
        found = list(input_dir.rglob(f"{stem}{smoothed_suffix}"))
        if found:
            rel = fused_path.relative_to(input_dir)
            unique_name = str((rel.parent / stem).as_posix()).replace("/", "__")
            pairs.append((unique_name, fused_path, found[0]))

    return pairs


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    person_infos = load_person_infos_from_dirs(args.video_dir.resolve(), args.input_dir.resolve(), args.input_dir.resolve(), args.suffix_fused, args.suffix_smoothed)

    print(f"[info] found {len(person_infos)} people in {input_dir}")

    for person_info in person_infos:
        name = person_info.left_video_path.stem if person_info.left_video_path else person_info.right_video_path.stem
        print(f"[run] {name}")

        output_dir = args.out_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        run_visualization(
            person_info=person_info,
            out_dir=output_dir,
        )
            

    print("[done] all pairs processed successfully")

def load_person_infos_from_dirs(
    video_dir: Path, kpt2d_dir: Path, kpt3d_dir: Path,
    fused_suffix: str = "_fused.npy", smoothed_suffix: str = "_smoothed.npy"
) -> List[OnePersonInfo]:
    """
    根据三个目录自动组装每个人的 OnePersonInfo。
    匹配规则：以 3D 结果（kpt3d_dir）为主，按 stem 匹配 2D/视频。
    """
    pairs = find_pairs(kpt3d_dir, fused_suffix, smoothed_suffix)
    person_infos: List[OnePersonInfo] = []
    for name, fused_path, smoothed_path in pairs:
        # 取唯一名（去掉 __ 分隔符还原路径）
        stem = name.split("__")[-1]
        left_2d = list((kpt2d_dir).rglob(f"{stem}_left_2d.npy"))
        right_2d = list((kpt2d_dir).rglob(f"{stem}_right_2d.npy"))
        left_vid = list((video_dir).rglob(f"{stem}_left.mp4"))
        right_vid = list((video_dir).rglob(f"{stem}_right.mp4"))
        person_infos.append(
            OnePersonInfo(
                left_video_path=left_vid[0] if left_vid else None,
                right_video_path=right_vid[0] if right_vid else None,
                left_2d_kpt_path=left_2d[0] if left_2d else None,
                right_2d_kpt_path=right_2d[0] if right_2d else None,
                fused_3d_kpt_path=fused_path,
                fused_smoothed_3d_kpt_path=smoothed_path,
            )
        )
    return person_infos



if __name__ == "__main__":
    main()
