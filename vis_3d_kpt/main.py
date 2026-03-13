#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Tuple

from .visualize_3d_results import run_visualization

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
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--skeleton",
        choices=("auto", "mhr70", "coco17", "none"),
        default="auto",
    )
    parser.add_argument(
        "--center-mode",
        choices=("none", "mean", "pelvis"),
        default="none",
    )
    parser.add_argument(
        "--view-layout",
        choices=("simple", "multi"),
        default="simple",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行 worker 数（1 表示串行）",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="遇到单个文件出错时继续处理其他文件",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="在当前进程中 import 可视化脚本并直接调用（无需 subprocess）",
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


def run_pair(
    name: str, fused_path: Path, smoothed_path: Path, args: argparse.Namespace
) -> Tuple[str, bool, Optional[str]]:
    """运行单个 pair，返回 (name, success, errmsg_or_none)"""
    output_dir = args.out_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        vis_args = argparse.Namespace(
            before=fused_path,
            after=smoothed_path,
            before_key=None,
            after_key=None,
            out_dir=output_dir,
            fps=args.fps,
            frame_idx=args.frame_idx,
            max_frames=args.max_frames,
            skeleton=args.skeleton,
            center_mode=args.center_mode,
            video_name=None,
            view_layout=args.view_layout,
            npz=None,
            npz_key=None,
        )
        run_visualization(vis_args)
        return name, True, None
    except Exception as e:
        return name, False, str(e)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(input_dir, args.suffix_fused, args.suffix_smoothed)
    if not pairs:
        raise FileNotFoundError(f"在 {input_dir} 中没有找到 fused/smoothed 配对文件")

    print(f"[info] found {len(pairs)} pairs in {input_dir}")

    failures: List[Tuple[str, str]] = []

    if args.workers <= 1:
        for name, fused_path, smoothed_path in pairs:
            print(f"[run] {name}")
            name, ok, err = run_pair(name, fused_path, smoothed_path, args)
            if not ok:
                failures.append((name, err or "unknown"))
                print(f"[error] {name}: {err}")
                if not args.continue_on_error:
                    break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_pair, name, fused_path, smoothed_path, args): name
                for name, fused_path, smoothed_path in pairs
            }
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    nm, ok, err = fut.result()
                    if not ok:
                        failures.append((nm, err or "unknown"))
                        print(f"[error] {nm}: {err}")
                        if not args.continue_on_error:
                            break
                except Exception as e:
                    failures.append((name, str(e)))
                    print(f"[error] {name}: {e}")

    if failures:
        print(f"[done] completed with {len(failures)} failures")
        for name, errmsg in failures:
            print(f" - {name}: {errmsg}")
    else:
        print("[done] all pairs processed successfully")


if __name__ == "__main__":
    main()
