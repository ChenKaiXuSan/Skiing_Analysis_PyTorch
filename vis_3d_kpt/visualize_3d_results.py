#!/usr/bin/env python3

import argparse
import shutil
import numpy as np
from typing import Any, Optional, Sequence, Iterable

# UNITY15 关节点ID（如有需要可自定义）
UNITY15_TARGET_IDS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62, 69]
from pathlib import Path

try:
    from numpy import lib

    NpzFile = lib.npyio.NpzFile
except ImportError:
    NpzFile = None

# COCO骨架边（如有需要可自定义）
COCO_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (0, 15),
    (14, 16),
    (15, 17),
]

UNITY15_BONE_EDGES_BY_ID = [
    (69, 5),
    (5, 7),
    (7, 62),
    (69, 6),
    (6, 8),
    (8, 41),
    (69, 9),
    (9, 11),
    (11, 13),
    (69, 10),
    (10, 12),
    (12, 14),
    (9, 10),
    (5, 6),
]

DEFAULT_VIEWS = [
    ("front_left", -25, 270),
    ("front_right", -25, 90),
    ("top", 85, 270),
    ("side", 0, 0),
]

SIMPLE_VIEWS = [
    ("perspective", -20, 255),
]

POSE_KEYS = (
    "pred_keypoints_3d",
    "keypoints_3d",
    "pose_3d",
    "positions_3d",
    "joints_3d",
)


def load_pose_sequence(
    file_path: Path,
    data_key: Optional[str] = None,
) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    loaded = np.load(file_path, allow_pickle=True)
    data = unwrap_loaded_object(loaded)
    pose = extract_pose_array(data, data_key=data_key)
    pose = np.asarray(pose, dtype=np.float64)

    if pose.ndim == 2:
        pose = pose[None, ...]
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise ValueError(f"3D 结果必须是 (T,J,3) 或 (J,3)，当前为 {pose.shape}")
    return pose


def unwrap_loaded_object(loaded: Any) -> Any:
    if isinstance(loaded, NpzFile):
        if len(loaded.files) == 1:
            return loaded[loaded.files[0]]
        return {name: loaded[name] for name in loaded.files}

    if isinstance(loaded, np.ndarray) and loaded.dtype != object:
        return loaded

    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()

    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        return loaded.tolist()

    return loaded


def extract_pose_array(
    data: Any,
    data_key: Optional[str] = None,
) -> np.ndarray:
    if isinstance(data, np.ndarray) and data.dtype != object:
        return data

    if data_key is not None:
        extracted = pick_key_from_object(data, data_key)
        return extract_pose_array(extracted)

    if isinstance(data, dict):
        if "outputs" in data:
            return stack_pose_frames(data["outputs"])
        for key in POSE_KEYS:
            if key in data:
                return extract_pose_array(data[key])
        for value in data.values():
            try:
                return extract_pose_array(value)
            except (TypeError, ValueError, KeyError):
                continue
        raise KeyError("未在 dict 中找到可用的 3D 结果")

    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError("输入序列为空")
        first = data[0]
        if isinstance(first, dict):
            return stack_pose_frames(data)
        array = np.asarray(data)
        if array.dtype != object:
            return array
        return stack_pose_frames(data)

    if isinstance(data, np.ndarray) and data.dtype == object:
        return extract_pose_array(data.tolist())

    raise TypeError(f"不支持的输入类型: {type(data)!r}")


def pick_key_from_object(data: Any, data_key: str) -> Any:
    if isinstance(data, dict):
        if data_key not in data:
            raise KeyError(f"字段 {data_key} 不存在")
        return data[data_key]

    if isinstance(data, NpzFile):
        if data_key not in data.files:
            raise KeyError(f"字段 {data_key} 不存在")
        return data[data_key]

    if hasattr(data, data_key):
        return getattr(data, data_key)

    raise KeyError(f"无法从当前对象读取字段 {data_key}")


def stack_pose_frames(frames: Sequence[Any]) -> np.ndarray:
    poses = []
    for frame in frames:
        if isinstance(frame, np.ndarray) and frame.dtype != object:
            pose = frame
        elif isinstance(frame, dict):
            pose = None
            for key in POSE_KEYS:
                if key in frame:
                    pose = frame[key]
                    break
            if pose is None:
                raise KeyError("帧字典中未找到 3D 结果字段")
        else:
            raise TypeError(f"不支持的帧类型: {type(frame)!r}")

        pose = np.asarray(pose, dtype=np.float64)
        if pose.ndim == 3 and pose.shape[0] == 1:
            pose = pose[0]
        if pose.ndim != 2 or pose.shape[-1] != 3:
            raise ValueError(f"单帧 3D 结果必须是 (J,3)，当前为 {pose.shape}")
        poses.append(pose)
    return np.stack(poses, axis=0)


def center_pose_sequence(sequence: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return sequence

    centered = sequence.copy()
    for idx, frame in enumerate(centered):
        anchor = estimate_frame_center(frame, mode)
        centered[idx] = frame - anchor
    return centered


def estimate_frame_center(frame: np.ndarray, mode: str) -> np.ndarray:
    valid = np.isfinite(frame).all(axis=1)
    if not np.any(valid):
        return np.zeros(3, dtype=np.float64)

    if mode == "mean":
        return np.nanmean(frame[valid], axis=0)

    if mode == "pelvis":
        if (
            frame.shape[0] > 10
            and np.isfinite(frame[9]).all()
            and np.isfinite(frame[10]).all()
        ):
            return 0.5 * (frame[9] + frame[10])
        return np.nanmean(frame[valid], axis=0)

    raise ValueError(f"未知 center mode: {mode}")


def get_edges(skeleton: str, num_joints: int) -> list[tuple[int, int]]:
    if skeleton == "none":
        return []
    if skeleton == "coco17":
        return [(a, b) for a, b in COCO_EDGES if a < num_joints and b < num_joints]
    if skeleton == "unity15":
        return get_unity15_edges(num_joints)
    if skeleton == "auto":
        if num_joints == 15:
            return get_unity15_edges(num_joints)
        if num_joints == 17:
            return [(a, b) for a, b in COCO_EDGES if a < num_joints and b < num_joints]
        return []
    raise ValueError(f"未知骨架预设: {skeleton}")


def get_edges_from_pose_info(
    pose_info: dict[str, Any],
    num_joints: int,
) -> list[tuple[int, int]]:
    keypoint_info = pose_info.get("keypoint_info", {})
    name_to_id = {
        item["name"]: int(item["id"])
        for item in keypoint_info.values()
        if isinstance(item, dict) and "name" in item and "id" in item
    }
    edges: list[tuple[int, int]] = []
    for item in pose_info.get("skeleton_info", {}).values():
        if not isinstance(item, dict) or "link" not in item:
            continue
        start_name, end_name = item["link"]
        if start_name not in name_to_id or end_name not in name_to_id:
            continue
        start_idx = name_to_id[start_name]
        end_idx = name_to_id[end_name]
        if start_idx < num_joints and end_idx < num_joints:
            edges.append((start_idx, end_idx))
    return edges


def get_unity15_edges(num_joints: int) -> list[tuple[int, int]]:
    if num_joints != len(UNITY15_TARGET_IDS):
        return []
    id_to_index = {joint_id: index for index, joint_id in enumerate(UNITY15_TARGET_IDS)}
    edges = []
    for start_id, end_id in UNITY15_BONE_EDGES_BY_ID:
        if start_id in id_to_index and end_id in id_to_index:
            edges.append((id_to_index[start_id], id_to_index[end_id]))
    return edges


def compute_plot_limits(
    sequences: Iterable[Optional[np.ndarray]],
    frame_idx: int,
    view_layout: str,
) -> Any:
    import matplotlib.pyplot as plt

    panels: list[tuple[str, np.ndarray]] = []
    if before_pose is not None:
        panels.append(("before", before_pose))
    if after_pose is not None:
        panels.append(("after", after_pose))

    views = SIMPLE_VIEWS if view_layout == "simple" else DEFAULT_VIEWS
    show_axes = view_layout != "simple"

    fig = plt.figure(figsize=(4.8 * len(views), 4.4 * len(panels)))
    fig.suptitle(f"3D pose frame {frame_idx:06d}", fontsize=14)

    for row_idx, (row_name, pose) in enumerate(panels):
        for col_idx, (view_name, elev, azim) in enumerate(views):
            subplot_idx = row_idx * len(views) + col_idx + 1
            ax = fig.add_subplot(
                len(panels),
                len(views),
                subplot_idx,
                projection="3d",
            )
            draw_pose(
                ax=ax,
                pose=pose,
                edges=edges,
                limits=limits,
                title=f"{row_name} | {view_name}",
                elev=elev,
                azim=azim,
                show_axes=show_axes,
            )

    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        bottom=0.04,
        top=0.92,
        wspace=0.12,
        hspace=0.18,
    )
    return fig


# use centralized save_figure from save_utils to keep exact old behavior
def save_figure_exact(image: Any, save_path: Path) -> None:
    save_figure(image, save_path)


def merge_frame_to_video_exact(save_path: Path, flag: str, fps: int = 30) -> Path:
    return merge_frame_to_video(save_path, flag, fps=fps)


def visualize_npz(
    npz_path: Path,
    data_key: Optional[str],
    out_dir: Path,
    max_frames: Optional[int] = None,
    frame_idx: int = 0,
    fps: int = 30,
    view_layout: str = "simple",
) -> None:
    """加载 .npz/.npy 并使用 SceneVisualizer / SkeletonVisualizer 可视化。

    输出：out_dir/frames/*.png, out_dir/single_frame/frame_*.png, out_dir/video/*.mp4
    """
    import matplotlib.pyplot as plt

    seq = load_pose_sequence(npz_path, data_key=data_key)
    if max_frames is not None:
        seq = seq[:max_frames]

    skeleton_vis, scene_vis = setup_visualizer()

    out_dir = out_dir.resolve()
    frames_dir = out_dir / "frames"
    single_dir = out_dir / "single_frame"
    video_dir = out_dir / "video"
    frames_dir.mkdir(parents=True, exist_ok=True)
    single_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # camera placeholders for SceneVisualizer
    C_L = np.zeros(3, dtype=np.float64)
    C_R = np.zeros(3, dtype=np.float64)
    left_focal = np.array(1000.0)
    right_focal = np.array(1000.0)

    for i in range(seq.shape[0]):
        pose = seq[i]

        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        # skeleton 3d
        skeleton_vis.draw_skeleton_3d(ax=ax1, points_3d=pose)

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        # scene (draw cameras + pose)
        scene_vis.draw_scene(
            ax=ax2,
            kpts_world=pose,
            C_L_world=C_L,
            C_R_world=C_R,
            left_focal_length=left_focal,
            right_focal_length=right_focal,
            frustum_depth=0.5,
            elev=-30,
            azim=270,
        )

        fig.suptitle(f"Frame {i:06d}")
        save_path = frames_dir / f"{i:06d}.png"
        save_figure_exact(fig, save_path)

        if i == frame_idx:
            shutil.copyfile(save_path, single_dir / f"frame_{i:06d}.png")

    merged = merge_frame_to_video_exact(out_dir, flag="frames", fps=fps)
    target = video_dir / "npz_visual.mp4"
    if merged != target:
        shutil.move(str(merged), str(target))
    print(f"[done] npz visualization saved: {out_dir}")


def resolve_video_name(
    before_exists: bool,
    after_exists: bool,
    custom_name: Optional[str],
) -> str:
    if custom_name:
        return custom_name
    if before_exists and after_exists:
        return "before_after_compare.mp4"
    if after_exists:
        return "after.mp4"
    return "before.mp4"


def run_visualization(args: argparse.Namespace) -> None:
    """Run visualization using an argparse.Namespace of options.

    This function contains the core logic originally in `main()` and is
    safe to import and call from other scripts.
    """
    out_dir = args.out_dir.resolve()
    frames_dir = out_dir / "frames"
    single_dir = out_dir / "single_frame"
    video_dir = out_dir / "video"
    frames_dir.mkdir(parents=True, exist_ok=True)
    single_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    before_seq = (
        load_pose_sequence(args.before, args.before_key)
        if getattr(args, "before", None)
        else None
    )
    after_seq = (
        load_pose_sequence(args.after, args.after_key)
        if getattr(args, "after", None)
        else None
    )

    if before_seq is not None:
        before_seq = center_pose_sequence(before_seq, args.center_mode)
    if after_seq is not None:
        after_seq = center_pose_sequence(after_seq, args.center_mode)

    num_frames_candidates = [
        seq.shape[0] for seq in (before_seq, after_seq) if seq is not None
    ]
    if not num_frames_candidates:
        raise RuntimeError("没有可用的 3D 序列")

    num_frames = min(num_frames_candidates)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("max_frames 必须大于 0")
        num_frames = min(num_frames, args.max_frames)
    if before_seq is not None and before_seq.shape[0] != num_frames:
        print(f"[warn] before 序列长度为 {before_seq.shape[0]}，将截断到 {num_frames}")
    if after_seq is not None and after_seq.shape[0] != num_frames:
        print(f"[warn] after 序列长度为 {after_seq.shape[0]}，将截断到 {num_frames}")

    if before_seq is not None:
        before_seq = before_seq[:num_frames]
    if after_seq is not None:
        after_seq = after_seq[:num_frames]

    ref_seq = after_seq if after_seq is not None else before_seq
    assert ref_seq is not None
    edges = get_edges(args.skeleton, ref_seq.shape[1])
    limits = compute_plot_limits([before_seq, after_seq])

    video_path = video_dir / resolve_video_name(
        before_exists=before_seq is not None,
        after_exists=after_seq is not None,
        custom_name=args.video_name,
    )

    for frame_idx in range(num_frames):
        frame_fig = render_comparison_frame(
            before_pose=(None if before_seq is None else before_seq[frame_idx]),
            after_pose=None if after_seq is None else after_seq[frame_idx],
            edges=edges,
            limits=limits,
            frame_idx=frame_idx,
            view_layout=args.view_layout,
        )

        frame_path = frames_dir / f"{frame_idx:06d}.png"
        save_figure_exact(frame_fig, frame_path)

        if frame_idx == args.frame_idx:
            single_path = single_dir / f"frame_{frame_idx:06d}.png"
            shutil.copyfile(frame_path, single_path)

    merged_path = merge_frame_to_video_exact(
        save_path=out_dir,
        flag="frames",
        fps=args.fps,
    )
    if merged_path != video_path:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(merged_path), str(video_path))

    if args.frame_idx < 0 or args.frame_idx >= num_frames:
        raise IndexError(f"frame_idx 超出范围: {args.frame_idx}, 总帧数为 {num_frames}")

    print(f"[done] 单帧图片: {single_dir / f'frame_{args.frame_idx:06d}.png'}")
    print(f"[done] 逐帧图片目录: {frames_dir}")
    print(f"[done] 视频: {video_path}")
