#!/usr/bin/env python3

import logging
import gc

from pathlib import Path
from typing import Iterator

from tqdm import tqdm
import numpy as np

from .load import (
    OnePersonInfo,
    load_2d_keypoints,
    load_helper,
    load_video_frames,
)
from .metadata.mhr70 import pose_info as mhr70_pose_info
from .visualization.save_utils import merge_frame_to_video
from .visualization.scene_visualizer import SceneVisualizer
from .visualization.skeleton_visualizer import SkeletonVisualizer

PrefusionSharedInputs = tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]

logger = logging.getLogger(__name__)


def setup_visualizer():
    """Setup visualizers with default pose meta (MHR70)."""

    skeleton_visualizer = SkeletonVisualizer(line_width=2, radius=5)
    skeleton_visualizer.set_pose_meta(mhr70_pose_info)

    scene_visualizer = SceneVisualizer(line_width=2, radius=5)
    scene_visualizer.set_pose_meta(mhr70_pose_info)

    return skeleton_visualizer, scene_visualizer


def run_visualization(
    person_info: OnePersonInfo,
    out_dir: Path,
) -> None:
    """Run visualization using an argparse.Namespace of options.

    This function contains the core logic originally in `main()` and is
    safe to import and call from other scripts.
    """
    left_frames = None
    right_frames = None
    left_2d_kpt = None
    right_2d_kpt = None
    fused_3d_kpt = None
    fused_smoothed_3d_kpt = None

    try:
        out_dir = out_dir.resolve()
        logger.info(
            "Starting visualization for %s to %s", person_info.person_name, str(out_dir)
        )

        (
            left_frames,
            right_frames,
            left_2d_kpt,
            right_2d_kpt,
            fused_3d_kpt,
            fused_smoothed_3d_kpt,
        ) = load_helper(person_info, memory_efficient=True)

        skeleton_visualizer, scene_visualizer = setup_visualizer()

        # * pro的左右视频长度不一致，按最短的来
        frame_count = min(
            len(left_frames),
            len(right_frames),
            len(left_2d_kpt),
            len(right_2d_kpt),
            fused_3d_kpt.shape[0],
            fused_smoothed_3d_kpt.shape[0],
        )

        for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
            # if frame_idx > 50:  # 先只处理前50帧，测试用
            #     break

            try:
                process_frame(
                    left_frames=left_frames[frame_idx],
                    right_frames=right_frames[frame_idx],
                    left_2d_kpt=left_2d_kpt[frame_idx],
                    right_2d_kpt=right_2d_kpt[frame_idx],
                    fused_3d_kpt=fused_3d_kpt[frame_idx],
                    fused_smoothed_3d_kpt=fused_smoothed_3d_kpt[frame_idx],
                    frame_idx=frame_idx,
                    out_root=out_dir,
                    skeleton_visualizer=skeleton_visualizer,
                    scene_visualizer=scene_visualizer,
                )
            except Exception as e:
                logger.error("Error processing frame %d: %s", frame_idx, e)
                continue

        logger.info("Visualization completed for %s", person_info.person_name)

        # 合成图片为视频
        merge_frame_to_video(out_dir, "fused", fps=30)
        merge_frame_to_video(out_dir, "smoothed", fps=30)
        merge_frame_to_video(out_dir, "frame_scene", fps=30)

    except Exception as e:
        logger.error("Error in visualization: %s", e)
        raise
    finally:
        # 结束之后清理内存
        try:
            if left_frames is not None:
                del left_frames
            if right_frames is not None:
                del right_frames
            if left_2d_kpt is not None:
                del left_2d_kpt
            if right_2d_kpt is not None:
                del right_2d_kpt
            if fused_3d_kpt is not None:
                del fused_3d_kpt
            if fused_smoothed_3d_kpt is not None:
                del fused_smoothed_3d_kpt
            gc.collect()
            logger.info("Memory cleaned up")
        except Exception:
            pass


def _transform_points_for_vis(
    points_3d: np.ndarray,
    person_name: str,
) -> np.ndarray:
    """Align coordinate system with fused visualization for fair comparison."""
    pts = points_3d.copy()
    pts[:, [1, 2]] = pts[:, [2, 1]]
    pts[:, 2] = -pts[:, 2]

    if "pro" in person_name.lower():
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]

    return pts


def _iter_prefusion_3d(side_dir: Path) -> Iterator[np.ndarray]:
    for npz_file in sorted(side_dir.rglob("*.npz")):
        with np.load(npz_file, allow_pickle=True) as loaded:
            outputs = loaded["outputs"]
            if len(outputs) == 0:
                continue

            first = outputs[0]
            if not isinstance(first, dict) or "pred_keypoints_3d" not in first:
                continue

            yield first["pred_keypoints_3d"]


def load_prefusion_shared_inputs(
    person_info: OnePersonInfo,
) -> PrefusionSharedInputs:
    """Load reusable inputs for left/right prefusion visualization once per person."""
    left_frames = list(
        load_video_frames(person_info.left_video_path, as_generator=True)
    )
    right_frames = list(
        load_video_frames(person_info.right_video_path, as_generator=True)
    )
    left_2d_kpt = list(
        load_2d_keypoints(person_info.left_2d_kpt_path, as_generator=True)
    )
    right_2d_kpt = list(
        load_2d_keypoints(person_info.right_2d_kpt_path, as_generator=True)
    )
    return (left_frames, right_frames, left_2d_kpt, right_2d_kpt)


def run_prefusion_visualization(
    person_info: OnePersonInfo,
    side: str,
    out_dir: Path,
    shared_inputs: PrefusionSharedInputs | None = None,
) -> None:
    """Visualize prefusion 3D points in run_visualization style."""
    left_frames = None
    right_frames = None
    left_2d_kpt = None
    right_2d_kpt = None
    prefusion_3d = None
    loaded_in_function = shared_inputs is None

    try:
        out_dir = out_dir.resolve()
        person_name = person_info.person_name
        side_dir = (
            person_info.left_2d_kpt_path
            if side == "left"
            else person_info.right_2d_kpt_path
        )

        logger.info(
            "Starting prefusion visualization for %s-%s to %s",
            person_name,
            side,
            str(out_dir),
        )

        if shared_inputs is None:
            shared_inputs = load_prefusion_shared_inputs(person_info)

        (
            left_frames,
            right_frames,
            left_2d_kpt,
            right_2d_kpt,
        ) = shared_inputs
        prefusion_3d = list(_iter_prefusion_3d(side_dir))

        skeleton_visualizer, scene_visualizer = setup_visualizer()

        frame_count = min(
            len(left_frames),
            len(right_frames),
            len(left_2d_kpt),
            len(right_2d_kpt),
            len(prefusion_3d),
        )

        for frame_idx in tqdm(
            range(frame_count),
            desc=f"Processing prefusion {person_name}-{side}",
        ):
            # if frame_idx > 50:  # 先只处理前50帧，测试用
            #     break

            points_3d = _transform_points_for_vis(
                prefusion_3d[frame_idx],
                person_name,
            )

            skeleton = skeleton_visualizer.draw_skeleton_3d(
                ax=None,
                points_3d=points_3d,
            )
            skeleton_visualizer.save(
                image=skeleton,
                save_path=out_dir / "fused" / f"{frame_idx}.png",
            )

            # prefusion 没有 smoothed，复用同一帧，保证输出结构对齐。
            skeleton_smoothed = skeleton_visualizer.draw_skeleton_3d(
                ax=None,
                points_3d=points_3d,
            )
            skeleton_visualizer.save(
                image=skeleton_smoothed,
                save_path=out_dir / "smoothed" / f"{frame_idx}.png",
            )

            left_kpt_with_frame = skeleton_visualizer.draw_skeleton(
                image=left_frames[frame_idx],
                keypoints=left_2d_kpt[frame_idx],
            )
            right_kpt_with_frame = skeleton_visualizer.draw_skeleton(
                image=right_frames[frame_idx],
                keypoints=right_2d_kpt[frame_idx],
            )

            frame_scene = scene_visualizer.draw_frame_with_scene(
                left_frame=left_kpt_with_frame,
                right_frame=right_kpt_with_frame,
                pose_3d=points_3d,
            )
            scene_visualizer.save(
                image=frame_scene,
                save_path=out_dir / "frame_scene" / f"{frame_idx}.png",
            )

        if frame_count == 0:
            logger.warning(
                "Skip %s-%s: no valid prefusion frames in %s",
                person_name,
                side,
                side_dir,
            )
            return

        merge_frame_to_video(out_dir, "fused", fps=30)
        merge_frame_to_video(out_dir, "smoothed", fps=30)
        merge_frame_to_video(out_dir, "frame_scene", fps=30)

        logger.info(
            "Prefusion visualization completed for %s-%s",
            person_name,
            side,
        )
    except Exception as e:
        logger.error("Error in prefusion visualization: %s", e)
        raise
    finally:
        try:
            if loaded_in_function:
                if left_frames is not None:
                    del left_frames
                if right_frames is not None:
                    del right_frames
                if left_2d_kpt is not None:
                    del left_2d_kpt
                if right_2d_kpt is not None:
                    del right_2d_kpt
            if prefusion_3d is not None:
                del prefusion_3d
            gc.collect()
            logger.info("Prefusion memory cleaned up")
        except Exception:
            pass


def process_frame(
    left_frames: np.ndarray,
    right_frames: np.ndarray,
    left_2d_kpt: np.ndarray,
    right_2d_kpt: np.ndarray,
    fused_3d_kpt: np.ndarray,
    fused_smoothed_3d_kpt: np.ndarray,
    frame_idx: int,
    out_root: Path,
    skeleton_visualizer: SkeletonVisualizer,
    scene_visualizer: SceneVisualizer,
) -> None:
    # 转变坐标系（SAM的坐标系是右手系，Z轴向前；我们希望的坐标系是Z轴向上）
    fused_3d_kpt = fused_3d_kpt.copy()
    fused_3d_kpt[:, [1, 2]] = fused_3d_kpt[:, [2, 1]]  # swap Y and Z
    fused_3d_kpt[:, 2] = -fused_3d_kpt[
        :, 2
    ]  # invert new Z (old Y) to make it forward-facing

    fused_smoothed_3d_kpt = fused_smoothed_3d_kpt.copy()
    fused_smoothed_3d_kpt[:, [1, 2]] = fused_smoothed_3d_kpt[:, [2, 1]]  # swap Y and Z
    fused_smoothed_3d_kpt[:, 2] = -fused_smoothed_3d_kpt[
        :, 2
    ]  # invert new Z (old Y) to make it forward-facing

    person_name = out_root.name
    if "pro" in person_name.lower():
        # pro数据需要再额外绕y轴旋转180度，使得人物朝向一致
        fused_3d_kpt[:, 0] = -fused_3d_kpt[:, 0]  # invert X for pro
        fused_3d_kpt[:, 1] = -fused_3d_kpt[:, 1]  # invert Y for pro
        fused_smoothed_3d_kpt[:, 0] = -fused_smoothed_3d_kpt[:, 0]  # invert X for pro
        fused_smoothed_3d_kpt[:, 1] = -fused_smoothed_3d_kpt[:, 1]  # invert Y for pro

    # ---------- 画骨架图 ----------
    # * 时间优化前的
    _skeleton = skeleton_visualizer.draw_skeleton_3d(ax=None, points_3d=fused_3d_kpt)

    skeleton_visualizer.save(
        image=_skeleton,
        save_path=out_root / "fused" / f"{frame_idx}.png",
    )

    # * 时间优化后的
    _skeleton_smoothed = skeleton_visualizer.draw_skeleton_3d(
        ax=None, points_3d=fused_smoothed_3d_kpt
    )

    skeleton_visualizer.save(
        image=_skeleton_smoothed,
        save_path=out_root / "smoothed" / f"{frame_idx}.png",
    )

    # 画左右frame + 2D骨架

    left_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=left_frames, keypoints=left_2d_kpt
    )
    right_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=right_frames, keypoints=right_2d_kpt
    )

    _frame_scene = scene_visualizer.draw_frame_with_scene(
        left_frame=left_kpt_with_frame,
        right_frame=right_kpt_with_frame,
        pose_3d=fused_smoothed_3d_kpt,
    )

    scene_visualizer.save(
        image=_frame_scene,
        save_path=out_root / "frame_scene" / f"{frame_idx}.png",
    )
