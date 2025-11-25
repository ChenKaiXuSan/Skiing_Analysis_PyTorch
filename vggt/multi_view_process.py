#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
vggt_video_infer.py
从单个视频抽帧并执行 VGGT 推理，可作为函数调用。
"""

import cv2
import shutil
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from torchvision.io import write_png

from vggt.reproject import reproject_and_visualize
from vggt.load import load_info, load_and_preprocess_images
from vggt.save import save_inference_results, update_pt_with_3d_info

from vggt.triangulate import triangulate_one_frame

from vggt.vggt.infer import CameraHead

from vggt.bundle_adjustment.main import run_local_ba
from vggt.vis.pose_visualization import visualize_3d_joints, save_stereo_pose_frame

logger = logging.getLogger(__name__)


K = np.array(
    [
        1116.9289548941917,
        0.0,
        955.77175993563799,
        0.0,
        1117.3341496962166,
        538.91061167202145,
        0.0,
        0.0,
        1.0,
    ]
).reshape(3, 3)

K_dist = np.array(
    [
        -1.1940477842823853,
        -15.440461757486913,
        0.00013163161053023783,
        0.00019082529328353381,
        98.843073622415901,
        -1.3588290520381034,
        -14.555841222727574,
        96.219667412855202,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)


# --------------------------------------------------------------------------- #
# Processing functions
# --------------------------------------------------------------------------- #
def process_multi_view_video(
    left_video_path: Path,
    left_pt_path: Path,
    right_video_path: Path,
    right_pt_path: Path,
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

    out_dir = out_root / "multi_view" / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Run-MV] {left_video_path} & {right_video_path} → {out_dir} | ")

    # * load info from pt and video
    left_kpts, left_kpt_scores, *_, left_frames = load_info(
        video_file_path=left_video_path.as_posix(), pt_file_path=left_pt_path.as_posix()
    )
    right_kpts, right_kpt_scores, *_, right_frames = load_info(
        video_file_path=right_video_path.as_posix(),
        pt_file_path=right_pt_path.as_posix(),
    )

    all_frame_raw_x3d = []
    all_frame_camera_extrinsics = []
    all_frame_camera_intrinsics = []
    all_frame_R = []
    all_frame_t = []
    all_frame_C = []

    camera_head = CameraHead(cfg, out_dir / "vggt_infer")

    for idx in tqdm(
        range(0, min(len(left_frames), len(right_frames))), desc="Processing frames"
    ):
        # if idx > 30:
        #     break

        # save images
        img_dir = out_dir / "raw_frames" / f"frame_{idx:04d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        write_png(left_frames[idx].permute(2, 0, 1), img_dir / f"left_{idx:04d}.png")
        write_png(right_frames[idx].permute(2, 0, 1), img_dir / f"right_{idx:04d}.png")

        # infer vggt
        camera_extrinsics, camera_intrinsics_resized, R, t, C = (
            camera_head.reconstruct_from_frames(
                imgs=[
                    left_frames[idx],
                    right_frames[idx],
                ],
                frame_id=idx,
            )
        )

        # TODO: 算repro和可视化的代码应该放到外面来
        x3d, reprojet_err = triangulate_one_frame(
            kptL=left_kpts[idx],
            kptR=right_kpts[idx],
            frame_L=left_frames[idx].numpy(),
            frame_R=right_frames[idx].numpy(),
            K=np.stack(
                [camera_intrinsics_resized[0], camera_intrinsics_resized[1]], axis=0
            ),
            R=np.stack([R[0], R[1]], axis=0),
            T=np.stack([t[0], t[1]], axis=0),
            save_dir=out_dir / "triangulation" / f"frame_{idx:04d}",
            dist=None,
            visualize_3d=True,
            frame_num=idx,
        )

        # write reprojetion error to log
        with open(out_dir / "reprojection_error.txt", "a") as f:
            f.write(f"Frame {idx:04d} Reprojection Error (in pixels):\n")
            for k, v in reprojet_err.items():
                if isinstance(v, np.ndarray):
                    continue
                f.write(f"  {k}: {v}\n")

        # filter by reprojection error
        # if reprojet_err.get("mean_err_L", np.inf) < cfg.infer.get(
        #     "reproj_err", 20
        # ) and reprojet_err.get("mean_err_R", np.inf) < cfg.infer.get("reproj_err", 20):
        #     best_out_dir = out_dir / "best_frames"
        #     best_out_dir.mkdir(parents=True, exist_ok=True)

        #     # copy the results to best_frames

        #     shutil.copy(
        #         out_dir / f"frames/frame_{idx:04d}/triangulation/stereo_pose_frame.jpg",
        #         best_out_dir / f"frame_{idx:04d}.jpg",
        #     )

        all_frame_raw_x3d.append(x3d)
        all_frame_camera_extrinsics.append(camera_extrinsics)
        all_frame_camera_intrinsics.append(camera_intrinsics_resized)
        all_frame_R.append(R)
        all_frame_t.append(t)
        all_frame_C.append(C)

    R_init = np.array(all_frame_R)  # (T,C,3,3)
    t_init = np.array(all_frame_t)  # (T,C,3)
    X3d_init = np.array(all_frame_raw_x3d)  # (T,J,3)
    x2d = np.array(
        [
            np.stack([left_kpts[i], right_kpts[i]], axis=0)
            for i in range(len(all_frame_raw_x3d))
        ]
    )  # (T,C,J,2)
    conf2d = np.array(
        [
            np.stack([left_kpt_scores[i], right_kpt_scores[i]], axis=0)
            for i in range(len(all_frame_raw_x3d))
        ]
    )  # (T,C,J)

    # bundle adjustment
    out_dir_ba = out_dir / "ba_results"
    out_dir_ba.mkdir(parents=True, exist_ok=True)

    # 对摄像机的K取平均
    avg_K = np.mean(np.array(all_frame_camera_intrinsics), axis=0)

    # R_init: (T, C, 3, 3), t_init: (T, C, 3)
    T, C = R_init.shape[:2]

    rvec_init = np.zeros((T, C, 3, 3), dtype=np.float32)
    for t in range(T):
        for c in range(C):
            rvec, _ = cv2.Rodrigues(R_init[t, c])
            rvec_init[t, c] = rvec.reshape(3)

    # 转成 torch
    rvec_init_torch = torch.from_numpy(rvec_init)  # (T,C,3,3)
    tvec_init_torch = torch.from_numpy(t_init)  # (T,C,3)
    X3d_init_torch = torch.from_numpy(X3d_init)  # (T,J,3)
    K_torch = torch.from_numpy(avg_K).float()  # (C,3,3)
    x2d_torch = torch.from_numpy(x2d).float()  # (T,C,J,2)
    conf2d_torch = torch.from_numpy(conf2d).float()  # (T,C,J)

    X_opt, t_opt = run_local_ba(
        K_torch=K_torch,  # (C,3,3)
        R_init_torch=rvec_init_torch,  # (T,C,3,3)  固定
        t_init_torch=tvec_init_torch,  # (T,C,3)    可优化
        X3d_init_torch=X3d_init_torch,  # (T,J,3)    可优化
        x2d_torch=x2d_torch,  # (T,C,J,2)
        conf2d_torch=conf2d_torch,  # (T,C,J)
        num_iters=cfg.bundle_adjustment.get("num_iters", 200),
        lr=cfg.bundle_adjustment.get("lr", 1e-3),
        device=cfg.infer.get("device", "cuda"),
    )

    # use the optimized results to draw and save
    for idx in range(X_opt.shape[0]):
        frame_out_dir = out_dir_ba / f"frame_{idx:04d}"
        frame_out_dir.mkdir(parents=True, exist_ok=True)

        frame_L = left_frames[idx].numpy()  # (H,W,3)
        frame_R = right_frames[idx].numpy()  # (H,W,3)
        kptL = left_kpts[idx]  # (J,2)
        kptR = right_kpts[idx]  # (J,2)
        R = np.stack([all_frame_R[idx][0], all_frame_R[idx][1]], axis=0)  # (C,3,3)
        t = t_opt[idx]  # (C,3)
        K = K_torch.cpu().numpy()  # (3,3)
        X3d = X_opt[idx]  # (J,3)

        # reproject and visualize
        res = reproject_and_visualize(
            img1=frame_L,
            img2=frame_R,
            X3=X3d,
            kptL=kptL,  # (J,2)
            kptR=kptR,  # (J,2)
            K1=K[0],
            dist1=K_dist,
            K2=K[1],
            dist2=K_dist,
            R=R,  # （2，3，3），世界》相机
            T=t,  # （2，3），世界》相机
            joint_names=None,  # 或者传 COCO 的关节名列表
            out_path=frame_out_dir / f"ba_reproj_{idx:04d}.jpg",
        )
        # ---- 可视化 3D ---- #
        visualize_3d_joints(
            R=R,
            T=t,
            K=K,
            joints_3d=X3d,
            save_path=frame_out_dir / f"3d_joints_{idx:04d}.png",
            title=f"3D Triangulated Result",
            image_size=(frame_L.shape[1], frame_L.shape[0]),
        )

        save_stereo_pose_frame(
            R=R,
            T=t,
            K=K,
            img_left=frame_L,
            img_right=frame_R,
            kpt_left=kptL,
            kpt_right=kptR,
            pose_3d=X3d,
            output_path=frame_out_dir / f"stereo_pose_frame_{idx:04d}.jpg",
            repoj_error=res,
            frame_num=idx,
        )

    return out_dir
