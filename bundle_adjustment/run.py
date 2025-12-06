#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/bundle_adjustment/main.py
Project: /workspace/code/vggt/bundle_adjustment
Created Date: Tuesday November 25th 2025
Author: Kaixu Chen
-----
Comment:
Local bundle adjustment with configurable optimisation modes.
You can choose to refine only 3D pose, pose+translation, or
full pose+camera (R,t) under multi-view geometric constraints.

Have a good code time :)
-----
Last Modified: Tuesday November 25th 2025 10:30:00 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple

from omegaconf import DictConfig
import torch
import torch.nn as nn

from .loss import (
    reprojection_loss,
    camera_smooth_loss,
    baseline_reg_loss,
    bone_length_loss,
    pose_temporal_loss,
)

from bundle_adjustment.load import (
    load_info,
    load_sam_3d_body_results,
    load_vggt_results,
    load_videopose3d_results,
)


# ------------------------- SO(3) helpers ------------------------- #
def rodrigues_batch(rvec: torch.Tensor) -> torch.Tensor:
    """
    批量 Rodrigues 公式，将轴角向量 r 转成旋转矩阵 R。

    rvec: (..., 3)
    return: (..., 3, 3)
    """
    orig_shape = rvec.shape
    device = rvec.device
    dtype = rvec.dtype

    r = rvec.reshape(-1, 3)  # (N,3)
    N = r.shape[0]
    theta = torch.linalg.norm(r, dim=1, keepdim=True)  # (N,1)
    eps = 1e-8
    k = r / (theta + eps)  # 单位旋转轴 (N,3)

    K = torch.zeros((N, 3, 3), device=device, dtype=dtype)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    K[:, 0, 1] = -kz
    K[:, 0, 2] = ky
    K[:, 1, 0] = kz
    K[:, 1, 2] = -kx
    K[:, 2, 0] = -ky
    K[:, 2, 1] = kx

    theta = theta.view(-1, 1, 1)  # (N,1,1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    eye = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    K2 = K @ K

    R = eye + sin_t * K + (1.0 - cos_t) * K2
    R = R.view(*orig_shape[:-1], 3, 3)
    return R


# ------------------------- BA main ------------------------- #
def run_local_ba(
    K_torch: torch.Tensor,  # (C,3,3)
    R_init_torch: torch.Tensor,  # (T,C,3,3)  作为基准旋转
    t_init_torch: torch.Tensor,  # (T,C,3)    初始平移
    X3d_init_torch: torch.Tensor,  # (T,J,3)    初始 3D 关节点
    x2d_torch: torch.Tensor,  # (T,C,J,2)
    conf2d_torch: torch.Tensor,  # (T,C,J)
    num_iters: int = 200,
    lr: float = 1e-3,
    device: str = "cuda",
    # ---- mode: 决定优化哪些参数 ----
    #   "pose_only"   : 只优化 X3d
    #   "pose_cam_t"  : 优化 X3d + t
    #   "full"        : 优化 X3d + t + R(增量)
    mode: str = "pose_only",
    # 各项 loss 的权重（可按模式自动设定）
    lambda_reproj: Optional[float] = None,
    lambda_cam_smooth: Optional[float] = None,
    lambda_baseline: Optional[float] = None,
    lambda_bone: Optional[float] = None,
    lambda_pose_smooth: Optional[float] = None,
    # 训练细节
    grad_clip: Optional[float] = 1.0,
    verbose: bool = True,
    return_history: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, list]]]:
    """
    Local bundle adjustment with configurable optimisation modes.

    Parameters
    ----------
    K_torch : (C,3,3)
        Camera intrinsics (assumed fixed).
    R_init_torch : (T,C,3,3)
        Initial rotations (world->camera) from VGGT or stereo calibration.
    t_init_torch : (T,C,3)
        Initial translations (world->camera).
    X3d_init_torch : (T,J,3)
        Initial 3D joints (e.g., from DLT triangulation).
    x2d_torch : (T,C,J,2)
        Observed 2D keypoints in pixel coordinates.
    conf2d_torch : (T,C,J)
        Confidence weights for 2D joints in [0,1].

    mode : {"pose_only","pose_cam_t","full"}
        - "pose_only"  : refine only X3d (最稳，推荐先用)
        - "pose_cam_t" : refine X3d and camera translation t
        - "full"       : refine X3d, t, and incremental rotations r_delta

    If any lambda_* is None, a reasonable default is chosen
    depending on the selected mode.

    Returns
    -------
    R_opt : (T,C,3,3)
        Optimised rotations (world->camera).
    t_opt : (T,C,3)
        Optimised translations.
    X_opt : (T,J,3)
        Optimised 3D joints.
    history : dict or None
        If return_history=True, contains per-iteration loss curves.
    """

    # ---------- 1) 根据 mode 设置默认权重 ---------- #
    if mode not in ["pose_only", "pose_cam_t", "full"]:
        raise ValueError(f"Unknown BA mode: {mode}")

    # 若用户没显式给，就用比较稳的默认值
    if lambda_reproj is None:
        lambda_reproj = 1.0
    if lambda_bone is None:
        lambda_bone = 1e-2
    if lambda_pose_smooth is None:
        lambda_pose_smooth = 1e-2

    # 与相机相关的约束，仅在需要优化相机时才有效
    if lambda_cam_smooth is None:
        lambda_cam_smooth = 0.0 if mode == "pose_only" else 1e-2
    if lambda_baseline is None:
        lambda_baseline = 0.0 if mode == "pose_only" else 1e-2

    # ---------- 2) 准备参数 ---------- #
    K = K_torch.to(device=device, dtype=torch.float32)
    R_init = R_init_torch.to(device=device, dtype=torch.float32)

    # 3D pose 总是可优化
    X_param = nn.Parameter(X3d_init_torch.to(device=device, dtype=torch.float32))

    # 根据 mode 决定是否优化 t, R
    optimize_t = mode in ["pose_cam_t", "full"]
    optimize_R = mode == "full"

    if optimize_t:
        t_param = nn.Parameter(t_init_torch.to(device=device, dtype=torch.float32))
    else:
        # 不优化时，就直接用常量
        t_param = t_init_torch.to(device=device, dtype=torch.float32)

    if optimize_R:
        # 增量旋转 r_delta，初始为 0
        r_delta_param = nn.Parameter(
            torch.zeros(
                R_init.shape[0],  # T
                R_init.shape[1],  # C
                3,
                3,
                device=device,
                dtype=torch.float32,
            )
        )
    else:
        r_delta_param = None  # 不参与优化

    # 组装 optimizer 参数
    optim_params = [X_param]
    if optimize_t:
        optim_params.append(t_param)
    if optimize_R:
        optim_params.append(r_delta_param)

    optimizer = torch.optim.Adam(optim_params, lr=lr)

    x2d = x2d_torch.to(device=device, dtype=torch.float32)
    conf2d = conf2d_torch.to(device=device, dtype=torch.float32)

    history: Dict[str, list] = {
        "total": [],
        "reproj": [],
        "cam_smooth": [],
        "baseline": [],
        "bone": [],
        "pose": [],
    }

    # ---------- 3) 优化循环 ---------- #
    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)

        # 当前旋转 R_cur：如果不优化 R，就直接用 R_init
        if optimize_R:
            # R_delta = rodrigues_batch(r_delta_param)     # (T,C,3,3)
            R_delta = r_delta_param
            R_cur = torch.matmul(R_delta, R_init)  # (T,C,3,3)
        else:
            R_cur = R_init

        # 当前平移 t_cur：如果不优化 t，就把常量视作 "param" 输入
        t_cur = t_param if optimize_t else t_param

        # 1) 重投影误差
        L_reproj = reprojection_loss(
            X_param,
            R_cur,
            t_cur,
            K,
            x2d,
            conf2d,
            w=lambda_reproj,
        )

        # 2) 相机约束：仅在 optimize_t 或 optimize_R 时启用
        if optimize_t or optimize_R:
            L_cam_smooth = camera_smooth_loss(
                R_cur,
                t_cur,
                w=lambda_cam_smooth,
            )
            L_baseline = baseline_reg_loss(
                R_cur,
                t_cur,
                w=lambda_baseline,
            )
        else:
            L_cam_smooth = torch.tensor(0.0, device=device)
            L_baseline = torch.tensor(0.0, device=device)

        # 3) 3D pose 几何 + 时序正则
        L_bone = bone_length_loss(
            X_param,
            ref_bone_len=None,
            w=lambda_bone,
        )
        L_pose = pose_temporal_loss(
            X_param,
            w=lambda_pose_smooth,
        )

        loss = L_reproj + L_cam_smooth + L_baseline + L_bone + L_pose
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=grad_clip)

        optimizer.step()

        # 记录历史
        history["total"].append(float(loss.detach().cpu()))
        history["reproj"].append(float(L_reproj.detach().cpu()))
        history["cam_smooth"].append(float(L_cam_smooth.detach().cpu()))
        history["baseline"].append(float(L_baseline.detach().cpu()))
        history["bone"].append(float(L_bone.detach().cpu()))
        history["pose"].append(float(L_pose.detach().cpu()))

        if verbose and ((it + 1) % 20 == 0 or it == 1 or it == num_iters):
            print(
                f"[BA-{mode}] iter={it + 1:4d} "
                f" total={loss.item():.4f} | "
                f" reproj={L_reproj.item():.4f} "
                f" cam_sm={L_cam_smooth.item():.4f} "
                f" base={L_baseline.item():.4f} "
                f" bone={L_bone.item():.4f} "
                f" pose={L_pose.item():.4f}"
            )

    # ---------- 4) 导出最终结果 ---------- #
    with torch.no_grad():
        if optimize_R:
            # R_delta_final = rodrigues_batch(r_delta_param)  # (T,C,3,3)
            R_delta_final = r_delta_param
            R_opt = torch.matmul(R_delta_final, R_init)
        else:
            R_opt = R_init.clone()

    X_opt = X_param.detach()
    t_opt = t_param.detach() if optimize_t else t_param.clone()

    if return_history:
        return R_opt.cpu(), t_opt.cpu(), X_opt.cpu(), history
    else:
        return R_opt.cpu(), t_opt.cpu(), X_opt.cpu(), None


def process_one_person(
    left_video_path: Path,
    left_pt_path: Path,
    left_sam3d_body_path: Path,
    right_video_path: Path,
    right_pt_path: Path,
    right_sam3d_body_path: Path,
    vggt_files: List[Path],
    videopose3d_files: List[Path],
    out_root: Path,
    cfg: DictConfig,
) -> Optional[Path]:
    """
    Process one person with multi-view bundle adjustment.

    Parameters
    ----------
    left_video_path : Path
        Path to the left video file.
    left_pt_path : Path
        Path to the left 2D keypoints file.
    right_video_path : Path
        Path to the right video file.
    right_pt_path : Path
        Path to the right 2D keypoints file.
    vggt_files : List[Path]
        List of VGGT numpy files for all views.
    videopose3d_files : List[Path]
        List of VideoPose3D numpy files for all views.
    out_root : Path
        Output root directory.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    out_dir : Optional[Path]
        Output directory if successful, None otherwise.
    """

    left_kpt, left_kpt_score, left_bboxes_xyxy, left_bboxes_scores, left_frame = (
        load_info(
            video_file_path=left_video_path.as_posix(),
            pt_file_path=left_pt_path.as_posix(),
            assume_normalized=False,
        )
    )

    right_kpt, right_kpt_score, right_bboxes_xyxy, right_bboxes_scores, right_frame = (
        load_info(
            video_file_path=right_video_path.as_posix(),
            pt_file_path=right_pt_path.as_posix(),
            assume_normalized=False,
        )
    )

    left_sam3d_body_res = load_sam_3d_body_results(left_sam3d_body_path.as_posix())
    right_sam3d_body_res = load_sam_3d_body_results(right_sam3d_body_path.as_posix())

    videopose3d_res = load_videopose3d_results(videopose3d_files)
    vggt_res = load_vggt_results(vggt_files)

    out_dir = run_local_ba(
        left_kpt,
        left_kpt_score,
        left_bboxes_xyxy,
        left_bboxes_scores,
        left_frame,
        right_kpt,
        right_kpt_score,
        right_bboxes_xyxy,
        right_bboxes_scores,
        right_frame,
        vggt_res,
        videopose3d_res,
        out_root,
        cfg,
    )
    return out_dir
