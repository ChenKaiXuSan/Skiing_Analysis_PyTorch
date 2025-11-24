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
from tqdm import tqdm
from typing import List, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import torch
from torchvision.io import write_png

# 依赖 VGGT 官方模块
from vggt.vggt.models.vggt import VGGT

from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map

from vggt.load import load_info, load_and_preprocess_images
from vggt.save import save_inference_results, update_pt_with_3d_info

from vggt.vis.vggt_camera_vis import plot_cameras_from_predictions

from vggt.triangulate import triangulate_one_frame


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


# ==========================
# 工具函数
# ==========================


def scale_intrinsics(
    K: np.ndarray, orig_size: tuple[int, int], new_size: tuple[int, int]
) -> np.ndarray:
    """
    当图像从原始分辨率 resize 到新分辨率时，同步缩放相机内参 K。

    K: (3,3) 内参矩阵，像素坐标
       [[fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]]
    orig_size: (H, W) 原始图像分辨率
    new_size:  (H, W) 新图像分辨率

    返回缩放后的 K_new
    """
    H0, W0 = orig_size
    H1, W1 = new_size

    sx = W1 / W0  # 水平方向缩放比例
    sy = H1 / H0  # 垂直方向缩放比例

    K_new = K.copy().astype(float)
    K_new[0, 0] *= sx  # fx
    K_new[1, 1] *= sy  # fy
    K_new[0, 2] *= sx  # cx
    K_new[1, 2] *= sy  # cy

    return K_new


def load_vggt_model(device="cuda", verbose=True):
    """加载预训练 VGGT 模型"""
    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=verbose
    )
    model.load_state_dict(state)
    model.eval().to(device)
    return model


@torch.no_grad()
def run_vggt(
    images: List[str],
    model,
    device="cuda",
) -> Dict[str, np.ndarray]:
    """对图像列表执行 VGGT 推理"""
    imgs = load_and_preprocess_images(images).to(device)
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    with torch.cuda.amp.autocast(dtype=dtype):
        preds = model(imgs)

    H, W = imgs.shape[-2:]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
    preds["extrinsic"] = extrinsic
    preds["intrinsic"] = intrinsic

    # 转 numpy
    out = {
        k: (v.detach().cpu().numpy().squeeze(0) if isinstance(v, torch.Tensor) else v)
        for k, v in preds.items()
    }
    out["pose_enc_list"] = None
    depth = out["depth"]
    out["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth, out["extrinsic"], out["intrinsic"]
    )
    return out


# ==========================
# 主函数接口
# ==========================
def reconstruct_from_frames(
    imgs: list[torch.Tensor],
    outdir: Path,
    conf_thres: float = 50.0,
    prediction_mode: str = "Depthmap and Camera Branch",
    verbose: bool = True,
    gpu: int = 0,
) -> Dict[str, str]:
    """
    从视频执行 VGGT 重建。
    返回：
        {
          'npz_path': str,
          'glb_path': str,
          'ply_path': Optional[str],
          'n_frames': int,
          'time': float
        }
    """

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    if "cuda" not in device:
        raise RuntimeError("VGGT 需要 GPU。")

    outdir.mkdir(parents=True, exist_ok=True)

    # 2. 加载模型
    model = load_vggt_model(device, verbose=verbose)

    # 3. 推理
    preds = run_vggt(imgs, model, device)

    plot_cameras_from_predictions(
        predictions=preds,
        out_path=outdir / "camera_poses" / "camera_poses.png",
        axis_len=0.1,
        include_points=False,  # 想看点云就开
        center_mode="mean",  # 不以相机为原点，而是整体居中
    )

    # 4. 保存结果
    result_info = save_inference_results(
        preds=preds,
        outdir=outdir,
        conf_thres=conf_thres,
        prediction_mode=prediction_mode,
    )

    return result_info


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
    left_kpts, *_, left_frames = load_info(
        video_file_path=left_video_path.as_posix(), pt_file_path=left_pt_path.as_posix()
    )
    right_kpts, *_, right_frames = load_info(
        video_file_path=right_video_path.as_posix(),
        pt_file_path=right_pt_path.as_posix(),
    )

    all_frame_x3d = []

    for idx in tqdm(
        range(0, min(len(left_frames), len(right_frames))), desc="Processing frames"
    ):
        # if idx > 60: break

        # save images
        img_dir = out_dir / "frames" / f"frame_{idx:04d}" / "raw_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        write_png(left_frames[idx].permute(2, 0, 1), img_dir / f"left_{idx:04d}.png")
        write_png(right_frames[idx].permute(2, 0, 1), img_dir / f"right_{idx:04d}.png")

        # infer vggt
        result = reconstruct_from_frames(
            imgs=[
                left_frames[idx],
                right_frames[idx],
            ],
            outdir=out_dir / "frames" / f"frame_{idx:04d}" / "vggt_res",
            conf_thres=cfg.infer.get("conf_thres", 50.0),
            prediction_mode=cfg.infer.get(
                "prediction_mode", "Depthmap and Camera Branch"
            ),
            gpu=cfg.infer.get("gpu", 0),
            verbose=bool(cfg.runtime.get("verbose", True)),
        )

        logger.info(
            f"[OK-MV] {left_video_path.name} & {right_video_path.name} | "
            f"frames number={idx} | "
            f"npz={Path(result['npz_path']).name if 'npz_path' in result else 'NA'} | "
            f"glb={Path(result['glb_path']).name if 'glb_path' in result else 'NA'} | "
        )

        # run triangulation and reprojection
        # 这里是相机在世界坐标系下的 R,t,C
        camera_extrinsics = result["preds"]["extrinsic"]
        camera_intrinsics = result["preds"]["intrinsic"]
        R, t, C = extrinsic_to_RT(camera_extrinsics)

        camera_intrinsics_resized = []
        for i in range(len(camera_intrinsics)):
            camera_intrinsics_resized.append(
                scale_intrinsics(
                    camera_intrinsics[i],
                    orig_size=(294, 518),
                    new_size=(left_frames[idx].shape[0], left_frames[idx].shape[1]),
                )
            )

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
            save_dir=out_dir / "frames" / f"frame_{idx:04d}" / "triangulation",
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
        if reprojet_err.get("mean_err_L", np.inf) < cfg.infer.get(
            "reproj_err", 20
        ) and reprojet_err.get("mean_err_R", np.inf) < cfg.infer.get("reproj_err", 20):
            best_out_dir = out_dir / "best_frames"
            best_out_dir.mkdir(parents=True, exist_ok=True)

            # copy the results to best_frames

            shutil.copy(
                out_dir / f"frames/frame_{idx:04d}/triangulation/stereo_pose_frame.jpg",
                best_out_dir / f"frame_{idx:04d}.jpg",
            )

        all_frame_x3d.append(x3d)

    # merge all best frames into one video

    best_frames = sorted(best_out_dir.glob("*.jpg"))
    if len(best_frames) > 0:
        first_frame = cv2.imread(best_frames[0].as_posix())
        height, width, _ = first_frame.shape
        video_writer = cv2.VideoWriter(
            (out_dir / "best_frames_video.mp4").as_posix(),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (width, height),
        )
        for frame_path in best_frames:
            frame = cv2.imread(frame_path.as_posix())
            video_writer.write(frame)
        video_writer.release()

    # update 3d information to pt file
    update_pt_with_3d_info(
        left_pt_path=left_pt_path,
        right_pt_path=right_pt_path,
        all_frame_x3d=all_frame_x3d,
        out_pt_path=out_dir / "fuse_3d_pose.pt",
        reprojet_err=reprojet_err,
    )
    return out_dir


def extrinsic_to_RT(extrinsic):
    """
    extrinsic: (T,3,4) / (T,4,4) / (3,4) / (4,4)
    return: R (T,3,3), t (T,3), C (T,3)
    """
    E = np.asarray(extrinsic)

    # 扩展到 (T,3,4)
    if E.ndim == 2:  # 单帧
        E = E[None, ...]
    if E.shape[-2:] == (4, 4):
        E = E[:, :3, :]

    R = E[:, :3, :3]  # (T,3,3)
    t = E[:, :3, 3]  # (T,3)
    C = -np.einsum("tij,tj->ti", R.transpose(0, 2, 1), t)  # (T,3)

    return R, t, C


def process_single_view_video(
    video_path: Path,
    pt_path: Path,
    out_root: Path,
    cfg: DictConfig,
) -> Optional[Path]:
    """
    处理双目视频。返回输出目录；失败返回 None。

    目前示例代码仍然只对 left_video_path 进行 VGGT 推理，
    主要提供一个“成对管理 + 输出目录区分”的框架。
    后续如果需要真正 multi-view 融合，可以在这里扩展。
    """
    subject = video_path.parent.name or "default"

    out_dir = out_root / "single_view" / subject / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Run-SV] {video_path} → {out_dir} | ")

    # * load info from pt and video
    *_, frames = load_info(
        video_file_path=video_path.as_posix(), pt_file_path=pt_path.as_posix()
    )

    # * 处理前后3f
    for idx in tqdm(range(0, len(frames) - 1), desc="Processing frames"):
        # save images
        img_dir = out_dir / f"frame_{idx:04d}" / "raw_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        write_png(frames[idx].permute(2, 0, 1), img_dir / f"frame_{idx:04d}.png")
        write_png(
            frames[idx + 1].permute(2, 0, 1), img_dir / f"frame_{idx + 1:04d}.png"
        )

        result = reconstruct_from_frames(
            imgs=[
                frames[idx],
                frames[idx + 1],
            ],
            outdir=str(out_dir) + f"/frame_{idx:04d}",
            conf_thres=cfg.infer.get("conf_thres", 50.0),
            prediction_mode=cfg.infer.get(
                "prediction_mode", "Depthmap and Camera Branch"
            ),
            keep_frames=cfg.infer.get("keep_frames", True),
            export_ply=cfg.infer.get("export_ply", False),
            random_sample=cfg.infer.get("random_sample", None),
            gpu=cfg.infer.get("gpu", 0),
            verbose=bool(cfg.runtime.get("verbose", True)),
        )

        logger.info(
            f"[OK-SV] {video_path.name} | "
            f"frames={result.get('n_frames', 'NA')} | "
            f"npz={Path(result['npz_path']).name if 'npz_path' in result else 'NA'} | "
            f"glb={Path(result['glb_path']).name if 'glb_path' in result else 'NA'} | "
            f"time={result.get('time', 0.0):.2f}s"
        )

    return out_dir
