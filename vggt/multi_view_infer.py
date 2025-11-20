#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/vggt/single_view_infer copy.py
Project: /workspace/code/vggt
Created Date: Thursday November 20th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday November 20th 2025 5:47:43 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
vggt_video_infer.py
从单个视频抽帧并执行 VGGT 推理，可作为函数调用。
"""

import os
import cv2
import time
import shutil
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Optional

# 依赖 VGGT 官方模块
from vggt.visual_util import predictions_to_glb
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map

from vggt.camera_vis import (
    plot_cameras_matplotlib,
    plot_cameras_timeline,
)

import logging

logger = logging.getLogger(__name__)


# ==========================
# 工具函数
# ==========================
def _resize_keep_aspect(img, max_long_edge=None):
    if not max_long_edge:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m > max_long_edge:
        scale = max_long_edge / float(m)
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


def extract_frames(
    video_path: str,
    out_dir: str,
    mode: str = "uniform",
    fps: float = 1.0,
    every_k: int = None,
    uniform_frames: int = None,
    max_frames: int = None,
    max_long_edge: int = None,
    verbose: bool = False,
) -> List[str]:
    """按指定策略抽帧"""
    os.makedirs(out_dir, exist_ok=True)
    vs = cv2.VideoCapture(video_path)
    total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = vs.get(cv2.CAP_PROP_FPS) or 30.0
    out_paths = []

    if verbose:
        print(f"[Extract] {mode} sampling from {total} frames")

    if mode == "uniform" and uniform_frames:
        idxs = np.linspace(0, total - 1, num=uniform_frames, dtype=int)
    elif mode == "every_k" and every_k:
        idxs = np.arange(0, total, every_k, dtype=int)
    else:  # fps mode
        step = max(int(native_fps / max(fps, 1e-6)), 1)
        idxs = np.arange(0, total, step, dtype=int)

    for i, fidx in enumerate(tqdm(idxs, disable=not verbose, desc="Extracting")):
        vs.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = vs.read()
        if not ok:
            continue
        frame = _resize_keep_aspect(frame, max_long_edge)
        out_path = os.path.join(out_dir, f"{i:06d}.png")
        cv2.imwrite(out_path, frame)
        out_paths.append(out_path)
        if max_frames and len(out_paths) >= max_frames:
            break

    vs.release()
    if verbose:
        print(f"Extracted {len(out_paths)} frames → {out_dir}")
    return out_paths


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
    images: List[str], model, device="cuda", verbose=True
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


def save_ply(points: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


# ==========================
# 主函数接口
# ==========================
def reconstruct_from_video(
    video_path: str,
    outdir: str,
    mode: str = "uniform",
    fps: float = 1.0,
    every_k: Optional[int] = None,
    uniform_frames: Optional[int] = None,
    max_frames: Optional[int] = None,
    max_long_edge: Optional[int] = None,
    conf_thres: float = 50.0,
    voxel_size: float = 0.0,
    random_sample: Optional[int] = None,
    export_ply: bool = False,
    prediction_mode: str = "Depthmap and Camera Branch",
    keep_frames: bool = True,
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
    t0 = time.time()
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    if "cuda" not in device:
        raise RuntimeError("VGGT 需要 GPU。")

    img_dir = os.path.join(outdir, "images")
    os.makedirs(outdir, exist_ok=True)

    # 1. 抽帧
    imgs = extract_frames(
        video_path,
        img_dir,
        mode=mode,
        fps=fps,
        every_k=every_k,
        uniform_frames=uniform_frames,
        max_frames=max_frames,
        max_long_edge=max_long_edge,
        verbose=verbose,
    )

    # 2. 加载模型
    model = load_vggt_model(device, verbose=verbose)

    # 3. 推理
    preds = run_vggt(imgs, model, device, verbose=verbose)

    # * draw camera frustums (optional)
    plot_cameras_matplotlib(
        preds,
        out_dir=os.path.join(outdir, "camera_poses"),
        axis_len=0.1,
        title="Estimated Camera Poses",
    )

    plot_cameras_timeline(
        preds,
        out_path=os.path.join(outdir, "camera_poses/cameras_timeline_x.png"),
        dx=1,
        timeline_axis="x",
        wrap=len(imgs),
        axis_len=10,
    )

    # 4. 保存 npz
    npz_path = os.path.join(outdir, "predictions.npz")
    np.savez(npz_path, **preds)

    # 5. 导出 glb
    glb_path = os.path.join(
        outdir, f"scene_conf{conf_thres}_mode{prediction_mode.replace(' ', '_')}.glb"
    )
    glb = predictions_to_glb(
        preds,
        conf_thres=conf_thres,
        filter_by_frames="All",
        show_cam=True,
        mask_black_bg=False,
        mask_white_bg=False,
        mask_sky=False,
        target_dir=outdir,
        prediction_mode=prediction_mode,
    )
    glb.export(file_obj=glb_path)
    print(f"Saved GLB → {glb_path}")

    # 6. 导出 ply（可选）
    ply_path = None
    if export_ply:
        pts = preds["world_points_from_depth"].reshape(-1, 3)
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if random_sample and pts.shape[0] > random_sample:
            idx = np.random.choice(len(pts), random_sample, replace=False)
            pts = pts[idx]
        ply_path = os.path.join(outdir, "pointcloud.ply")
        save_ply(pts, ply_path)

    if not keep_frames:
        shutil.rmtree(img_dir, ignore_errors=True)

    return dict(
        npz_path=npz_path,
        glb_path=glb_path,
        ply_path=ply_path,
        n_frames=len(imgs),
        time=time.time() - t0,
    )
