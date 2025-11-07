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
import traceback
import matplotlib.pyplot as plt

# 依赖 VGGT 官方模块
from vggt.visual_util import predictions_to_glb
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map

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


def extract_frames(video_path: str, out_dir: str,
                   mode: str = "uniform",
                   fps: float = 1.0,
                   every_k: int = None,
                   uniform_frames: int = None,
                   max_frames: int = None,
                   max_long_edge: int = None,
                   verbose: bool = False) -> List[str]:
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
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=verbose)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


@torch.no_grad()
def run_vggt(images: List[str], model, device="cuda", verbose=True) -> Dict[str, np.ndarray]:
    """对图像列表执行 VGGT 推理"""
    imgs = load_and_preprocess_images(images).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.cuda.amp.autocast(dtype=dtype):
        preds = model(imgs)

    H, W = imgs.shape[-2:]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
    preds["extrinsic"] = extrinsic
    preds["intrinsic"] = intrinsic

    # 转 numpy
    out = {k: (v.detach().cpu().numpy().squeeze(0)
               if isinstance(v, torch.Tensor) else v)
           for k, v in preds.items()}
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

def plot_cameras_matplotlib(preds,
                            out_dir=".",
                            axis_len=0.1,
                            title="Camera Poses",
                            show_id=True,
                            include_points=False):
    """
    可视化相机位置并保存三个视角的图片（正视、俯视、侧视）
    ------------------------------------------------------------
    Args:
        preds: dict，包含 "extrinsic" (S,3,4) 或 (S,4,4)
        out_dir: 输出文件夹路径
        axis_len: 每个相机坐标轴的长度
        title: 图标题
        show_id: 是否显示相机编号
        include_points: 是否绘制世界点云 (preds["world_points_from_depth"])
    """
    os.makedirs(out_dir, exist_ok=True)

    E = preds["extrinsic"]
    if E.shape[-2:] == (3, 4):
        R = E[..., :3, :3]
        t = E[..., :3, 3]
    else:
        R = E[..., :3, :3]
        t = E[..., :3, 3]

    # 相机中心（世界坐标系）
    C = -np.einsum("sij,sj->si", R.transpose(0, 2, 1), t)

    # 世界点云（可选）
    if include_points and "world_points_from_depth" in preds:
        pts = preds["world_points_from_depth"].reshape(-1, 3)
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
    else:
        pts = None

    def _plot_one(ax, elev, azim, view_name):
        ax.cla()
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=20, c="tab:blue", label="Cameras")

        # 绘制相机坐标轴
        Xw = np.einsum("sij,j->si", R.transpose(0, 2, 1), np.array([1, 0, 0]))
        Yw = np.einsum("sij,j->si", R.transpose(0, 2, 1), np.array([0, 1, 0]))
        Zw = np.einsum("sij,j->si", R.transpose(0, 2, 1), np.array([0, 0, 1]))

        for i in range(C.shape[0]):
            o = C[i]
            ax.plot([o[0], o[0] + axis_len * Xw[i, 0]],
                    [o[1], o[1] + axis_len * Xw[i, 1]],
                    [o[2], o[2] + axis_len * Xw[i, 2]], 'r', lw=1)
            ax.plot([o[0], o[0] + axis_len * Yw[i, 0]],
                    [o[1], o[1] + axis_len * Yw[i, 1]],
                    [o[2], o[2] + axis_len * Yw[i, 2]], 'g', lw=1)
            ax.plot([o[0], o[0] + axis_len * Zw[i, 0]],
                    [o[1], o[1] + axis_len * Zw[i, 1]],
                    [o[2], o[2] + axis_len * Zw[i, 2]], 'b', lw=1)
            if show_id:
                ax.text(o[0], o[1], o[2],
                        f"{i}", color="black", fontsize=9,
                        ha='center', va='bottom', weight='bold')

        if pts is not None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=1, c="gray", alpha=0.3, label="Points")

        ax.set_title(f"{title} ({view_name})")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)

    # 创建画布
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # === 视角配置 ===
    views = [
        dict(name="default", elev=30, azim=60, bg="#ffffff"),
        dict(name="front", elev=0, azim=90, bg="#f0f7ff"),
        dict(name="top", elev=90, azim=-90, bg="#fff8f0"),
        dict(name="side", elev=0, azim=0, bg="#f8fff0"),
    ]

    for view in views:
        _plot_one(ax, view["elev"], view["azim"], view["name"])
        out_path = os.path.join(out_dir, f"cameras_{view['name']}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"[Saved] {out_path}")

    plt.close()

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("VGGT 需要 GPU。")

    img_dir = os.path.join(outdir, "images")
    os.makedirs(outdir, exist_ok=True)

    # 1. 抽帧
    imgs = extract_frames(
        video_path, img_dir,
        mode=mode, fps=fps, every_k=every_k, uniform_frames=uniform_frames,
        max_frames=max_frames, max_long_edge=max_long_edge, verbose=verbose
    )

    # 2. 加载模型
    model = load_vggt_model(device, verbose=verbose)

    # 3. 推理
    preds = run_vggt(imgs, model, device, verbose=verbose)

    # * draw camera frustums (optional)
    cam_png = os.path.join(outdir, "camera_poses")
    plot_cameras_matplotlib(preds, out_dir=cam_png, axis_len=0.1, title="Estimated Camera Poses")

    # 4. 保存 npz
    npz_path = os.path.join(outdir, "predictions.npz")
    np.savez(npz_path, **preds)

    # 5. 导出 glb
    glb_path = os.path.join(outdir, f"scene_conf{conf_thres}_mode{prediction_mode.replace(' ', '_')}.glb")
    glb = predictions_to_glb(
        preds, conf_thres=conf_thres, filter_by_frames="All",
        show_cam=True, mask_black_bg=False, mask_white_bg=False,
        mask_sky=False, target_dir=outdir, prediction_mode=prediction_mode
    )
    glb.export(file_obj=glb_path)
    print(f"Saved GLB → {glb_path}")

    # 5.1 保存 PNG 预览
    try:
        preview_path = os.path.join(outdir, "scene_preview.png")

        try:
            # 优先使用 trimesh 的 pyglet 渲染
            png_bytes = glb.save_image(resolution=(1024, 768), visible=True)
            with open(preview_path, "wb") as f:
                f.write(png_bytes)
            print(f"[Info] Saved GLB preview → {preview_path}")
        
        except Exception as inner_e:
            # 如果 pyglet 不可用，自动退回到 matplotlib 绘制
            print(f"[Warning] pyglet backend failed ({inner_e}), using matplotlib fallback.")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            pts = preds["world_points_from_depth"].reshape(-1, 3)
            mask = np.isfinite(pts).all(axis=1)
            pts = pts[mask]
            if pts.shape[0] > 200000:
                # 避免点太多渲染太慢
                idx = np.random.choice(len(pts), 200000, replace=False)
                pts = pts[idx]

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c='blue', alpha=0.4)
            ax.set_title("VGGT 3D Reconstruction Preview")
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(preview_path, dpi=200)
            plt.close(fig)
            logger.info(f"Saved fallback preview → {preview_path}")

    except Exception as e:
        logger.warning(f"Failed to save PNG preview: {e}")
        traceback.print_exc()

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
        time=time.time() - t0
    )