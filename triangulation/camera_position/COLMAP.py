#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import tempfile

import numpy as np
import pycolmap


@dataclass
class PoseRT:
    name: str
    cam_id: int
    # world -> camera
    R_wc: np.ndarray  # (3,3)
    t_wc: np.ndarray  # (3,)
    # camera -> world
    R_cw: np.ndarray  # (3,3)
    t_cw: np.ndarray  # (3,)
    C: np.ndarray     # (3,) camera center in world  == -R^T t


def _to_cam_params_value(v: Optional[List[float] | np.ndarray]) -> Optional[object]:
    """
    pycolmap 不同版本对 ImageReaderOptions.camera_params 支持 list 或逗号字符串。
    这里统一做兼容：优先给 list，失败时再给字符串。
    """
    if v is None:
        return None
    arr = np.asarray(v, dtype=float).tolist()
    return arr  # 先用 list，失败再兜底为 "1,2,3,..."（见下方 try/except）


def _extract_and_match(
    db_path: Path,
    image_dir: Path,
    *,
    mask_dir: Optional[Path] = None,
    single_camera: bool = True,
    camera_model: Optional[str] = None,
    camera_params: Optional[List[float]] = None,
    matcher: str = "sequential",
    device: pycolmap.Device = pycolmap.Device.auto,
) -> None:
    # ---- 1) 特征提取 ----
    iro = pycolmap.ImageReaderOptions()
    if camera_model:
        iro.camera_model = str(camera_model)
    cam_param_val = _to_cam_params_value(camera_params)
    if cam_param_val is not None:
        try:
            iro.camera_params = cam_param_val               # 有的版本接受 list
        except Exception:
            iro.camera_params = ",".join(map(str, cam_param_val))  # 兜底为字符串
    if mask_dir:
        iro.mask_path = str(mask_dir)

    sift_ext = pycolmap.SiftExtractionOptions()

    # 有的版本才有 CameraMode
    camera_mode = None
    try:
        camera_mode = (
            pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO
        )
    except Exception:
        pass

    # 兼容不同版本的关键字：reader_options / image_reader_options
    kwargs_common = dict(
        database_path=str(db_path),
        image_path=str(image_dir),
        sift_options=sift_ext,
        device=device,
    )
    if camera_mode is not None:
        kwargs_common["camera_mode"] = camera_mode

    called = False
    try:
        pycolmap.extract_features(reader_options=iro, **kwargs_common)
        called = True
    except TypeError:
        # 老版本
        pycolmap.extract_features(image_reader_options=iro, **kwargs_common)
        called = True
    if not called:
        raise RuntimeError("pycolmap.extract_features failed to dispatch.")

    # ---- 2) 匹配 ----
    sift_match = pycolmap.SiftMatchingOptions()
    verify_opt = pycolmap.TwoViewGeometryOptions()

    if matcher == "sequential":
        seq_opt = pycolmap.SequentialMatchingOptions()
        seq_opt.overlap = 2
        # 关键：关闭回环，避免 vocab-tree 在线下载导致崩溃
        seq_opt.loop_detection = False

        pycolmap.match_sequential(
            database_path=str(db_path),
            sift_options=sift_match,
            matching_options=seq_opt,
            verification_options=verify_opt,
            device=device,
        )
    elif matcher == "exhaustive":
        ex_opt = pycolmap.ExhaustiveMatchingOptions()
        pycolmap.match_exhaustive(
            database_path=str(db_path),
            sift_options=sift_match,
            matching_options=ex_opt,
            verification_options=verify_opt,
            device=device,
        )
    else:
        raise ValueError(f"Unknown matcher: {matcher}")


def estimate_Rt_with_colmap(
    image_dir: str | Path,
    *,
    mask_dir: Optional[str | Path] = None,
    single_camera: bool = True,
    camera_model: Optional[str] = None,
    camera_params: Optional[List[float]] = None,
    matcher: str = "sequential",
    device: pycolmap.Device = pycolmap.Device.auto,
) -> Tuple[Dict[str, PoseRT], Dict[int, dict]]:
    """
    仅返回位姿与内参；全部在临时目录中完成，不保留任何文件。
    返回:
      poses: {image_name: PoseRT(...)}  （world->cam 的 R_wc,t_wc；以及 cam->world 的 R_cw,t_cw 与 C）
      intrinsics: {camera_id: {"model","width","height","params"}}
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    mdir = Path(mask_dir) if mask_dir else None

    with tempfile.TemporaryDirectory(prefix="pycolmap_tmp_") as tmp:
        tmp = Path(tmp)
        db_path = tmp / "database.db"
        sparse_dir = tmp / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # 1) 特征与匹配
        _extract_and_match(
            db_path=db_path,
            image_dir=image_dir,
            mask_dir=mdir,
            single_camera=single_camera,
            camera_model=camera_model,
            camera_params=camera_params,
            matcher=matcher,
            device=device,
        )

        # 2) 增量式重建
        rec_map = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(image_dir),
            output_path=str(sparse_dir),
        )
        if not rec_map:
            raise RuntimeError("COLMAP failed: no reconstruction produced.")

        # 取注册张数最多的模型
        best_id = max(rec_map, key=lambda k: rec_map[k].num_reg_images())
        rec = rec_map[best_id]

        # 3) 汇总位姿（pycolmap 的 qvec/tvec 为 world->cam）
        poses: Dict[str, PoseRT] = {}
        for img_id, im in rec.images.items():
            q = np.asarray(im.qvec, dtype=np.float64)
            t = np.asarray(im.tvec, dtype=np.float64)
            R_wc = pycolmap.qvec_to_rotmat(q)  # world->cam
            t_wc = t
            R_cw = R_wc.T
            C = -R_cw @ t_wc              # camera center in world
            t_cw = C                      # 等价于 -R^T t

            poses[im.name] = PoseRT(
                name=im.name,
                cam_id=im.cam_id,
                R_wc=R_wc,
                t_wc=t_wc,
                R_cw=R_cw,
                t_cw=t_cw,
                C=C,
            )

        # 4) 相机内参
        intrinsics: Dict[int, dict] = {}
        for cam_id, cam in rec.cameras.items():
            intrinsics[cam_id] = dict(
                model=str(cam.model),
                width=int(cam.width),
                height=int(cam.height),
                params=np.asarray(cam.params, dtype=np.float64),
            )

        return poses, intrinsics


if __name__ == "__main__":
    poses, intrinsics = estimate_Rt_with_colmap(
        image_dir="/workspace/data/vis/d2/run_3/filter_img/pose/osmo_1",
        # mask_dir="/workspace/data/colmap/masks",
        matcher="exhaustive",         # 或 "exhaustive"
        single_camera=False,          # 两机不共享内参就 False
        # camera_model="OPENCV",
        # camera_params=[fx, fy, cx, cy, k1, k2, p1, p2, k3],
    )

    # 使用示例
    some_name = next(iter(poses.keys()))
    R = poses[some_name].R_wc
    t = poses[some_name].t_wc
    C = poses[some_name].C
    print(some_name, "\nR=\n", R, "\nt=", t, "\nC=", C)
