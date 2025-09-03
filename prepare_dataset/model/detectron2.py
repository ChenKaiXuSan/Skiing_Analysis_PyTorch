# d2_infer.py
import os
from pathlib import Path
from typing import Optional, Tuple, List, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from prepare_dataset.utils import process_none


# ---------- 1) 构建 cfg / predictor ----------
def build_cfg_and_predictor(
    cfg_path_or_zoo: str,
    score_thresh: float = 0.7,
    device: str = "cuda",
    weights_url_or_path: Optional[str] = None,
):
    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path_or_zoo))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = weights_url_or_path or model_zoo.get_checkpoint_url(
        cfg_path_or_zoo
    )
    # 明确输入格式，避免颜色错位
    cfg.INPUT.FORMAT = "BGR"
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


# ---------- 工具：把帧变成 numpy BGR uint8 ----------
def to_numpy_bgr(
    frame: Union[np.ndarray, torch.Tensor], assume_rgb: bool = False
) -> np.ndarray:
    """
    支持:
      - torch.Tensor [H,W,3] 或 [3,H,W]，float(0..1/255) 或 uint8
      - np.ndarray   [H,W,3]
    返回: np.uint8 BGR
    """
    if isinstance(frame, torch.Tensor):
        f = frame.detach().cpu()
        if f.ndim == 3 and f.shape[0] in (1, 3):  # [C,H,W] -> [H,W,C]
            f = f.permute(1, 2, 0)
        arr = f.numpy()
    elif isinstance(frame, np.ndarray):
        arr = frame
    else:
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    # 标准化到 uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)

    # 根据标志转换颜色
    if assume_rgb:
        arr = arr[:, :, ::-1]  # RGB->BGR
    return arr


# ---------- 2) 单帧推理 ----------
def infer_instances(predictor: DefaultPredictor, image_bgr) -> "Instances":
    img = to_numpy_bgr(image_bgr)
    outputs = predictor(img)
    return outputs["instances"].to("cpu")


# ---------- 3) 提取 bbox / kpt 数组 ----------
def instances_to_arrays(instances) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
      bboxes    : [N,5]  -> x1,y1,x2,y2,score (float32)
      kpts_xy   : [N,K,2]  (float32)
      kpts_conf : [N,K]    (float32)
    若无对应内容返回空数组。
    """
    # boxes
    if hasattr(instances, "pred_boxes") and len(instances) > 0:
        xyxy = instances.pred_boxes.tensor.numpy().astype(np.float32)  # [N,4]
        scores = instances.scores.numpy().astype(np.float32)[:, None]  # [N,1]
        bboxes = np.concatenate([xyxy, scores], axis=1)  # [N,5]
    else:
        bboxes = np.zeros((0, 5), dtype=np.float32)

    # keypoints
    if hasattr(instances, "pred_keypoints") and len(instances) > 0:
        kpts = instances.pred_keypoints.numpy().astype(np.float32)  # [N,K,3] (x,y,prob)
        kpts_xy = kpts[..., :2]  # [N,K,2]
        kpts_conf = kpts[..., 2]  # [N,K]
    else:
        kpts_xy = np.zeros((0, 0, 2), dtype=np.float32)
        kpts_conf = np.zeros((0, 0), dtype=np.float32)

    return bboxes, kpts_xy, kpts_conf


# ---------- 4) 可视化（可选） ----------
def visualize_instances(
    image_bgr: Union[np.ndarray, torch.Tensor],
    instances,
    cfg,
    dataset_name_fallback: Optional[str] = "coco_2017_train",
    scale: float = 1.0,
) -> np.ndarray:
    """返回可视化后的 BGR 图像"""
    img_bgr = to_numpy_bgr(image_bgr)
    img_rgb = img_bgr[:, :, ::-1]

    # Visualizer 需要 metadata；如果 cfg.DATASETS.TRAIN 为空，用 fallback
    if len(cfg.DATASETS.TRAIN) > 0:
        md = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    else:
        if (
            dataset_name_fallback
            and dataset_name_fallback not in MetadataCatalog.list()
        ):
            MetadataCatalog.get(dataset_name_fallback)
        md = MetadataCatalog.get(dataset_name_fallback)

    v = Visualizer(img_rgb, metadata=md, scale=scale, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(instances)
    vis_rgb = out.get_image()
    return vis_rgb[:, :, ::-1]  # -> BGR


def _center_xyxy(xyxy: np.ndarray) -> np.ndarray:
    """
    xyxy: [...,4] -> 返回中心 [...,2]
    """
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return np.stack([cx, cy], axis=-1)


def _pick_person_by_prev_center(
    bboxes_xyxy: np.ndarray, prev_center_xy: np.ndarray
) -> int:
    """
    从多个 bbox 中选择与上一帧中心最近的 index
    bboxes_xyxy: [N,4], prev_center_xy: [2]
    """
    if bboxes_xyxy.shape[0] == 1:
        return 0
    centers = _center_xyxy(bboxes_xyxy)  # [N,2]
    d2 = np.sum((centers - prev_center_xy[None, :]) ** 2, axis=1)  # [N]
    return int(np.argmin(d2))


class Detectron2Wrapper:
    def __init__(self, config):
        self.cfg, self.predictor = build_cfg_and_predictor(
            config.detectron2.cfg,
            getattr(config, "score_thresh", 0.7),
            getattr(config, "device", "cuda"),
            getattr(config, "weights", None),
        )
        root = Path(config.extract_dataset.save_path)
        self.save_dir = root / "vis" / "d2"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_vis = bool(getattr(config, "vis", True))
        self.vis_stride = int(getattr(config, "vis_stride", 1))
        self.assume_rgb = bool(getattr(config, "assume_rgb", False))

    def __call__(self, vframes: Union[torch.Tensor, np.ndarray, List], video_path: str):
        """
        返回:
          pose:          [T, K, 2]  (float32)   — 所选人的关键点坐标
          pose_score:    [T, K]     (float32)   — 所选人的关键点置信度
          bboxes_xyxy:   [T, 4]     (float32)   — 所选人的 bbox (x1,y1,x2,y2)
          none_index:    List[int]             — 未检出帧索引（NaN 占位）
        """
        stem = Path(video_path).stem
        out_dir = self.save_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        iterable = vframes

        poses_xy_list: List[np.ndarray] = []  # 每帧 [K,2] 或 NaN
        poses_sc_list: List[np.ndarray] = []  # 每帧 [K]   或 NaN
        bboxes_list: List[np.ndarray] = []  # 每帧 [4]   或 NaN
        none_index: List[int] = []

        prev_center = None  # 上一帧选中人的 bbox 中心

        for idx, frame in tqdm(
            enumerate(iterable), total=len(iterable), desc="Detectron2 Inference"
        ):
            img = to_numpy_bgr(frame, assume_rgb=self.assume_rgb)
            instances = infer_instances(self.predictor, img)

            # 当前帧的全部候选
            bboxes, kpts_xy_all, kpts_conf_all = instances_to_arrays(
                instances
            )  # bboxes [N,5], kpts [N,K,2], conf [N,K]
            N = bboxes.shape[0]

            if idx == 0:
                if N == 0:
                    # 首帧缺失
                    none_index.append(idx)
                    poses_xy_list.append(np.full((0, 2), np.nan, dtype=np.float32))
                    poses_sc_list.append(np.full((0,), np.nan, dtype=np.float32))
                    bboxes_list.append(np.full((4,), np.nan, dtype=np.float32))
                    prev_center = None
                else:
                    # 置信度最高
                    pick = int(np.argmax(bboxes[:, 4])) if N > 1 else 0
                    xyxy = bboxes[pick, :4][None, :]  # [1,4]
                    prev_center = _center_xyxy(xyxy)[0]  # [2]
                    poses_xy_list.append(kpts_xy_all[pick])  # [K,2]
                    poses_sc_list.append(kpts_conf_all[pick])  # [K]
                    bboxes_list.append(bboxes[pick, :4])  # [4]
            else:
                if N == 0:
                    none_index.append(idx)
                    # 后面会按 K pad；bbox 直接 NaN
                    poses_xy_list.append(np.full_like(poses_xy_list[0], np.nan))
                    poses_sc_list.append(np.full_like(poses_sc_list[0], np.nan))
                    bboxes_list.append(np.full((4,), np.nan, dtype=np.float32))
                    # prev_center 不更新
                else:
                    if prev_center is None:
                        pick = int(np.argmax(bboxes[:, 4])) if N > 1 else 0
                    else:
                        pick = _pick_person_by_prev_center(bboxes[:, :4], prev_center)
                    xyxy = bboxes[pick, :4][None, :]
                    prev_center = _center_xyxy(xyxy)[0]
                    poses_xy_list.append(kpts_xy_all[pick])
                    poses_sc_list.append(kpts_conf_all[pick])
                    bboxes_list.append(bboxes[pick, :4])

            # 可视化（抽帧）
            if self.save_vis and (idx % self.vis_stride == 0):
                vis = visualize_instances(
                    img, instances, self.cfg, dataset_name_fallback="coco_2017_train"
                )
                cv2.imwrite(str(out_dir / f"frame_{idx:04d}.jpg"), vis)

        # —— 统一 K 大小（找一个非空帧确定 K）——
        K = None
        for arr in poses_xy_list:
            if arr.size != 0 and not np.isnan(arr).all():
                K = arr.shape[0]
                break
        if K is None:
            # 全部缺失
            T = len(poses_xy_list)
            return (
                torch.empty((T, 0, 2), dtype=torch.float32),
                torch.empty((T, 0), dtype=torch.float32),
                torch.full((T, 4), np.nan, dtype=torch.float32),
                none_index,
            )

        # pad 到 (T,K,2) / (T,K)
        def _pad_xy(a):
            if a.size == 0 or np.isnan(a).all():
                return np.full((K, 2), np.nan, dtype=np.float32)
            if a.shape[0] == K:
                return a.astype(np.float32)
            out = np.full((K, 2), np.nan, dtype=np.float32)
            m = min(K, a.shape[0])
            out[:m] = a[:m]
            return out

        def _pad_sc(s):
            if s.size == 0 or np.isnan(s).all():
                return np.full((K,), np.nan, dtype=np.float32)
            if s.shape[0] == K:
                return s.astype(np.float32)
            out = np.full((K,), np.nan, dtype=np.float32)
            m = min(K, s.shape[0])
            out[:m] = s[:m]
            return out

        poses_xy_arr = np.stack([_pad_xy(a) for a in poses_xy_list], axis=0)  # [T,K,2]
        poses_sc_arr = np.stack([_pad_sc(s) for s in poses_sc_list], axis=0)  # [T,K]
        bboxes_arr = np.stack(
            [
                b if b.shape == (4,) else np.full((4,), np.nan, dtype=np.float32)
                for b in bboxes_list
            ],
            axis=0,
        )  # [T,4]

        # 如果你希望也对 bbox 做缺失插补（例如线性插值），可在此调用自定义函数：
        # bboxes_arr = interpolate_bboxes(bboxes_arr)  # 可选

        pose = torch.from_numpy(poses_xy_arr)  # [T,K,2]
        pose_score = torch.from_numpy(poses_sc_arr)  # [T,K]
        bboxes_xyxy = torch.from_numpy(bboxes_arr)  # [T,4]
        return pose, pose_score, bboxes_xyxy, none_index
