#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/prepare_dataset/main_chunk.py
Project: /workspace/code/prepare_dataset
Created Date: Friday September 5th 2025
Author: Kaixu Chen
-----
Comment:
这里为了防止RAM不够造成的OOM问题，采用分块处理视频的方式
但是由于preprocess只接受整段视频进行处理，所以这里的分块处理是为了在内存有限的情况下，尽量保证处理的连续性和完整性。
问题是保存的推理结果只有当前块的结果，无法保证跨块的连续性。

Have a good code time :)
-----
Last Modified: Friday September 5th 2025 10:34:45 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

import hydra
import torch
from torchvision.io import VideoReader, read_video

from prepare_dataset.process.preprocess import Preprocess

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


# --------------------------- 小工具 ---------------------------


def _iter_videos(root: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        yield from (p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    else:
        yield from (p for p in root.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def _safe_save_pt(pt_path: Path, obj: Dict[str, Any], legacy: bool = True) -> None:
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pt_path.with_suffix(pt_path.suffix + ".tmp")
    torch.save(obj, tmp, _use_new_zipfile_serialization=not legacy)
    os.replace(tmp, pt_path)  # 原子替换
    logger.info(
        "Saved pt -> %s (%.1f MB)", pt_path, os.path.getsize(pt_path) / 1024 / 1024
    )


def _target_pt_path(save_root: Path, person: str, video_stem: str) -> Path:
    return save_root / "pt" / person / f"{video_stem}.pt"


def _cat_safe(a: torch.Tensor, b: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if a.numel() == 0:
        return b
    if b.numel() == 0:
        return a
    return torch.cat([a, b], dim=dim)


def _yolo_cat(
    dst: Dict[str, torch.Tensor], src: Dict[str, torch.Tensor], drop_first_frame: bool
) -> Dict[str, torch.Tensor]:
    sl = slice(1, None) if drop_first_frame else slice(None)
    return {
        "bbox": _cat_safe(dst["bbox"], src["bbox"][sl]),
        "mask": _cat_safe(dst["mask"], src["mask"][sl]),
        "keypoints": _cat_safe(dst["keypoints"], src["keypoints"][sl]),
        "keypoints_score": _cat_safe(
            dst["keypoints_score"], src["keypoints_score"][sl]
        ),
    }


def _d2_cat(
    dst: Dict[str, torch.Tensor], src: Dict[str, torch.Tensor], drop_first_frame: bool
) -> Dict[str, torch.Tensor]:
    sl = slice(1, None) if drop_first_frame else slice(None)
    return {
        "bbox": _cat_safe(dst["bbox"], src["bbox"][sl]),
        "keypoints": _cat_safe(dst["keypoints"], src["keypoints"][sl]),
        "keypoints_score": _cat_safe(
            dst["keypoints_score"], src["keypoints_score"][sl]
        ),
    }


# --------------------------- 核心：单视频流式分块处理 ---------------------------


def process_video_chunked(
    config,
    person: str,
    video_path: Path,
    *,
    chunk_size: int = 128,  # 每块帧数（除第一块外实际读取 N+1，含 1 帧重叠）
) -> Dict[str, Any]:
    """
    以低峰值内存的方式处理单个视频：
      - 使用 VideoReader 按块读取
      - 每块前置 1 帧重叠，保证光流跨块连续，且跟踪更稳定
      - 对于逐帧量（depth/yolo/d2）在后续块丢弃第 0 帧的重复结果
      - 对于光流（T-1,2,H,W）保留整块结果（包含跨块第一对）
    """
    preprocess = Preprocess(config=config, person=person)

    # 读取元信息（容错：VideoReader 不可用时退回 read_video）
    fps = 0.0
    duration = 0.0
    try:
        vr = VideoReader(str(video_path), "video")
        meta = vr.get_metadata()
        fps = float(meta.get("video", {}).get("fps", [0.0])[0] or 0.0)
        duration = float(meta.get("video", {}).get("duration", [0.0])[0] or 0.0)
    except Exception as e:
        logger.warning(
            "VideoReader init failed (%s), fallback to read_video meta: %s",
            video_path.name,
            e,
        )
        _, _, info = read_video(video_path, pts_unit="sec", output_format="THWC")
        fps = float(info.get("video_fps", 0.0))
        duration = float(info.get("duration", 0.0))
        vr = VideoReader(str(video_path), "video")  # 再试一次用于迭代

    vr.set_current_stream("video")

    # 累积结果容器（全部放 CPU，避免占 GPU）
    total_T = 0
    first_chunk = True

    acc_depth = torch.empty((0, 1, 0, 0))  # 将以正确 H,W 替换
    acc_flow = torch.empty((0, 2, 0, 0))
    acc_yolo = {
        "bbox": torch.empty((0, 4)),
        "mask": torch.empty((0, 1, 0, 0)),
        "keypoints": torch.empty((0, 17, 3)),
        "keypoints_score": torch.empty((0, 17)),
    }
    acc_d2 = {
        "bbox": torch.empty((0, 4)),
        "keypoints": torch.empty((0, 17, 3)),
        "keypoints_score": torch.empty((0, 17)),
    }
    acc_none_idx: List[int] = []

    last_frame: Optional[torch.Tensor] = None  # THW C=3 (uint8)

    with torch.inference_mode():
        while True:
            buf: List[torch.Tensor] = []

            if not first_chunk and last_frame is not None:
                buf.append(last_frame)  # 重叠帧

            # 读取 chunk_size 帧
            for _ in range(chunk_size):
                try:
                    fr = next(vr)
                except StopIteration:
                    break
                # fr['data'] 是 (C,H,W) uint8，转成 (H,W,C)
                buf.append(fr["data"].permute(1, 2, 0).contiguous())

            if len(buf) == 0 or (not first_chunk and len(buf) == 1):
                break  # 没有新帧

            # 记录下一块要用的重叠帧
            last_frame = buf[-1].clone()

            vframes = torch.stack(buf, dim=0)  # (t,H,W,C) uint8 on CPU

            # 推理
            out = preprocess(vframes, video_path)

            # 附加 none_index（注意索引平移）
            local_none = list(out.get("none_index", []))
            if first_chunk:
                # 第一块：直接平移到全局（相对 0）
                acc_none_idx.extend(local_none)
            else:
                # 后续块：丢弃帧 0 的结果，所以本地帧 i(>=1) 对应全局帧 (total_T + i - 1)
                acc_none_idx.extend([total_T + i - 1 for i in local_none if i >= 1])

            # 取尺寸
            t, h, w, _ = vframes.shape

            # 初始化 acc 的 H,W（首次）
            if first_chunk:
                acc_depth = torch.empty((0, 1, h, w))
                acc_flow = torch.empty((0, 2, h, w))
                acc_yolo["mask"] = torch.empty((0, 1, h, w))
                acc_d2["keypoints"] = torch.empty((0, 17, 3))
                acc_yolo["keypoints"] = torch.empty((0, 17, 3))

            # 拼接 depth（逐帧量，后续块丢 index 0）
            dep = out["depth"]  # (t,1,h,w) 或 (0,1,h,w)
            if dep.numel() > 0:
                dep_use = dep if first_chunk else dep[1:]
                acc_depth = _cat_safe(acc_depth, dep_use)

            # 拼接 optical flow（全保留，包含跨块第一对）
            flo = out["optical_flow"]  # (t-1,2,h,w) 或 (0,2,h,w)
            if flo.numel() > 0:
                acc_flow = _cat_safe(acc_flow, flo)

            # YOLO
            y = out["YOLO"]
            acc_yolo = _yolo_cat(acc_yolo, y, drop_first_frame=not first_chunk)

            # D2
            d2 = out["detectron2"]
            acc_d2 = _d2_cat(acc_d2, d2, drop_first_frame=not first_chunk)

            # 全局帧数累加（逐帧量新增 = 本地 t - (0/1)）
            total_T += t if first_chunk else (t - 1)
            first_chunk = False

    # 打包 pt_info（保持与你原先结构一致）
    pt_info: Dict[str, Any] = {
        "optical_flow": acc_flow.contiguous(),  # (T-1,2,H,W)
        "depth": acc_depth.contiguous(),  # (T,1,H,W)
        "none_index": acc_none_idx,
        "YOLO": {
            "bbox": acc_yolo["bbox"].contiguous(),  # (T,4)
            "mask": acc_yolo["mask"].contiguous(),  # (T,1,H,W)
            "keypoints": acc_yolo["keypoints"].contiguous(),  # (T,17,3)
            "keypoints_score": acc_yolo["keypoints_score"].contiguous(),  # (T,17)
        },
        "detectron2": {
            "bbox": acc_d2["bbox"].contiguous(),
            "keypoints": acc_d2["keypoints"].contiguous(),
            "keypoints_score": acc_d2["keypoints_score"].contiguous(),
        },
        # 元数据
        "video_name": video_path.stem,
        "video_path": str(video_path),
        "frame_count": total_T,
        "img_shape": (
            int(acc_depth.shape[-2]) if acc_depth.numel() else None,
            int(acc_depth.shape[-1]) if acc_depth.numel() else None,
        ),
        "fps": fps,
        "duration": duration,
    }

    return pt_info


# --------------------------- 按人处理 & Hydra 入口 ---------------------------


def process_one_person(config, person: str) -> None:
    raw_root = Path(config.extract_dataset.data_path)
    save_root = Path(config.extract_dataset.save_path)

    person_dir = raw_root / person
    if not person_dir.exists():
        logger.warning("Person dir not found: %s", person_dir)
        return

    overwrite: bool = bool(getattr(config.extract_dataset, "overwrite", False))
    recursive: bool = bool(getattr(config.extract_dataset, "recursive", False))

    chunk_size: int = int(
        getattr(getattr(config, "system", object()), "chunk_size", 128)
    )
    num_threads: int = (
        int(getattr(getattr(config, "system", object()), "num_threads", 0)) or 0
    )
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        logger.info("Set torch.set_num_threads(%d)", num_threads)

    logger.info(
        "Start person=%s (recursive=%s overwrite=%s chunk=%d)",
        person,
        recursive,
        overwrite,
        chunk_size,
    )

    for video_path in _iter_videos(person_dir, recursive=recursive):
        out_pt = _target_pt_path(save_root, person, video_path.stem)
        if out_pt.exists() and not overwrite:
            logger.info("Skip existed: %s", out_pt)
            continue

        try:
            pt_info = process_video_chunked(
                config,
                person,
                video_path,
                chunk_size=chunk_size,
            )
            _safe_save_pt(out_pt, pt_info, legacy=True)

        except Exception as e:
            logger.exception("Failed on %s: %s", video_path, e)

        finally:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@hydra.main(config_path="../configs/", config_name="prepare_dataset", version_base=None)
def main(config):
    persons: List[str] = list(getattr(getattr(config, "run", object()), "persons", []))
    if not persons:
        persons = [f"run_{i}" for i in range(3, 7)]
    for person in persons:
        process_one_person(config, person)


if __name__ == "__main__":
    main()
