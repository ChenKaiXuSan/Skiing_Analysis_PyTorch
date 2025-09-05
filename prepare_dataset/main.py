#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import hydra
import torch
from torchvision.io import read_video

from prepare_dataset.process.preprocess import Preprocess

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


# --------------------------- 工具函数 ---------------------------


def _iter_videos(root: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        yield from (p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    else:
        yield from (p for p in root.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def _target_pt_path(save_root: Path, person: str, video_stem: str) -> Path:
    # 与原先一致：.../pt/<person>/<video>.pt
    return save_root / "pt" / person / f"{video_stem}.pt"


def _safe_save_pt(pt_path: Path, obj: Dict[str, Any], legacy_zip: bool = True) -> None:
    """原子写入，避免中途失败留下半文件；legacy_zip=True 使用旧序列化（内存更稳）"""
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pt_path.with_suffix(pt_path.suffix + ".tmp")
    torch.save(obj, tmp, _use_new_zipfile_serialization=not legacy_zip)
    os.replace(tmp, pt_path)
    try:
        size_mb = os.path.getsize(pt_path) / 1024 / 1024
        logger.info("Saved pt -> %s (%.1f MB)", pt_path, size_mb)
    except Exception:
        logger.info("Saved pt -> %s", pt_path)


# --------------------------- 单视频：整段一次性推理 ---------------------------


def process_video_whole(
    config,
    person: str,
    video_path: Path,
    *,
    embed_frames_in_pt: bool = True,  # 是否把原始帧写入 pt_info["frames"]
) -> Dict[str, Any]:
    """
    整段视频一次性读取 + 推理，返回 pt_info（字段与之前保持一致）。
    注意：长视频 + 多大模型组合可能占用较高内存，请按需在 Hydra 配置里关闭 embed_frames_in_pt。
    """
    # 读取整段视频 (THWC, uint8)
    vframes, _, info = read_video(video_path, pts_unit="sec", output_format="THWC")
    T, H, W, C = vframes.shape
    fps = float(info.get("video_fps", 0.0))
    duration = float(info.get("duration", 0.0))
    logger.info(
        "Read video: %s | shape=%s fps=%.3f dur=%.3fs",
        video_path.name,
        (T, H, W, C),
        fps,
        duration,
    )

    # 推理
    preprocess = Preprocess(config=config, person=person)
    with torch.inference_mode():
        pt_info = preprocess(vframes, video_path)

    # 附加元信息
    pt_info["video_name"] = video_path.stem
    pt_info["video_path"] = str(video_path)
    pt_info["frame_count"] = T
    pt_info["img_shape"] = (H, W)
    pt_info["fps"] = fps
    pt_info["duration"] = duration

    # 是否把原始帧也塞进 pt（注意体积/内存）
    if embed_frames_in_pt:
        pt_info["frames"] = vframes.cpu()  # THWC, uint8
    # 如不嵌入，下游可用 video_path 重新解码

    return pt_info


# --------------------------- 按 person 处理 & Hydra 入口 ---------------------------


def process_one_person(config, person: str) -> None:
    raw_root = Path(config.extract_dataset.data_path)
    save_root = Path(config.extract_dataset.save_path)

    person_dir = raw_root / person
    if not person_dir.exists():
        logger.warning("Person dir not found: %s", person_dir)
        return

    overwrite: bool = bool(getattr(config.extract_dataset, "overwrite", False))
    recursive: bool = bool(getattr(config.extract_dataset, "recursive", False))
    embed_frames_in_pt: bool = bool(
        getattr(config.extract_dataset, "embed_frames_in_pt", True)
    )
    legacy_zip: bool = bool(getattr(config.extract_dataset, "legacy_zip", True))
    num_threads: int = (
        int(getattr(getattr(config, "system", object()), "num_threads", 0)) or 0
    )
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        logger.info("Set torch.set_num_threads(%d)", num_threads)

    logger.info(
        "Start person=%s (recursive=%s overwrite=%s embed_frames_in_pt=%s)",
        person,
        recursive,
        overwrite,
        embed_frames_in_pt,
    )

    for video_path in _iter_videos(person_dir, recursive=recursive):
        out_pt = _target_pt_path(save_root, person, video_path.stem)
        if out_pt.exists() and not overwrite:
            logger.info("Skip existed: %s", out_pt)
            continue

        try:
            pt_info = process_video_whole(
                config,
                person,
                video_path,
                embed_frames_in_pt=embed_frames_in_pt,
            )
            _safe_save_pt(out_pt, pt_info, legacy_zip)

        except Exception as e:
            logger.exception("Failed on %s: %s", video_path, e)

        finally:
            # 释放内存，避免多视频长跑时累积
            try:
                del pt_info
            except Exception:
                pass
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
