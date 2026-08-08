"""
File: utils.py
Project: utils
Created Date: 2023-09-03 13:02:25
Author: chenkaixu
-----
Comment:

Have a good code time!
-----
Last Modified: 2023-09-03 13:03:05
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

"""

import logging
from pathlib import Path
from typing import List

import cv2
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _sorted_frame_paths(frame_dir: Path) -> List[Path]:
    """Return frame paths sorted by the numeric prefix before the first underscore."""

    return sorted(
        [p for p in frame_dir.iterdir() if p.is_file()],
        key=lambda path: int(path.stem.split("_")[0]),
    )


def merge_frame_to_video(
    save_path: Path, person: str, video_name: str, flag: str, filter: bool = False
) -> None:

    if filter:
        _save_path = save_path / "vis" / "filter_img" / flag / person / video_name
        _out_path = save_path / "vis" / "filter_video" / flag / person
    else:
        _save_path = save_path / "vis" / "img" / flag / person / video_name
        _out_path = save_path / "vis" / "video" / flag / person

    frames = _sorted_frame_paths(_save_path)
    if not frames:
        raise FileNotFoundError(f"No frames found in {_save_path}")

    if not _out_path.exists():
        _out_path.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read first frame: {frames[0]}")

    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(_out_path / video_name) + ".mp4", fourcc, 30.0, (width, height)
    )

    try:
        out.write(first_frame)
        for f in tqdm(frames[1:], desc=f"Save {flag}-{video_name}", total=len(frames)):
            img = cv2.imread(str(f))
            if img is None:
                logger.warning("Skipping unreadable frame: %s", f)
                continue
            out.write(img)
    finally:
        out.release()

    logger.info(f"Video saved to {_out_path / video_name}.mp4")


def process_none(batch_Dict: dict[torch.Tensor], none_index: list):
    """
    Replace ``None`` entries in ``batch_Dict`` using the nearest valid frame.

    The function now prefers the next available frame; if no later frame exists,
    it falls back to the most recent previous frame. This prevents sequences from
    remaining ``None`` when consecutive frames are missing, which would
    otherwise break downstream stacking.
    """

    if not none_index:
        return batch_Dict

    ordered_indices = sorted(batch_Dict.keys())
    forward_fill = {}
    backward_fill = {}

    last_valid = None
    for idx in ordered_indices:
        value = batch_Dict[idx]
        if value is not None:
            last_valid = value
        forward_fill[idx] = last_valid

    next_valid = None
    for idx in reversed(ordered_indices):
        value = batch_Dict[idx]
        if value is not None:
            next_valid = value
        backward_fill[idx] = next_valid

    filled_batch = batch_Dict.copy()
    for idx in none_index:
        if idx not in batch_Dict:
            continue

        replacement = backward_fill.get(idx) or forward_fill.get(idx)
        if replacement is None:
            logger.warning("All frames are missing; index %s remains None", idx)
            continue

        filled_batch[idx] = replacement

    return filled_batch
