from pathlib import Path
import torch
from tqdm import tqdm
import logging

import cv2
from .sam3d import SAM3DBodyPipeline

from torchvision.io import read_video
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def process_one_video(
    video_path: Path,
    out_dir: Path,
    flag: str,
    cfg: DictConfig,
):
    """处理单个视频文件的镜头编辑。"""

    subject = video_path.parent.name or "default"

    out_dir = out_dir / subject / flag
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = read_video(video_path.as_posix(), pts_unit="sec", output_format="THWC")[0]

    pipe = SAM3DBodyPipeline(cfg=cfg)

    for idx in tqdm(range(0, frames.shape[0]), desc="Processing frames"):
        # if idx > 1:
        #     break
        vis_img, outputs = pipe.process_image(
            image=frames[idx].numpy(), return_outputs=True
        )

        out_path = out_dir / f"frame_{idx:04d}_vis.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path.as_posix(), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        logger.info(f"[Saved] {out_path}")

    # final
    torch.cuda.empty_cache()
    del pipe

    return out_dir
