import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .infer import process_one_video

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# GPU Manager for Multi-GPU Support
# --------------------------------------------------------------------------- #
class GPUManager:
    """管理 GPU 资源分配"""
    
    def __init__(self, num_gpus: Optional[int] = None):
        """
        初始化 GPU 管理器
        
        Args:
            num_gpus: 要使用的 GPU 数量。None 表示使用所有可用 GPU
        """
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if num_gpus is None:
                self.num_gpus = available_gpus
            else:
                self.num_gpus = min(num_gpus, available_gpus)
        else:
            logger.warning("CUDA not available, falling back to CPU")
            self.num_gpus = 0
        
        # GPU 队列，用于分配
        self.gpu_queue = Queue()
        for i in range(self.num_gpus):
            self.gpu_queue.put(i)
        
        logger.info(f"GPUManager initialized with {self.num_gpus} GPUs")
    
    def get_gpu(self) -> int:
        """获取一个可用的 GPU ID（阻塞直到有可用 GPU）"""
        gpu_id = self.gpu_queue.get()
        return gpu_id
    
    def release_gpu(self, gpu_id: int):
        """释放一个 GPU"""
        if self.num_gpus > 0:
            self.gpu_queue.put(gpu_id)


def find_files(
    subject_dir: Path,
    patterns: List[str],
    recursive: bool = False,
) -> List[Path]:
    """在 subject_dir 下按模式查找文件（视频或 pt）。"""
    files: List[Path] = []
    if recursive:
        for pat in patterns:
            files.extend(subject_dir.rglob(pat))
    else:
        for pat in patterns:
            files.extend(subject_dir.glob(pat))
    return sorted({f.resolve() for f in files})


# --------------------------------------------------------------------------- #
# Task Processing with GPU Support
# --------------------------------------------------------------------------- #
def process_video_task(
    gpu_manager: GPUManager,
    flag: str,
    subject_name: str,
    vid: Path,
    out_root: Path,
    inference_output_path: Path,
    cfg: DictConfig,
) -> Tuple[str, str, bool, Optional[str]]:
    """
    处理单个视频任务（在线程中执行）
    
    Args:
        gpu_manager: GPU 管理器
        flag: 视角标签（"left" 或 "right"）
        subject_name: 被测对象名称
        vid: 视频文件路径
        out_root: 输出根目录
        inference_output_path: 推理结果输出路径
        cfg: 配置
    
    Returns:
        (flag, subject_name, success, error_message)
    """
    gpu_id = gpu_manager.get_gpu()
    
    try:
        logger.info(f"[GPU {gpu_id}] {flag:5s} {subject_name:20s} START")
        
        # 更新配置中的 GPU 设备
        cfg.infer.gpu = gpu_id
        
        out_dir = process_one_video(
            video_path=vid,
            pt_path=None,
            out_dir=out_root / subject_name / flag,
            inference_output_path=inference_output_path / subject_name / flag,
            cfg=cfg,
        )
        
        logger.info(f"[GPU {gpu_id}] {flag:5s} {subject_name:20s} DONE")
        return (flag, subject_name, True, None)
    
    except Exception as e:
        error_msg = f"[GPU {gpu_id}] {flag:5s} {subject_name:20s} ERROR: {str(e)}"
        logger.error(error_msg)
        return (flag, subject_name, False, error_msg)
    
    finally:
        gpu_manager.release_gpu(gpu_id)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    # logging 设置
    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    # 读取路径
    if cfg.infer.type == "video":
        video_root = Path(cfg.paths.video_path).resolve()
        out_root = Path(cfg.paths.log_path).resolve()
        inference_output_path = Path(cfg.paths.result_output_path).resolve()
        if not video_root.exists():
            raise FileNotFoundError(f"video_path not found: {video_root}")
    elif cfg.infer.type == "unity":
        video_root = Path(cfg.paths.unity.video_path).resolve()
        out_root = Path(cfg.paths.log_path).resolve()
        inference_output_path = Path(cfg.paths.result_output_path).resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    inference_output_path.mkdir(parents=True, exist_ok=True)

    recursive = bool(cfg.dataset.get("recursive", False))

    # 搜索 patterns
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]

    # ---------------------------------------------------------------------- #
    # 扫描 video_root
    # ---------------------------------------------------------------------- #
    subjects_video = sorted([p for p in video_root.iterdir() if p.is_dir()])

    if not subjects_video:
        raise FileNotFoundError(f"No subject folders under: {video_root}")

    logger.info(f"Found {len(subjects_video)} subjects in: {video_root}")

    # { subject_name: [video files] }
    videos_map: Dict[str, List[Path]] = {}
    for subject_dir in subjects_video:
        vids = find_files(subject_dir, vid_patterns, recursive)
        if vids:
            videos_map[subject_dir.name] = vids
        else:
            logger.warning(f"[No video] {subject_dir}")

    # ---------------------------------------------------------------------- #
    # 构建视频处理任务
    # ---------------------------------------------------------------------- #
    _pairs: List[Tuple[str, str, Path]] = []

    logger.info("Building video processing tasks...")

    subjects = sorted(videos_map.keys())
    if not subjects:
        raise ValueError("No video files found")

    for subject_name in subjects:
        vids = videos_map[subject_name]
        for vid in vids:
            if "left" in vid.stem:
                _pairs.append(("left", subject_name, vid))
            elif "right" in vid.stem:
                _pairs.append(("right", subject_name, vid))
            else:
                # 默认无标记的视频作为右视角
                _pairs.append(("right", subject_name, vid))

    logger.info(f"Total matched subjects: {len(subjects)}")

    # ---------------------------------------------------------------------- #
    # Multi-GPU Multi-Threading Execution
    # ---------------------------------------------------------------------- #
    gpu_manager = GPUManager(num_gpus=cfg.infer.get("num_gpus", None))
    num_workers = cfg.infer.get("num_workers", gpu_manager.num_gpus if gpu_manager.num_gpus > 0 else 4)
    
    logger.info(f"Starting inference with {num_workers} workers on {gpu_manager.num_gpus} GPUs")
    
    # 构建任务列表
    tasks = []
    for flag, subject_name, vid in _pairs:
        tasks.append({
            "flag": flag,
            "subject_name": subject_name,
            "vid": vid,
        })
    
    # 使用 ThreadPoolExecutor 执行任务
    completed_count = 0
    failed_count = 0
    failed_tasks = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(
                process_video_task,
                gpu_manager,
                task["flag"],
                task["subject_name"],
                task["vid"],
                out_root,
                inference_output_path,
                cfg,
            ): task for task in tasks
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                flag, subject_name, success, error_msg = future.result()
                if success:
                    completed_count += 1
                else:
                    failed_count += 1
                    failed_tasks.append((flag, subject_name, error_msg))
            except Exception as e:
                failed_count += 1
                failed_tasks.append((task["flag"], task["subject_name"], str(e)))
                logger.error(f"Task execution failed: {task['flag']} {task['subject_name']} - {str(e)}")
    
    # 输出最终统计
    logger.info("==== INFERENCE SUMMARY ====")
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Completed: {completed_count}")
    logger.info(f"Failed: {failed_count}")
    
    if failed_tasks:
        logger.warning("Failed tasks:")
        for flag, subject_name, error_msg in failed_tasks:
            logger.warning(f"  - {flag} {subject_name}: {error_msg}")
    
    logger.info("==== ALL DONE ====")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
