# 3D Vis

这个目录把 front_side 里和 3D 视角可视化直接相关的能力单独抽出来了。

当前入口：`visualize_3d_results.py`

批量入口：`main.py`（兼容入口：`run.py`）

用途：

- 输入融合前和/或融合后的 3D 关键点结果
- 输出一个指定帧的单帧可视化图片
- 输出全部帧的逐帧 PNG
- 输出整段 mp4 视频
- 支持你当前 fused_smoothed_results 目录里的 15 关节结果

支持的输入格式：

- 直接保存的 numpy 数组，形状为 (T, J, 3) 或 (J, 3)
- npy 里保存的 dict/list/object
- sam3d_body 风格的 npz，包含 outputs，每一帧里有 pred_keypoints_3d

示例：

```bash
python -m vid_3d_kpt.visualize_3d_results \
  --before /path/to/before.npy \
  --after /path/to/after.npy \
  --out-dir /path/to/vis_output \
  --frame-idx 10 \
  --fps 30 \
  --skeleton auto
```

如果要直接批量处理你现在这类结果目录：

```bash
python -m vid_3d_kpt.main \
  --input-dir /workspace/data/dual_view_pose/fused_smoothed_results/person_pairs \
  --out-dir /workspace/data/dual_view_pose/fused_smoothed_results/person_pairs_vis \
  --frame-idx 30 \
  --max-frames 120 \
  --fps 30 \
  --skeleton auto
```

Unity 那组也可以直接跑：

```bash
python -m vid_3d_kpt.visualize_3d_results \
  --before /workspace/data/dual_view_pose/fused_smoothed_results/unity_pairs/male/left__right_fused.npy \
  --after /workspace/data/dual_view_pose/fused_smoothed_results/unity_pairs/male/left__right_smoothed.npy \
  --out-dir /workspace/data/dual_view_pose/fused_smoothed_results/unity_pairs/male_vis \
  --frame-idx 30 \
  --fps 30 \
  --skeleton auto
```

输出目录结构：

- frames/: 每一帧的 3D 可视化图
- single_frame/: 额外导出的指定帧图片
- video/: 保存 mp4 视频

常用参数：

- --before, --after: 融合前后 3D 结果文件
- --before-key, --after-key: 如果文件里有多个字段，可以显式指定字段名
- --frame-idx: 要额外保存的单帧索引
- --max-frames: 限制渲染帧数，适合快速检查
- --view-layout: simple 为轻量单视角视频模式，multi 为四视角模式
- --skeleton: auto / mhr70 / coco17 / none
- --center-mode: none / mean / pelvis

说明：

- 如果 before 和 after 的帧数不同，会自动截断到较短长度。
- 如果只有一份输入，也可以单独渲染该序列。
- 该工具默认用 matplotlib 离屏渲染，适合服务器环境。
- 对于你这批 (T, 15, 3) 的 fused/smoothed 结果，auto 会尝试使用 COCO17 的骨架连线（如果关节点数量匹配）。
- 默认使用 simple 轻量布局，更适合长视频；如果你要多视角静态检查，可以显式传 --view-layout multi。

Python import 用法：

```python
from pathlib import Path
import argparse

from vid_3d_kpt import run_visualization

args = argparse.Namespace(
  before=Path("/path/to/before.npy"),
  after=Path("/path/to/after.npy"),
  before_key=None,
  after_key=None,
  out_dir=Path("/tmp/vis_out"),
  fps=30,
  frame_idx=0,
  max_frames=120,
  skeleton="auto",
  center_mode="none",
  video_name=None,
  view_layout="simple",
  npz=None,
  npz_key=None,
)
run_visualization(args)
```