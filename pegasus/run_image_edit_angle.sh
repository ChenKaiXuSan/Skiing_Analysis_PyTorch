#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gen_S                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N qwen_image_edit_angle   # 🏷 作业名称
#PBS -o logs/pegasus/qwen_image_edit_angle.log            # 📤 标准输出日志
#PBS -e logs/pegasus/qwen_image_edit_angle_err.log            # ❌ 错误输出日志

# === 切换到作业提交目录 ===
cd /work/SKIING/chenkaixu/code/Skiing_Analysis_PyTorch

mkdir -p logs/pegasus/

# === 加载 Python + 激活 Conda 环境 ===
module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda activate /home/SKIING/chenkaixu/miniconda3/envs/qwen/

# === 可选：打印 GPU 状态 ===
nvidia-smi

NUM_WORKERS=$(nproc)
# 输出当前环境信息
echo "Current working directory: $(pwd)"
echo "Total CPU cores: $NUM_WORKERS, use $((NUM_WORKERS / 3)) for data loading"
echo "Current Python version: $(python --version)"
echo "Current virtual environment: $(which python)"

# params 
root_path=/work/SKIING/chenkaixu/data/skiing

# === 运行你的训练脚本（Hydra 参数可以加在后面）===
python -m image_edit.main paths.video_path=${root_path}/side_raw model.root_path=/work/1/SKIING/chenkaixu/code/Skiing_Analysis_PyTorch/ckpt/qwen infer.gpu=0