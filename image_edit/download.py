
from huggingface_hub import snapshot_download

# 下载主模型
snapshot_download(repo_id="Qwen/Qwen-Image-Edit-2509", local_dir="./models/Qwen-Image-Edit-2509")

# 下载 transformer 权重
snapshot_download(repo_id="linoyts/Qwen-Image-Edit-Rapid-AIO", local_dir="./models/Qwen-Image-Edit-Rapid-AIO")

# 下载 LoRA 权重
snapshot_download(repo_id="dx8152/Qwen-Edit-2509-Multiple-angles", local_dir="./lora/multiple-angles")

# snapshot_download(repo_id="dx8152/Qwen-Edit-2509-Multi-Angle-Lighting", local_dir="./lora/multi-angle-lighting")
