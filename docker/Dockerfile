# docker image（PyTorch + CUDA + Python）
FROM pytorch/pytorch:latest

# 设置环境变量
ENV PYTHONUNBUFFERED TRUE
ENV TZ=Asia/Tokyo

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    libsndfile1 libturbojpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# # add user 
# USER model-server
# RUN useradd -m model-server \
#     && mkdir -p /home/model-server/tmp

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    pytorch-lightning \
    torch \
    torchvision \
    ultralytics==8.3.113 \
    hydra-core \
    scikit-learn \
    imblearn \
    grad-cam \
    captum \
    tensorboard \
    rich \ 
    av \
    nvitop \
