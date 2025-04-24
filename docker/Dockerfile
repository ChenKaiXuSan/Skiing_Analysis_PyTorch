# docker image（PyTorch + CUDA + Python）
ARG PYTORCH="2.5.1"
ARG CUDA="12.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 设置环境变量
ENV PYTHONUNBUFFERED TRUE
ENV TZ=Asia/Tokyo

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    # MMDET Requirements
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    libsndfile1 libturbojpeg \
    && rm -rf /var/lib/apt/lists/*

RUN export FORCE_CUDA=1

WORKDIR /workspace

# # add user 
# USER model-server
# RUN useradd -m model-server \
#     && mkdir -p /home/model-server/tmp

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 7. 默认命令
CMD ["bash"]
