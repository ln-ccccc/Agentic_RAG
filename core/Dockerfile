FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 环境变量配置
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 安装基础系统依赖 (配置阿里云 apt 源)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    build-essential \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda (配置清华 conda 源和阿里云 pip 源)
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda3 \
    && rm miniconda.sh \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    && conda config --set show_channel_urls yes \
    && conda config --set channel_priority strict \
    && conda config --remove channels defaults || true \
    && pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && conda init bash \
    && conda clean -a -y

# 设置工作目录
WORKDIR /workspace/core

# 提前拷贝依赖文件以利用 Docker 缓存
COPY requirements.txt requirements-verl.txt ./

# ---------------------------------------------------------
# 环境 1: agenticrag (推理、评测、数据合成与 SFT)
# ---------------------------------------------------------
RUN conda create -n agenticrag python=3.11 -y && \
    conda run -n agenticrag pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    conda run -n agenticrag pip install -r requirements.txt && \
    conda clean -a -y

# ---------------------------------------------------------
# 环境 2: verl (GRPO 强化学习专属)
# ---------------------------------------------------------
RUN conda create -n verl python=3.12 -y && \
    # 使用符合项目的 torch 版本
    conda run -n verl pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu124 || \
    conda run -n verl pip install torch --index-url https://download.pytorch.org/whl/cu124 && \
    conda run -n verl pip install -r requirements-verl.txt && \
    conda clean -a -y

# 将代码拷贝进容器（注：在使用 docker-compose 时，这层会被本地挂载覆盖，从而实现代码热更新）
COPY . .

# 尝试安装子模块 LLaMA-Factory (SFT) 和 verl (RL)
# 如果子模块文件夹存在则执行安装，以支持本地直接构建
RUN if [ -d "LLaMA-Factory" ]; then conda run -n agenticrag pip install -e ./LLaMA-Factory[metrics]; fi
RUN if [ -d "verl" ]; then conda run -n verl pip install -e ./verl; fi

# 默认进入 bash，并激活基础的 agenticrag 环境
RUN echo "conda activate agenticrag" >> ~/.bashrc
CMD ["/bin/bash"]
