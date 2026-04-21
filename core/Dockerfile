FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 环境变量配置
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 配置 Ubuntu APT 国内源 (阿里云)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装基础系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    build-essential \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda (使用清华源加速下载)
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda3 \
    && rm miniconda.sh \
    && conda config --set auto_activate_base false \
    && conda config --set show_channel_urls true \
    && conda config --set always_yes yes \
    && conda init bash \
    && conda clean -a -y

# 配置 Conda 国内源 (清华源)
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    && conda config --set channel_priority strict \
    && conda config --remove channels defaults || true

# 配置 PIP 国内源 (清华源)
RUN /opt/miniconda3/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

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

# 将代码拷贝进容器
COPY . .

# 尝试安装子模块 LLaMA-Factory (SFT) 和 verl (RL)
# 注意：LLaMA-Factory 安装时需要读取其自己目录下的 README.md 等文件
RUN if [ -d "LLaMA-Factory" ] && [ -f "LLaMA-Factory/setup.py" ]; then \
        cd LLaMA-Factory && conda run -n agenticrag pip install -e .[metrics] && cd ..; \
    fi
RUN if [ -d "verl" ] && [ -f "verl/setup.py" ]; then \
        cd verl && conda run -n verl pip install -e . && cd ..; \
    fi

# 默认进入 bash，并激活基础的 agenticrag 环境
RUN echo "conda activate agenticrag" >> ~/.bashrc
CMD ["/bin/bash"]
