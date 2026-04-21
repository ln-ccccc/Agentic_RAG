# AgenticRAG: Linux (Ubuntu/CentOS) 生产环境部署指南

本文档专门针对在原生 Linux 环境（如 Ubuntu 22.04 / CentOS 8）下进行项目部署、本地大模型服务拉起以及完整训练流水线的环境配置。

如果你是使用拥有独立 GPU 算力的 Linux 服务器进行研究和训练，请严格按照本指南进行操作。

---

## 1. 基础环境与驱动要求

在开始之前，请确保您的 Linux 服务器满足以下前置条件：
- **操作系统**：Ubuntu 20.04/22.04 或 CentOS 8/9
- **GPU 驱动**：已安装 NVIDIA Driver，支持 CUDA 12.1 及以上版本。
- **CUDA Toolkit**：建议安装 CUDA 12.1 或 12.4。可以通过 `nvcc -V` 检查。
- **Python 环境管理**：已安装 Miniconda 或 Anaconda。

---

## 2. Conda 环境隔离策略 (核心)

由于项目中包含了 `vLLM`（用于推理）、`LLaMA-Factory`（用于 SFT）和 `verl`（用于 GRPO 强化学习），这些底层库对 PyTorch 版本、CUDA 算子和 Triton 的依赖存在冲突。因此，**必须严格区分两个独立的 Conda 环境**。

### 2.1 创建 AgenticRAG 环境 (用于推理、数据合成与 SFT)
此环境主要负责日常的 RAG 检索、Agent 编排执行、基于 vLLM 的本地推理，以及调用 LLaMA-Factory 进行微调。

```bash
# 1. 创建环境
conda create -n agenticrag python=3.11 -y
conda activate agenticrag

# 2. 安装与 CUDA 12.x 兼容的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目主依赖（包含了 vLLM、Faiss、Datasets 等）
pip install -r requirements.txt

# 4. (可选) 如果需要执行 SFT 微调，安装 LLaMA-Factory 源码
cd LLaMA-Factory
pip install -e .[metrics]
cd ..
```

### 2.2 创建 VERL 环境 (专用于 GRPO 强化学习)
`verl` 是一个专门用于 LLM 强化学习的框架，对底层的分布式训练库（如 DeepSpeed、Megatron 等）依赖极深，因此必须独立建环。

```bash
# 1. 创建独立环境 (推荐 Python 3.12)
conda create -n verl python=3.12 -y
conda activate verl

# 2. 安装特定版本的 PyTorch (verl 推荐 2.8.0 或更高版本)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 verl 源码
cd verl
pip install -e .
cd ..

# 4. 安装强化学习相关依赖
pip install -r requirements-verl.txt
```

---

## 3. 本地大语言模型 (LLM) 部署

在 Linux 服务器上，我们强烈推荐使用 **vLLM** 部署本地开源大模型（如 Qwen 系列或 DeepSeek 系列），这能提供最高的推理吞吐量，且不会产生 API 调用费用。

### 3.1 下载模型权重
推荐使用 `huggingface-cli` 或 `modelscope` 将大模型和检索小模型下载到服务器的 SSD 硬盘上。

```bash
mkdir -p models

# 开启 HuggingFace 国内镜像源 (国内网络必选)
export HF_ENDPOINT=https://hf-mirror.com

# 1. 下载 BGE 检索模型（必选）
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3

# 2. 下载用于推理的 LLM 基座（以 Qwen3-4B 为例）
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

# 3. (可选) 下载更强力的 LLM 作为 Judge 裁判模型
huggingface-cli download Qwen/Qwen3-32B --local-dir models/Qwen3-32B
```

### 3.2 使用 vLLM 拉起服务
在 Linux 环境下，我们使用 `tmux` 或 `screen` 在后台拉起 vLLM 的 API Server。请在 `agenticrag` 环境下执行：

```bash
conda activate agenticrag

# 启动 Agent 推理模型 (监听 9097 端口)
python -m vllm.entrypoints.openai.api_server \
    --model ./models/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --port 9097 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 32768 \
    --trust-remote-code
```

*(如果您的显存足够大（如 A100 80G x 2），可以在另一个终端拉起 32B 的 Judge 模型，监听不同端口（如 8086）。)*

### 3.3 环境变量配置
在主终端中，将环境变量指向你刚刚部署的本地服务：

```bash
# 1. 配置 Agent 引擎
export VLLM_BASE_URL="http://localhost:9097/v1"
export AGENT_LLM_MODEL="Qwen3-4B"

# 2. 配置基础依赖路径
export MODEL_HUB="./models"
export PROMPT_LANG="zh"

# 3. 配置 Judge 引擎 (如果是本地拉起的 32B，填本地地址；如果图省事，也可以这里填云端 API)
export JUDGE_BASE_URL="http://localhost:8086/v1"
export JUDGE_LLM_MODEL="Qwen3-32B"
```
*建议将这些 `export` 命令写入项目的 `.env` 文件或 `~/.bashrc` 中。*

---

## 4. 后续步骤：数据准备与训练

至此，Linux 服务器的基础算力与模型服务已经 Ready。接下来的流程与常规环境一致：

1. **准备数据与索引**：
   在 `agenticrag` 环境下，执行下载语料和构建 FAISS/BM25 索引的脚本。
   ```bash
   python scripts/download_datasets.py
   python scripts/build_index.py
   ```

2. **单条全链路测试**：
   ```bash
   python scripts/run_pipeline.py "名创优品2023年的净利润是多少？它的主要竞争对手是谁？"
   ```

3. **启动 SFT 训练**：
   依赖 `agenticrag` 环境和 LLaMA-Factory 库。
   ```bash
   bash training/sft_pipeline.sh
   ```

4. **启动 GRPO 强化学习**：
   - 终端 A (环境: `agenticrag`): 启动提供 Embedding 的检索服务 `bash training/start_retrieval_server.sh`
   - 终端 B (环境: `verl`): 执行强化学习入口 `bash training/start_grpo.sh`