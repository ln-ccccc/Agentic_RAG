# AgenticRAG: 新手入门与数据准备指南 (Getting Started Guide)

欢迎学习 AgenticRAG 项目！对于刚刚拿到代码的开发者，由于本项目不仅包含推理和检索编排，还涉及完整的数据合成、SFT 和 GRPO 强化学习闭环，初看可能会觉得庞大。

不用担心，本文档将带你从**环境搭建**、**数据准备**到**跑通第一个最小测试用例**，一步步掌握这个系统。

---

## 1. 核心环境与模型准备

为了避免依赖冲突，项目将“推理/评测/SFT”与“GRPO强化学习”严格分为了两个环境。作为新手，你首先只需要配置基础的推理环境。

### 1.1 创建基础推理环境
```bash
conda create -n agenticrag python=3.11
conda activate agenticrag
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 1.2 下载必要的本地检索模型
系统底层的向量检索和重排重度依赖 BGE 系列模型。你需要先将它们下载到本地的 `models/` 目录下：
```bash
mkdir -p models
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
```

### 1.3 启动本地 LLM 服务 (vLLM)
AgenticRAG 的核心大脑是 LLM。项目通过 OpenAI 兼容接口调用模型，推荐在本地使用 vLLM 拉起一个较小参数的模型（如 Qwen3-4B）进行测试：
```bash
# 在一个新终端中运行
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --port 9097 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 32768
```
随后，在你的主终端中配置环境变量，告诉系统 LLM 在哪里：
```bash
export VLLM_BASE_URL="http://localhost:9097/v1"
export AGENT_LLM_MODEL="Qwen3-4B"
export MODEL_HUB="./models"
export PROMPT_LANG="zh"
```

---

## 2. 数据与索引的准备

要让检索系统跑起来，我们必须要有语料库（Corpus）和对应的检索索引（FAISS、BM25 等）。针对新手，我们推荐两种方式获取数据：

### 方式一：下载开源基准数据集（最快，推荐新手使用）
项目中提供了一个脚本，可以直接从 HuggingFace 下载 `AgenticRAGTracer` 数据集。
```bash
# 下载数据，这会自动在 data/datasets/ 下生成 corpus.json 和 qa_pairs.json
python scripts/download_datasets.py
```

### 方式二：垂直领域多跳 QA 合成（进阶）
如果你想基于自己的私有研报或财报数据进行训练，你可以使用项目提供的数据合成流水线：
1. 准备你的私有语料文件（文本或 PDF 解析后）。
2. 运行 `python scripts/gen_seed_qa.py` 生成种子问答。
3. 运行 `python scripts/domain_multihop_synthesis.py` 拼接并合成复杂的多跳问题。
4. 运行 `python scripts/judge_synthesis.py` 过滤掉低质量数据。

### 2.1 构建检索索引
无论你用哪种方式得到了 `corpus.json`，下一步都是将其构建为检索索引。这包括 FAISS 向量索引、BM25 关键词倒排索引和知识图谱。

```bash
# 1. 构建向量与标量索引 (FAISS & BM25)
# 这会在 data/indexes/ 下生成 faiss.index, bm25.pkl, chunk_ids.json 等文件
python scripts/build_index.py

# 2. (可选) 构建知识图谱索引
# 这会调用 LLM 抽取三元组，并构建基于 NetworkX 的实体图谱
python scripts/build_knowledge_graph.py
```

---

## 3. 运行最小测试用例 (Hello World)

当环境、模型和索引都准备就绪后，你可以通过运行单条测试脚本来观察 AgenticRAG 是如何工作的。

项目中提供了一个极佳的调试脚本：`scripts/run_pipeline.py`。这个脚本会直接调用核心的 LangGraph 编排引擎，并**在控制台详细打印出 Agent 的每一步思考轨迹**。

### 运行命令：
```bash
# 确保你已经设置了 VLLM_BASE_URL 等环境变量
python scripts/run_pipeline.py "名创优品2023年的净利润是多少？它的主要竞争对手是谁？"
```

### 你将观察到的执行轨迹（Execution Trace）：
1. **Router**：判断这是一个复杂问题，路由到多跳处理流程。
2. **Planner**：大模型将问题拆解，决定先调用 `keyword_search` 或 `semantic_search` 查找“名创优品2023年净利润”。
3. **Executor**：系统真实执行检索，返回包含利润数据的文本 Chunk。
4. **Verifier**：大模型评估当前证据，发现还缺少“竞争对手”的信息。
5. **Replan (Planner)**：大模型发起第二轮查询，搜索名创优品的竞争对手。
6. **Synthesizer**：在证据收集充分后，生成最终的准确答案。

通过观察这个单步脚本的输出，你可以最直观地理解 `agents/graph.py` 中的代码流转逻辑。

---

## 4. 下一步学习建议

当你成功跑通了单条测试后，你可以按照以下顺序继续深入源码：
1. **阅读 Agent 编排逻辑**：打开 `agents/graph.py`，对照你刚才看到的 Trace，理解 State 是如何在 Planner、Executor 和 Verifier 之间流转的。
2. **批量评测**：尝试运行 `python scripts/eval_agentic.py --model Qwen3-4B --max-samples 50`，看看模型在多条测试数据上的综合表现（EM、F1 等指标）。
3. **探索训练闭环**：阅读 [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)，了解如何将这些 Agent 轨迹转换为 SFT 数据，并最终跑通 GRPO 强化学习对齐。
