# AgenticRAG: 模型训练与对齐指南 (Training Guide)

本项目提供了一个端到端的模型训练流水线，旨在使基础大语言模型（LLM）掌握复杂的 Agentic RAG（基于智能体的检索增强生成）能力，并特别针对金融多跳问答场景进行了优化。

整个训练流程分为两个主要阶段：**SFT（监督微调）** 和 **GRPO（组相对策略优化）强化学习**。本指南将详细说明这两个阶段的目标、数据准备和执行步骤。

---

## 1. 训练环境准备

请确保您已按照项目根目录下的 `README.md` 正确配置了两个独立的 Conda 环境：
- **`agenticrag`** 环境：用于数据合成、格式转换和 SFT 训练（基于 `LLaMA-Factory`）。
- **`verl`** 环境：专门用于 GRPO 强化学习训练（基于 `verl` 框架）。

这两个环境相互隔离，以避免依赖冲突（例如不同的 PyTorch 版本或 vLLM 版本）。

---

## 2. 阶段一：SFT (监督微调)

**目标**：通过高质量的“Oracle”推理轨迹，让模型初步学会工具调用的格式（例如 `<tool_call>` 和 `<answer>` 标签的使用），以及基本的规划（Plan）、执行（Execute）和验证（Verify）的思考过程（ReAct 范式）。

### 2.1 数据准备与格式转换

1. **构建 Oracle Trace**：
   运行脚本生成理想的 Agent 推理轨迹。这些轨迹包含了模型在回答多跳问题时应当采取的正确步骤和检索动作。
   ```bash
   conda activate agenticrag
   python scripts/build_oracle_traces.py
   ```

2. **转换为 ReAct 格式**：
   将生成的轨迹转换为 SFT 训练所需的 ReAct（Reasoning and Acting）对话格式。
   ```bash
   python scripts/trace_to_sft.py --lang zh
   ```

3. **适配 LLaMA-Factory 格式**：
   将数据转换为 `LLaMA-Factory` 支持的 JSON/JSONL 格式。
   ```bash
   python scripts/convert_sft_to_llamafactory.py
   ```

### 2.2 执行 SFT 流水线

项目提供了一个一站式的 Bash 脚本 `training/sft_pipeline.sh`，它会自动完成以下任务：
- 调用 `LLaMA-Factory` 使用准备好的数据进行 LoRA 微调。
- 将训练好的 LoRA 权重与基座模型合并。
- 自动启动 vLLM 开启对所有生成 Checkpoint 的串行自动化评测。

**启动命令**：
```bash
bash training/sft_pipeline.sh
```

*(注：您可以在 `training/sft_zh_react.yaml` 中修改具体的训练超参数，如 learning rate, batch size 等。)*

---

## 3. 阶段二：GRPO 强化学习对齐

**目标**：SFT 只能教会模型“看起来像”在调用工具，但在面对复杂的、未见过的问题时，模型容易产生幻觉、过早放弃检索或被检索出的噪声误导。GRPO 强化学习阶段通过复杂的奖励函数（Reward Function），在探索与试错中“逼迫”模型学会真正的推理和正确引用证据（Grounded Answer）。

### 3.1 奖励机制 (Reward Design)

GRPO 训练的核心在于 `training/reward_agentic_rag.py` 中定义的奖励函数。它从以下五个维度对模型的每一次尝试进行打分：
1. **Hop Precision-Recall (30%)**：奖励精准命中金标准文档，惩罚引入无关噪声。
2. **Faithfulness (25%)**：通过 LLM-as-a-Judge 评估生成的答案是否完全由检索到的证据所支持。
3. **Correctness (25%)**：评估预测答案与金标准答案在语义上是否一致。
4. **Grounded Answer (10%)**：奖励答案文本与证据 Token 的高重合度。
5. **Format (10%)**：格式合规性奖励，确保模型输出符合解析规范。
*额外机制*：对搜索不足（未完成必要多跳推理）的行为给予扣分惩罚。

### 3.2 数据准备

将 SFT 之后或用于强化学习的 QA 数据准备为 `verl` 框架支持的 Parquet 格式：
```bash
conda activate agenticrag
python scripts/prepare_agentic_grpo_data.py
```

### 3.3 启动检索服务

由于在强化学习训练过程中，模型（Actor）需要实时与环境（Environment）交互，即执行真实的检索动作，因此必须先启动后端的检索服务：
```bash
conda activate agenticrag
bash training/start_retrieval_server.sh
```
此服务会加载 BGE-M3 Embedding 和 Reranker 模型，并对外提供 HTTP 检索接口（由 `training/tools/retrieval_server.py` 实现）。

### 3.4 执行 GRPO 训练

切换到 `verl` 训练环境，并启动强化学习流程：
```bash
conda activate verl
export VERL_DIR=$(pwd)/verl  # 确保指向正确的 verl 目录

# 启动训练
bash training/start_grpo.sh
```

在训练过程中，您可以通过观察输出的各类 Reward 分数变化，监控模型是否在检索精准度、答案忠实度和正确性上取得了进步。

---

## 4. 评测与诊断 (Evaluation)

无论是在 SFT 之后还是 GRPO 之后，都应该使用 `evaluation/` 目录下的工具对模型进行全面评测。

1. **Pipeline 模式评测**：测试整个系统的端到端表现。
   ```bash
   python scripts/run_cloud_eval.py --model Qwen3-4B --workers 1
   ```
2. **Agentic 模式评测**：深入分析 Agent 的规划与工具调用能力。
   ```bash
   python scripts/eval_agentic.py --model Qwen3-4B --max-samples 50
   ```
3. **LLM Judge 裁判打分**：使用 `gpt-oss-120b` 或 GPT-4o 对生成的答案进行 Correctness、Faithfulness 和 Context Precision 的三维度细致评估。
