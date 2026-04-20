# AgenticRAG: 系统架构设计与核心模块解析 (System Architecture)

本文档旨在介绍 AgenticRAG 项目的核心架构和关键模块。该项目围绕金融多跳问答（Multi-hop QA）构建了基于 LangGraph 的端到端 Agentic RAG 系统，并整合了 SFT（监督微调）和 GRPO（组相对策略优化）的强化学习训练流水线。

---

## 1. 核心编排架构：PEV 循环图

系统的核心大脑是基于 LangGraph 实现的 **PEV (Planner-Executor-Verifier)** 多智能体编排循环图。该架构不再是传统 RAG 的单向流水线，而是赋予了模型自主决策与多步推理的能力。

- **源码位置**：`agents/graph.py` 及其子模块。

### 1.1 状态管理 (State)
`state.py` 定义了全局状态对象（State），在整个图中流动，维护以下关键信息：
- 原始用户查询（query）
- 当前制定的子查询计划（plan）
- 已收集的证据（evidence）
- 调用的工具记录（tool_calls）
- 验证结果（verification_result）

### 1.2 核心节点 (Nodes)
PEV 架构由以下四大核心节点组成：

1. **Router (路由节点)**：
   - 接收用户查询，判断其复杂度。
   - 简单查询被路由至快速通道（`simple_rag`），复杂查询进入多跳处理流程（`multi_hop`）。
2. **Planner (规划节点)**：
   - 将复杂查询拆解为具体的子查询任务。
   - 根据当前的证据储备，选择最合适的检索工具。
3. **Executor (执行节点)**：
   - 解析 Planner 生成的工具调用指令。
   - 调用下游的 Retrieval 模块（如语义检索、图谱检索等）执行真实动作，并将结果追加到 Evidence 中。
4. **Verifier (验证节点)**：
   - 检查已收集的证据是否足以回答用户的原始问题。
   - 若**证据充足**或**达到最大推理步数限制（Budget Exhausted）**，则图流转至 Synthesizer 生成最终答案。
   - 若**证据不足**，则触发 **Replan**，返回 Planner 节点继续进行下一跳检索。

5. **Synthesizer (合成节点)**：
   - 基于收集到的所有可靠证据（Evidence），整合生成最终的回答。

---

## 2. 检索工具箱 (Retrieval Toolbox)

面对金融数据的多样性和复杂性，系统在 `retrieval/` 目录下提供了一套丰富的检索工具，供 Executor 节点灵活调度：

- **语义检索 (`semantic_search.py`)**：基于 BGE-M3 模型的 FAISS 向量检索，捕捉深层语义相似度。
- **关键词检索 (`keyword_search.py`)**：基于 jieba 分词的 BM25 检索，精准匹配专有名词、公司名称或数字指标。
- **知识图谱检索 (`graph_search.py`)**：基于 NetworkX 构建的金融实体关系图谱，支持基于实体关联的推理。
- **混合检索 (`hybrid_search.py`)**：使用倒数秩融合（RRF）将多路召回（语义 + 关键词）结果结合，并通过 `bge-reranker-v2-m3` 模型进行 CrossEncoder 级重排（Rerank），大幅提升 Context Precision。
- **原文读取 (`read_chunk.py`)**：允许 Agent 在获得特定段落标识（chunk_id）后，直接读取原文档上下文，获取更详尽的财务细节。

---

## 3. 模型训练：SFT 与 GRPO 强化学习流水线

本项目不仅仅是一个推理框架，更是完整的“训练-对齐”闭环，这是架构的第二大核心支柱。

### 3.1 SFT 监督微调流水线 (`sft_pipeline.sh`)
- **目标**：赋予基座大语言模型初步的工具调用与规划能力（格式对齐）。
- **流程**：
  1. 通过 `scripts/build_oracle_traces.py` 脚本，将理想的 Agent 推理轨迹转化为标准的 ReAct 格式。
  2. 依托 `LLaMA-Factory` 框架，执行 LoRA 监督微调，并将微调后的权重与基座模型合并。
  3. 自动化流水线可一键启动 vLLM 进行模型拉起，对所有 Checkpoint 执行串行自动化评测。

### 3.2 GRPO 强化学习流水线 (`start_grpo.sh` & `reward_agentic_rag.py`)
- **目标**：基于强化学习（RL）解决模型在多步检索中“什么时候该停”、“怎样检索更准”以及“如何基于证据诚实回答”的对齐难题。
- **实现**：依托 `verl` 强化学习框架，采用多轮 Agentic RL 训练。
- **复杂奖励机制 (Reward Function)**：`training/reward_agentic_rag.py` 实现了精细的奖励塑形，综合考量以下维度：
  - **Hop Precision-Recall (0.30)**：奖励精准命中金标准文档且不引入噪声的行为。
  - **Faithfulness (0.25) & Correctness (0.25)**：通过引入强力 LLM-as-a-Judge（如 GPT-4o 或百亿级开源模型）动态评估答案的忠实度和正确性。
  - **Grounded Answer (0.10)**：奖励答案文本与证据 Token 的高重合度。
  - **Format (0.10)**：规范模型严格输出 `<answer>` 和 `<tool_call>` 标签。
  - **搜索不足惩罚**：惩罚过早放弃检索、未完成必要多跳推理的行为。

---

## 4. 自动化评测框架 (Evaluation Framework)

为了闭环验证模型的改进效果，系统在 `evaluation/` 目录下构建了完善的评测套件：

- **LLM Judge (`llm_judge.py`)**：利用强力裁判模型进行三维度打分：
  - **Faithfulness（忠实度）**：生成的答案是否完全由检索到的证据支持。
  - **Context Precision（上下文精度）**：检索到的文档是否有效覆盖了金标准参考信息。
  - **Correctness（正确性）**：模型给出的预测答案与标准答案语义上是否一致。
- **诊断指标 (`metrics.py` & `hop_aware_eval.py`)**：计算传统的 Exact Match (EM)、F1 Score，并输出多跳检索命中率（Hop Recall），精确定位错误发生在检索端还是生成端。

---

## 5. 数据流转与合成脚本

项目通过 `scripts/` 下的 20 余个脚本串联起了整个工程生命周期：
- **无监督语料 -> QA 对**：`gen_seed_qa.py` 与 `domain_multihop_synthesis.py` 负责从金融研报、财报库中合成高质量的多跳 QA 数据集。
- **数据清洗与过滤**：`judge_synthesis.py` 与 `clean_synthesis.py` 用于剔除低质量合成数据。
- 这些数据最终流向 `data/` 目录，被分割为金融全语料索引、SFT 训练集和 GRPO Parquet 训练集。
