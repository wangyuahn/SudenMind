# SudenMind Transformer Chatbot

[English](https://www.google.com/search?q=%23english-version) | [中文版](https://www.google.com/search?q=%23%E4%B8%AD%E6%96%87%E7%89%88)

---

## 中文版

### 1. 项目简介

**SudenMind** 是一个基于 PyTorch 原生 Transformer 架构实现的端到端（End-to-End）聊天机器人。本项目采用了 **Pre-LN (norm_first)** 结构以确保快速收敛，并在预处理阶段通过 **Q+A 拼接策略** 提升了模型对上下文逻辑的理解能力。

### 2. 核心特性

* **架构优化**：采用 Pre-LayerNorm 结构，彻底解决深层 Transformer Loss 不下降的问题。
* **高效预处理**：序列拼接逻辑固化在 `process.py` 中，减少训练时的 CPU 开销。
* **灵活生成**：支持 `temperature` 参数调节，平衡回答的稳定性与创造力。
* **训练保障**：集成 Linear Warmup 学习率预热策略。

### 3. 文件结构

* `process.py`: 语料分词、词表构建及数据预训练。
* `model.py`: 定义 `SudenMind` 模型架构与位置编码。
* `datasets.py`: 轻量化数据加载器。
* `train.py`: 核心训练循环（包含 Warmup 和早停逻辑）。
* `chat.py`: 交互式对话接口。

### 4. 快速开始

1. **准备数据**：将你的对话语料存放在 `data/corpus.txt`（格式：`问题\t回答`）。
2. **数据处理**：
```bash
python process.py

```


3. **开始训练**：
```bash
python train.py

```


4. **开始对话**：
```bash
python chat.py

```