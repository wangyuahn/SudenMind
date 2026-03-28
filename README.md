# SudenMind

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
</p>

<p align="center">
  <a href="docs/README_EN.md">
    <img src="https://img.shields.io/badge/📖-English%20Version-2ea44f?style=for-the-badge">
  </a>
</p>

SudenMind 是一个基于 **AttnRes (Attention with Residual)** 架构的中文对话生成模型。采用 Encoder-Decoder 设计，使用自定义 2 层 AttnRes 编码器 + 6 层 AttnRes 解码器，支持跨层残差连接，使用 **ShareGPT/ChatML** 业界标准对话格式，适配 ChatGLM tokenizer，针对中文对话场景优化。

---

## ✨ 核心特性

- **AttnRes 架构**：每层可动态关注之前所有层的输出，信息流动更丰富
- **自定义 Encoder-Decoder**：6层AttnResEncoder + 6层AttnResDecoder，每层都可访问之前所有层
- **MoE (混合专家) 集成**：所有 FFN 层替换为 MoE 层，8 个专家，top_k=2
- **ShareGPT/ChatML 格式**：使用业界标准对话格式，兼容 OpenAI/Qwen/ChatGLM2/3
- **ChatGLM Tokenizer**：适配 ChatGLM-6B tokenizer（词表大小 65024）
- **混合精度训练**：FP16 加速，节省显存，支持 AMD ROCm
- **ONNX 导出**：支持导出 ONNX 格式，可用 Netron 可视化模型结构

---

## 🏗️ 模型架构

基于 [Attention Residuals](https://arxiv.org/abs/2603.15031) 论文的 Encoder-Decoder 架构：

```
输入 (batch, seq_len)
    ↓
Token Embedding + Position Encoding
    ↓
AttnRes Encoder × 2 ← 自定义编码器（双向注意力）
  ├─ 第0层: 双向自注意力 → MoE → output_0
  ├─ 第1层: 双向自注意力 → MoE → AttnRes([output_0]) → output_1
  └─ ...每层都可访问之前所有层的输出
    ↓
AttnRes Decoder × 6 ← 解码器（因果注意力）
  ├─ 自注意力 (带因果掩码)
  ├─ 跨层残差注意力 (AttnRes)
  └─ MoE 前馈网络 (8 个专家，top_k=2)
    ↓
Linear → Softmax
    ↓
输出 (batch, seq_len, vocab_size=65024)
```

---

## 📝 ShareGPT/ChatML 对话格式

SudenMind 使用业界标准的 **ShareGPT/ChatML** 格式：

### 训练格式

```
[gMASK] [BOS] <|im_start|>user
你好
<|im_end|>
<|im_start|>assistant
你好！很高兴为你服务。
<|im_end|>
<|im_start|>user
今天天气怎么样？
<|im_end|>
<|im_start|>assistant
今天天气很好！
<|im_end|>
[EOS]
```

### 推理格式

```
[gMASK] [BOS] <|im_start|>user
你好
<|im_end|>
<|im_start|>assistant
你好！很高兴为你服务。
<|im_end|>
<|im_start|>user
今天天气怎么样？
<|im_end|>
<|im_start|>assistant
```

模型会生成回复直到遇到 `<|im_end|>` 或 `[EOS]`。

### 与业界对比

| 模型 | 格式 |
|------|------|
| **SudenMind** | `<|im_start|>user\n{内容}<|im_end|>` |
| OpenAI GPT-4 | `<|im_start|>user<|im_sep|>{内容}<|im_end|>` |
| Qwen | `<|im_start|>user\n{内容}<|im_end|>` |
| ChatGLM2/3 | `[gMASK] [BOS] <|user|>\n{内容}<|assistant|>` |

---

## 📁 文件结构

```
SudenMind/
├── config.json             # ⭐ 超参数配置文件
├── data/
│   └── cache/              # 数据集缓存目录
├── model/                  # 模型保存目录
│   ├── sudenmind.pth       # 训练好的模型
│   └── sudenmind.onnx      # ONNX模型
├── src/                    # 源代码
│   ├── model.py            # AttnRes 模型定义
│   ├── data_utils.py       # ShareGPT/ChatML 格式数据处理
│   ├── train.py            # 训练脚本
│   ├── chat.py             # 交互式对话
│   ├── moe.py              # MoE 模块
│   └── view_module.py      # 模型可视化
├── docs/                   # 文档
│   └── README_EN.md
└── README.md               # 本文件
```

---

## ⚙️ 配置说明

所有超参数集中在 `config.json` 中管理：

```json
{
  "model": {
    "vocab_size": 65024,
    "d_model": 512,
    "d_fnn": 1024,
    "nhead": 8,
    "n_layers": 6,
    "num_experts": 8,
    "top_k": 2,
    "tokenizer_name": "THUDM/chatglm-6b"
  },
  "training": {
    "lr": 0.0003,
    "batch_size": 8,
    "max_epochs": 200,
    "use_amp": true
  },
  "data": {
    "format": "sharegpt",
    "max_seq_len": 512,
    "max_history": 5,
    "role_markers": {
      "user": "user",
      "assistant": "assistant"
    },
    "special_tokens": {
      "im_start": "<|im_start|>",
      "im_end": "<|im_end|>",
      "gmask_id": 64790,
      "bos_id": 64792,
      "eos_id": 2
    }
  }
}
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
conda create -n sudenmind python=3.10
conda activate sudenmind

# AMD ROCm 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# 或 CUDA 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 其他依赖
pip install transformers datasets sentencepiece onnx onnxruntime netron
```

### 2. 训练模型

```bash
python src/train.py
```

**训练特性**：
- LCCC 数据集自动加载
- ShareGPT/ChatML 格式自动转换
- 混合精度训练 (FP16)
- Cosine Annealing 学习率调度
- 早停机制

### 3. 对话测试

```bash
python src/chat.py
```

对话示例：
```
==================================================
SudenMind 对话系统 (ShareGPT/ChatML 格式)
==================================================
命令: 'quit' 退出 | 'clear' 清空历史 | 'history' 查看历史
--------------------------------------------------

You: 你好
Assistant: 你好！很高兴为你服务。

You: 今天天气怎么样？
Assistant: 今天天气很好，阳光明媚！
```

---

## 📚 参考文献

[1] Kimi Team, et al. "Attention Residuals." arXiv preprint arXiv:2603.15031 (2026).
[2] Du, Zhengxiao, et al. "GLM: General Language Model Pretraining with Autoregressive Blank Infilling." arXiv preprint arXiv:2103.10360 (2021).

---

**版本**: 6.0 (ShareGPT/ChatML 格式版)  
**状态**: ✅ 重构完成

## 🔄 版本变更日志

### v6.0 (当前版本)
- **完全重构为 ShareGPT/ChatML 格式**
- **删除所有中文角色前缀** ("用户:" / "助手:")
- **使用业界标准格式**：`<|im_start|>user/assistant`
- **兼容 OpenAI/Qwen/ChatGLM2/3** 等主流模型格式

### v5.0
- 移除 BERT 编码器，改为自定义 2 层 AttnResEncoder
- 适配 ChatGLM tokenizer（词表大小 65024）
