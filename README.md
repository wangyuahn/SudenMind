# SudenMind

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
  <img src="https://img.shields.io/badge/Website-ai.sufun.space-blue.svg">
</p>

<p align="center">
  <a href="docs/README_EN.md">
    <img src="https://img.shields.io/badge/📖-English%20Version-2ea44f?style=for-the-badge">
  </a>
</p>

SudenMind 是一个基于 **AttnRes (Attention Residuals)** 架构的中文对话生成模型。采用 Encoder-Decoder 设计，使用自定义 2 层 AttnRes 编码器 + 6 层 AttnRes 解码器，支持跨层残差连接，使用 **ShareGPT/ChatML** 业界标准对话格式，适配 ChatGLM tokenizer，针对中文对话场景优化。新增门控机制（Gate）和 KV 缓存（KV Cache）推理加速。

---

## ✨ 核心特性

- **AttnRes 架构**：每层可动态关注之前所有层的输出，信息流动更丰富
- **自定义 Encoder-Decoder**：2层AttnRes编码器 + 6层AttnRes解码器，每层都可访问之前所有层
- **门控机制 (Gate)**：创新的门控网络动态平衡MoE输出与跨层残差输出，根据输入特征自适应调整特征融合比例
- **KV 缓存推理**：支持键值缓存，大幅提升自回归生成速度
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
  ├─ 门控特征融合 (Gate)
  └─ MoE 前馈网络 (8 个专家，top_k=2)
    ↓
Linear → Softmax
    ↓
输出 (batch, seq_len, vocab_size=65024)
```

### AttnRes 门控机制详解

在每个 AttnRes 层中，门控网络计算如下：

```
Gate = Sigmoid([MoE_out; Res_out] * W_g)
Output = (1 - Gate) ⊗ MoE_out + Gate ⊗ Res_out
```

其中：
- MoE_out：混合专家前馈网络输出
- Res_out：跨层残差注意力输出  
- W_g：可学习的门控权重矩阵
- ⊗：逐元素乘法

这个机制使模型能够根据当前输入的语义特征，动态决定采用更多的创新特征（MoE输出）还是保留历史信息（残差输出），显著提升了建模灵活性。

### KV 缓存推理

为了加速自回归生成，模型在解码过程中缓存历史键值对（Key-Value pairs）：

- 编码器输出的键值对在首次前向传播时计算并缓存
- 解码器每一步只需计算当前token的键值对，将其追加到缓存中
- 这样避免了重复计算历史token的键值对，将时间复杂度从 O(n²) 降至 O(n)

---

## 📝 ShareGPT/ChatML 对话格式

SudenMind 使用业界标准的 **ShareGPT/ChatML** 格式，并在 `data_utils.py` 中实现以下规则：

- 以 `[BOS]` 开头，`[EOS]` 结尾。
- 轮次以 `<|im_start|>{role}\n` 开头，助手或用户块以 `<|im_end|>` 结束。
- 仅对 `assistant` 轮次的内容部分计算 loss；`role header`/`<|im_end|>`/`newline` 等位置设为 `-100`。

### 训练格式

```
[BOS] <|im_start|>user
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
[BOS] <|im_start|>user
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

| 模型 | 格式 | 特点 |
|------|------|------|
| **SudenMind** | `<|im_start|>user` / `<|im_start|>assistant` 且均以 `<|im_end|>` 结束 | 标准 ShareGPT/ChatML，支持 MoE + Gate + KV Cache |
| OpenAI GPT-4 | `⟨im_start⟩user⟨im_sep⟩{内容}⟨im_end⟩` | 使用 `⟨im_sep⟩` 分隔符 |
| Qwen | `⟨im_start⟩user\n{内容}⟨im_end⟩` | 兼容 ShareGPT/ChatML |
| ChatGLM2/3 | `[BOS] ⟨user⟩\n{内容}⟨/assistant⟩` | 语义类似，角色分隔形式略有差异 |

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
│   ├── model.py            # AttnRes 模型定义（含门控机制和KV缓存）
│   ├── data_utils.py       # ShareGPT/ChatML 格式数据处理
│   ├── train.py            # 训练脚本
│   ├── chat.py             # 交互式对话（支持KV缓存推理）
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
    "d_model": 512,
    "d_fnn": 1024,
    "nhead": 8,
    "n_layers": 6,
    "dropout": 0.1,
    "max_position_embeddings": 5000,
    "num_experts": 8,
    "top_k": 2,
    "aux_loss_coef": 0.01,
    "tokenizer_name": "THUDM/chatglm-6b"
  },
  "training": {
    "lr": 0.0003,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "eps": 1e-08,
    "warmup_ratio": 0.05,
    "max_epochs": 200,
    "label_smoothing": 0,
    "ignore_index": -100,
    "batch_size": 1,
    "max_norm": 5.0,
    "patience": 30,
    "target_loss": 0.03,
    "use_amp": false
  },
  "data": {
    "config": "base",
    "split": "train",
    "max_history": 5,
    "max_seq_len": 512
  },
  "generation": {
    "max_length": 200,
    "temperature": 0.3,
    "top_k": 50,
    "onnx_seq_len": 32,
    "onnx_opset": 11
  },
  "paths": {}
}
```

> 注意：当前配置不再在 `config.json` 里通过显式字段声明 `<|im_start|>/<|im_end|>` 等特殊 token，使用 `THUDM/chatglm-6b` tokenizer 的内置映射。

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

# 其他依赖（硬性版本要求）
pip install transformers>=4.30.0 datasets>=2.0.0 sentencepiece>=0.1.99 onnx>=1.14.0 onnxruntime>=1.15.0 netron>=7.0.0
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

### 3. 对话测试（支持 KV 缓存推理）

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

## 🔗 项目链接

- **官网**：https://ai.sufun.space
- **GitHub**：https://github.com/wangyuahn/SudenMind
- **文档**：https://ai.sufun.space/docs

---

## 📄 版本信息

**版本**: 6.1 (AttnRes Gate & KV Cache 版本)  
**状态**: ✅ 特性增强完成  
**最后更新**: 2026-03-29

## 🔄 版本变更日志

### v6.1 (当前版本)
- **移除gMASK标记**：简化对话格式，仅保留[BOS]作为序列开始标记
- **新增门控机制 (Gate)**：动态平衡MoE输出与跨层残差输出
- **新增KV缓存推理**：大幅提升自回归生成速度
- **优化模型架构**：改进AttnRes层的信息流动机制
- **更新网站链接**：官网迁移至 ai.sufun.space

### v6.0
- **完全重构为 ShareGPT/ChatML 格式**
- **删除所有中文角色前缀** ("用户:" / "助手:")
- **使用业界标准格式**：`赠user/受assistant`
- **兼容 OpenAI/Qwen/ChatGLM2/3** 等主流模型格式

### v5.0
- 移除 BERT 编码器，改为自定义 2 层 AttnResEncoder
- 适配 ChatGLM tokenizer（词表大小 65024）