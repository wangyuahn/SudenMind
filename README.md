# SudenMind

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
</p>

<p align="center">
  <a href="README_EN.md">
    <img src="https://img.shields.io/badge/📖-English%20Version-2ea44f?style=for-the-badge">
  </a>
</p>

SudenMind 是一个基于 **AttnRes (Attention with Residual)**  架构的中文对话生成模型。采用 Decoder-Only 设计，支持跨层残差连接，使用可学习位置编码，针对中文对话场景优化。

---

## ✨ 核心特性

- **AttnRes 架构**：每层可动态关注之前所有层的输出，信息流动更丰富
- **Decoder-Only 设计**：标准自回归生成，适合对话任务
- **可学习位置编码**：比固定正弦编码更灵活，适应不同长度
- **混合精度训练**：FP16 加速，节省显存，支持 AMD ROCm
- **ONNX 导出**：支持导出 ONNX 格式，可用 Netron 可视化模型结构
- **批量优先 (batch_first)**：符合 PyTorch 标准，易于理解和使用

---

## 🏗️ 模型架构

基于 [Attention Residuals](https://arxiv.org/abs/2603.15031) 论文的 Decoder-Only 架构：

```
输入 (batch, seq_len)
    ↓
Embedding + 可学习位置编码
    ↓
AttnRes Decoder × 6
  ├─ 自注意力 (带因果掩码)
  ├─ 跨层残差注意力 (AttnRes) ← 核心创新
  │   └─ 每层通过 softmax attention 动态选择之前所有层的输出
  └─ 前馈网络
    ↓
Linear → Softmax
    ↓
输出 (batch, seq_len, vocab_size)
```

**AttnRes 核心思想**：
不同于标准 Transformer 的固定残差连接（`output = x + f(x)`），AttnRes 允许第 i 层通过学习的注意力权重，选择性地聚合之前所有层的输出：
```python
# 标准残差
output = fnn_out + res_out

# AttnRes: 动态加权聚合之前所有层的输出
attn_weights = softmax(scores)  # 学习每个之前层的重要性
res_out = sum(attn_weights[i] * prev_outputs[i] for i in range(n))
output = fnn_out + res_out
```

**关键参数**（可修改）：
- `d_model`: 256 (embedding 维度)
- `nhead`: 8 (注意力头数)
- `d_fnn`: 512 (前馈网络维度)
- `n_layers`: 6 (AttnRes 层数)
- `dropout`: 0.1
- `batch_fire`: False (当前使用顺序模式，非 Block AttnRes)

---

## 📁 文件结构

```
SudenMind/
├── config.json             # ⭐ 超参数配置文件 (集中管理所有参数)
├── data/
│   ├── corpus.txt          # 原始语料 (问题\t回答)
│   ├── chat_data.json      # 处理后的训练数据
│   └── vocab.json          # 词表
├── model/
│   └── sudenmind.pth       # 训练好的模型
├── model.py                # AttnRes 模型定义
├── datasets.py             # 数据集与数据加载
├── process.py              # 数据预处理与词表构建
├── train.py                # 训练脚本 (支持混合精度)
├── chat.py                 # 交互式对话
└── README.md               # 本文件
```

---

## ⚙️ 配置说明

所有超参数集中在 `config.json` 中管理，**无需修改代码**即可调整：

```json
{
  "model": {
    "d_model": 256,        // Embedding 维度
    "d_fnn": 512,          // 前馈网络维度
    "nhead": 8,            // 注意力头数
    "n_layers": 6,         // 层数
    "dropout": 0.1
  },
  "training": {
    "lr": 0.001,           // 学习率
    "batch_size": 64,      // 批量大小
    "max_epochs": 500,     // 最大训练轮数
    "patience": 30,        // 早停耐心值
    "target_loss": 0.2,    // 目标损失
    "label_smoothing": 0.05,
    "use_amp": true        // 是否使用混合精度
  },
  "generation": {
    "max_length": 100,     // 最大生成长度
    "temperature": 0.6,    // 采样温度
    "top_k": 5
  }
}
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建环境
conda create -n sudenmind python=3.10
conda activate sudenmind

# 安装依赖 (AMD ROCm 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# 或 CUDA 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 其他依赖
pip install jieba onnx onnxruntime netron
```

### 2. 准备数据

将对话语料放入 `data/corpus.txt`，格式为：**问题\t回答**（制表符分隔）

```
你好\t你好！很高兴见到你。
今天天气怎么样\t今天挺暖和的，适合出门走走。
```

### 3. 数据预处理

```bash
python process.py
```

这会生成：
- `data/vocab.json`: 词表
- `data/chat_data.json`: 训练数据

### 4. 训练模型

```bash
python train.py
```

**训练特性**：
- 自动使用 GPU (CUDA/ROCm)
- 混合精度训练 (FP16)
- Cosine Annealing 学习率调度
- Label Smoothing
- 早停机制 (Early Stopping)

**修改训练参数**（在 `train.py` 中）：
```python
# 学习率、batch size 等在 Trainer 类中设置
trainer = Trainer(model, chat_data, device=device, vocab_size=vocab_size, lr=5e-4)
```

### 5. 对话测试

```bash
python chat.py
```

---

## 📊 ONNX 导出与可视化

训练完成后，模型会自动导出为 ONNX 格式。你也可以手动查看：

```bash
# 安装 Netron
pip install netron

# 启动可视化
netron model/sudenmind.onnx
```

浏览器会自动打开，显示完整的模型结构图。

---

## 🎯 优化建议

### 提升生成质量

1. **增加数据量**：至少 2000+ 高质量对话对
2. **降低学习率**：如果 loss 震荡，从 `5e-4` 降到 `1e-4`
3. **增加模型容量**：增大 `d_model` 到 512，`n_layers` 到 8（需要更多显存）
4. **延长训练**：将 `target_loss` 设到 0.2 以下

### 显存优化

如果显存不足（< 8GB）：
- 减小 `batch_size` 到 32 或 16
- 减小 `d_model` 到 128
- 减小 `seq_len`（在 `train.py` 中修改 `export_to_onnx` 的默认参数）

---

## 📄 许可证

MIT License

---

## 📚 参考文献

[1] Kimi Team, et al. "Attention Residuals." arXiv preprint arXiv:2603.15031 (2026). https://arxiv.org/abs/2603.15031

## 🙏 致谢

- 使用 [jieba](https://github.com/fxsjy/jieba) 进行中文分词
- 基于 PyTorch 构建
- 参考 Kimi Team 的 Attention Residuals 论文实现跨层残差注意力机制
