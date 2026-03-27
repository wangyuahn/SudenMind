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

SudenMind 是一个基于 **AttnRes (Attention with Residual)**  架构的中文对话生成模型。采用 Encoder-Decoder 设计，使用 BERT 作为编码器，支持跨层残差连接，针对中文对话场景优化。

---

## ✨ 核心特性

- **AttnRes 架构**：每层可动态关注之前所有层的输出，信息流动更丰富
- **MoE (混合专家) 集成**：所有 FFN 层替换为 MoE 层，4 个专家，top_k=2，提升模型容量
- **Encoder-Decoder 设计**：使用 BERT 作为编码器，AttnRes 作为解码器，提升语义理解能力
- **混合精度训练**：FP16 加速，节省显存，支持 AMD ROCm
- **ONNX 导出**：支持导出 ONNX 格式，可用 Netron 可视化模型结构
- **批量优先 (batch_first)**：符合 PyTorch 标准，易于理解和使用

---

## 🏗️ 模型架构

基于 [Attention Residuals](https://arxiv.org/abs/2603.15031) 论文的 Encoder-Decoder 架构：

```
输入 (batch, seq_len)
    ↓
BERT 编码器 (可选冻结)
  └─ 预训练 BERT 模型提取语义特征
    ↓
BERT 适配器 (768 → d_model)
    ↓
AttnRes Decoder × 6
  ├─ 自注意力 (带因果掩码)
  ├─ 跨层残差注意力 (AttnRes) ← 核心创新
  │   └─ 每层通过 softmax attention 动态选择之前所有层的输出
  └─ MoE 前馈网络 (4 个专家，top_k=2) ← 强制 MoE 集成
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

**MoE (混合专家) 集成**：
所有 FFN 层被强制替换为 MoE 层，每个 MoE 层包含 4 个专家网络，每个 token 只激活 top_k=2 个专家：
```python
# 原 FFN 层
output = ffn(x)

# 新 MoE 层
router_output = router(x)  # 计算专家权重
selected_experts = top_k(router_output, k=2)  # 选择 top-2 专家
output = sum(selected_experts[i] * expert_i(x) for i in range(2))
aux_loss = load_balancing_loss(router_output)  # 辅助损失，平衡专家负载
```
MoE 通过稀疏激活增加模型容量而不显著增加计算成本，适合大规模语言模型。

**关键参数**（可修改）：
- `d_model`: 256 (embedding 维度)
- `d_fnn`: 512 (前馈网络维度)
- `nhead`: 8 (注意力头数)
- `n_layers`: 6 (AttnRes 层数)
- `dropout`: 0.1
- `batch_fire`: False (当前使用顺序模式，非 Block AttnRes)
- `num_experts`: 4 (MoE 专家数量)
- `top_k`: 2 (每个 token 激活的专家数)
- `aux_loss_coef`: 0.01 (MoE 辅助损失系数)
- `bert_model_name`: "bert-base-chinese" (BERT 模型名称)
- `freeze_bert`: True (是否冻结 BERT 参数)
- `not_freeze_bert_num_layers`: 3 (不冻结的 BERT 层数)

---

## 📁 文件结构

```
SudenMind/
├── config.json             # ⭐ 超参数配置文件 (集中管理所有参数)
├── data/
│   └── cache/              # 数据集缓存目录
│       └── thu_coai_lccc_base_train.json  # LCCC 数据集缓存
├── model/                  # 模型保存目录
│   ├── sudenmind.pth       # 训练好的模型
│   └── sudenmind.onnx      # ONNX模型
├── src/                    # 源代码
│   ├── model.py            # AttnRes 模型定义 (包含 BERT 编码器)
│   ├── data_utils.py       # 数据集与数据加载 (LCCC 数据集)
│   ├── train.py            # 训练脚本 (支持混合精度)
│   ├── chat.py             # 交互式对话
│   ├── moe.py              # MoE 模块
│   └── view_module.py      # 模型可视化
├── tests/                  # 测试文件
│   ├── test_integration.py     # 完整集成测试
│   ├── test_bert_integration.py # BERT集成测试
│   ├── quick_test.py           # 快速测试
│   └── test_onnx_export.py     # ONNX导出测试
├── docs/                   # 文档
│   ├── BERT_INTEGRATION_SUMMARY.md
│   ├── README_EN.md
│   └── RELEASE_NOTES.md
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
    "dropout": 0.1,
    "num_experts": 4,      // MoE 专家数量
    "top_k": 2,            // 每个 token 激活的专家数
    "aux_loss_coef": 0.01,  // MoE 辅助损失系数
    "bert_model_name": "bert-base-chinese",  // BERT 模型名称
    "freeze_bert": true,   // 是否冻结 BERT 参数
    "not_freeze_bert_num_layers": 3  // 不冻结的 BERT 层数
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
  },
  "data": {
    "max_seq_len": 512,    // 最大序列长度
    "max_history": 5       // 最大历史对话轮数
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
pip install transformers datasets onnx onnxruntime netron
```

### 2. 数据准备

项目使用 **LCCC (Large-scale Cleaned Chinese Conversation)** 数据集，会自动从 Hugging Face 加载并缓存到 `data/cache/` 目录。

### 3. 训练模型

```bash
python src/train.py
```

**训练特性**：
- 自动从 Hugging Face 加载 LCCC 数据集
- 自动缓存数据集到 `data/cache/` 目录
- 自动使用 GPU (CUDA/ROCm)
- 混合精度训练 (FP16)
- Cosine Annealing 学习率调度
- Label Smoothing
- 早停机制 (Early Stopping)

### 4. 对话测试

```bash
python src/chat.py
```

**测试 BERT 分词器**：
```bash
python src/chat.py --test-tokenizer
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
[2] Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv preprint arXiv:1810.04805 (2018).

## 🙏 致谢

- 使用 [transformers](https://github.com/huggingface/transformers) 库的BERT模型
- 基于 [Attention Residuals](https://arxiv.org/abs/2603.15031) 论文实现跨层残差注意力机制
- 基于 PyTorch 构建

---
**版本**: 4.0 (Encoder-Decoder 版)
**状态**: ✅ 集成完成，测试通过

基于 **AttnRes (Attention with Residual)** 架构的中文对话生成模型，采用 Encoder-Decoder 设计，使用 BERT 作为编码器，支持跨层残差连接，针对中文对话场景优化。

## 项目结构

```
SudenMind_BERT/
├── README.md                    # 项目说明
├── config.json                  # 配置文件
├── src/                         # 源代码
│   ├── model.py                # 模型架构 (BERT集成)
│   ├── datasets.py             # 数据加载 (LCCC-base + BERT)
│   ├── train.py                # 训练脚本 (BERT集成)
│   ├── chat.py                 # 对话脚本 (BERT集成)
│   ├── moe.py                  # MoE模块
│   └── view_module.py          # 模型可视化
├── tests/                       # 测试文件
│   ├── test_integration.py     # 完整集成测试
│   ├── test_bert_integration.py # BERT集成测试
│   ├── quick_test.py           # 快速测试
│   └── test_onnx_export.py     # ONNX导出测试
├── docs/                        # 文档
│   ├── BERT_INTEGRATION_SUMMARY.md
│   ├── README_EN.md
│   └── RELEASE_NOTES.md
├── data/                        # 数据目录
│   └── cache/                  # 数据集缓存目录
│       └── thu_coai_lccc_base_train.json  # LCCC 数据集缓存
├── model/                       # 模型保存目录
│   ├── sudenmind.pth          # 模型权重
│   └── sudenmind.onnx         # ONNX模型
├── scripts/                     # 脚本工具
│   └── .github/               # GitHub Actions
└── __pycache__/                # Python缓存
```

## 核心特性

### 1. BERT集成
- **BERT编码器**: 使用 `bert-base-chinese` 作为编码器
- **参数冻结**: BERT参数部分冻结，不参与训练
- **维度适配**: BERT 768维 → 模型256维
- **WordPiece分词**: 比jieba更细粒度的分词

### 2. 模型架构
- **Encoder-Decoder 设计**: 使用 BERT 作为编码器，AttnRes 作为解码器
- **Attention Residuals (AttnRes)**: 跨层残差注意力机制
- **Mixture of Experts (MoE)**: 4个专家网络，top_k=2
- **位置编码**: 使用BERT位置编码

### 3. 训练特性
- **混合精度训练**: FP16加速，节省显存
- **Cosine Annealing**: 带Warmup的学习率调度
- **标签平滑**: 防止过拟合
- **早停机制**: 防止过拟合
- **梯度裁剪**: 防止梯度爆炸

## 快速开始

### 1. 环境准备
```bash
# 激活conda环境
conda activate pytorch291

# 安装依赖
pip install transformers datasets onnx onnxruntime netron
```

### 2. 训练模型
```bash
# 开始训练 (BERT参数部分冻结)
python src/train.py
# 数据将在训练时自动从Hugging Face加载
# LCCC-base数据集包含6,820,506个高质量中文对话
# 支持多轮对话，最大历史长度可配置
```

### 5. 对话测试
```bash
# 交互式对话
python src/chat.py

# 测试BERT分词器
python src/chat.py --test-tokenizer
```

## 配置说明

### 模型配置 (config.json)
```json
{
  "model": {
    "d_model": 256,        // Embedding 维度
    "d_fnn": 512,          // 前馈网络维度
    "nhead": 8,            // 注意力头数
    "n_layers": 6,         // 层数
    "dropout": 0.1,
    "num_experts": 4,      // MoE 专家数量
    "top_k": 2,            // 每个 token 激活的专家数
    "aux_loss_coef": 0.01,  // MoE 辅助损失系数
    "bert_model_name": "bert-base-chinese",  // BERT 模型名称
    "freeze_bert": true,   // 是否冻结 BERT 参数
    "not_freeze_bert_num_layers": 3  // 不冻结的 BERT 层数
  }
}
```

### 训练配置
```json
{
  "training": {
    "lr": 0.001,                # 学习率
    "batch_size": 64,           # 批量大小
    "max_epochs": 500,          # 最大训练轮数
    "target_loss": 0.2,         # 目标损失值
    "use_amp": true             # 使用混合精度训练
  }
}
```

## 参数统计

- **总参数**: 121,730,696
- **BERT参数**: 102,267,648 (部分冻结)
- **可训练参数**: 19,463,048
- **冻结比例**: 81.0%

## 测试脚本

### 完整测试
```bash
python tests/test_integration.py
```

### BERT集成测试
```bash
python tests/test_bert_integration.py
```

### 快速测试
```bash
python tests/quick_test.py
```

## 注意事项

### 1. 数据要求
- 对话对至少2000+对，质量越高越好
- 格式: `问题\t回答` (制表符分隔)
- 中文文本，支持标点符号

### 2. 硬件要求
- **GPU内存**: 建议8GB+ (BERT模型较大)
- **系统内存**: 建议16GB+
- **存储空间**: 建议10GB+ (用于模型和数据)

### 3. 训练建议
- BERT参数已冻结，学习率可适当提高
- 主要训练适配层和AttnResDecoder
- 监控MoE辅助损失，确保专家负载均衡

### 4. 性能优化
- 使用混合精度训练节省显存
- 调整batch_size适应显存
- 考虑使用梯度检查点

## 文件说明

### 核心文件
- `src/model.py`: BERT集成的主模型架构 (Encoder-Decoder)
- `src/data_utils.py`: LCCC-base数据加载和多轮对话处理
- `src/train.py`: 训练脚本，支持混合精度
- `src/chat.py`: 交互式对话脚本
- `src/moe.py`: MoE (Mixture of Experts) 模块
- `src/view_module.py`: 模型可视化工具

### 测试文件
- `tests/test_integration.py`: 完整集成测试
- `tests/test_bert_integration.py`: BERT功能测试
- `tests/quick_test.py`: 快速功能验证

### 文档
- `docs/BERT_INTEGRATION_SUMMARY.md`: BERT集成详细报告
- `config.json`: 所有配置参数

## 许可证

MIT License

## 致谢

- 使用 [transformers](https://github.com/huggingface/transformers) 库的BERT模型
- 基于 [Attention Residuals](https://arxiv.org/abs/2603.15031) 论文
- 参考原始 SudenMind 项目架构

## 联系方式

如有问题，请参考文档或联系开发者。

---
**版本**: 4.0 (Encoder-Decoder 版)
**状态**: ✅ 集成完成，测试通过