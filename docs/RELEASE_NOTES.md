# SudenMind v4.0.0 Release Notes

> **English** | [中文](#中文)

---

## English

### ✨ Highlights

- **Encoder-Decoder Architecture**: Added BERT as encoder, enhancing semantic understanding
- **AttnRes Architecture**: Implements cross-layer residual attention from Kimi Team paper (arXiv:2603.15031)
- **Quality Data**: 1000+ high-quality Chinese dialogue pairs (50+ chars each), removed low-quality encyclopedia entries
- **Training Optimized**: Mixed precision + Cosine Annealing + Label Smoothing for better convergence
- **ONNX Export**: Export support with Netron visualization
- **Full Type Hints**: Strict Pylance compatibility

### ⚠️ Important Notice

**Pre-trained model NOT included in source code!**

You need to:
1. Create `model/` directory manually
2. Place `sudenmind.pth` into `model/` directory  
3. Then run `python chat.py`

### Directory Structure

```
SudenMind/
├── model/              # Create this folder manually
│   └── sudenmind.pth  # Put your pre-trained model here
├── data/
│   └── cache/         # Dataset cache directory
│       └── thu_coai_lccc_base_train.json  # LCCC dataset cache
├── src/
│   ├── model.py       # AttnRes model definition (with BERT encoder)
│   ├── data_utils.py  # Dataset and data loading (LCCC dataset)
│   ├── train.py       # Training script
│   ├── chat.py        # Chat interface
│   ├── moe.py         # MoE module
│   └── view_module.py # Model visualization
└── ...
```

### Quick Start

```bash
# 1. Prepare model directory
mkdir model
# Copy your sudenmind.pth into model/

# 2. Install dependencies
pip install transformers datasets

# 3. Start chatting
python src/chat.py

# Or train from scratch
python src/train.py
```

### Breaking Changes from v3.x

| Component | v3.x | v4.0.0 | Compatible? |
|-----------|------|--------|-------------|
| Architecture | Decoder-Only | Encoder-Decoder | ❌ No |
| Encoder | None | BERT | ❌ No |
| Model Parameters | d_model, d_fnn | d_model, d_fnn | ✅ Yes |
| Weights | v3.x checkpoint | v4.0.0 checkpoint | ❌ No |

**Migration Guide**:
1. Backup old model (if any)
2. Pull v4.0.0 code
3. Install transformers and datasets dependencies
4. Retrain or download new pre-trained weights

### System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ or ROCm 6.4+
- VRAM: 12GB+ (training), 6GB+ (inference) (increased due to BERT)
- Storage: 5GB+ (increased due to BERT)

### Model Specifications

- Architecture: Encoder-Decoder (BERT + AttnRes)
- Encoder: BERT-base-chinese
- Decoder Layers: 6
- d_model: 256
- nhead: 8
- d_fnn: 512
- Position Encoding: BERT positional encoding
- Dropout: 0.1

### References

- [Attention Residuals](https://arxiv.org/abs/2603.15031) - Kimi Team, arXiv:2603.15031 (2026)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

---

## 中文

### ✨ 主要更新

- **Encoder-Decoder 架构**：添加 BERT 作为编码器，提升语义理解能力
- **AttnRes 架构**：参考 Kimi Team 论文 (arXiv:2603.15031) 实现跨层残差注意力机制
- **数据全面升级**：重写 1000+ 条高质量中文对话（每对 50+ 字），删除低质量百科条目
- **训练优化**：混合精度训练 + Cosine Annealing 学习率调度 + Label Smoothing
- **ONNX 导出**：支持导出 ONNX 格式，可用 Netron 可视化模型结构
- **完整类型注解**：通过 Pylance 严格模式检查

### ⚠️ 重要提示

**源码中不包含预训练模型！**

使用前请按以下步骤操作：
1. 手动创建 `model/` 文件夹
2. 将 `sudenmind.pth` 模型文件放入 `model/` 目录
3. 运行 `python src/chat.py` 开始对话

### 目录结构

```
SudenMind/
├── model/              # 需手动创建此文件夹
│   └── sudenmind.pth  # 放入预训练模型文件
├── data/               # 数据目录
│   └── cache/         # 数据集缓存目录
│       └── thu_coai_lccc_base_train.json  # LCCC 数据集缓存
├── src/               # 源代码目录
│   ├── model.py       # 模型定义（包含 BERT 编码器）
│   ├── train.py       # 训练脚本
│   └── chat.py        # 对话接口
└── ...
```

### 快速开始

```bash
# 1. 准备模型目录
mkdir model
# 将 sudenmind.pth 复制到 model/ 目录

# 2. 安装依赖
pip install transformers datasets

# 3. 开始对话
python src/chat.py

# 或从头训练
python src/train.py    # 训练模型（自动加载 LCCC 数据集）
```

### v3.x 迁移说明（破坏性变更）

| 组件 | v3.x | v4.0.0 | 兼容性 |
|------|------|--------|--------|
| 架构 | Decoder-Only | Encoder-Decoder | ❌ 不兼容 |
| 编码器 | 无 | BERT | ❌ 不兼容 |
| 模型参数 | d_model, d_fnn | d_model, d_fnn | ✅ 兼容 |
| 权重文件 | v3.x 检查点 | v4.0.0 检查点 | ❌ 不兼容 |

**迁移步骤**：
1. 备份旧模型（如有）
2. 拉取 v4.0.0 代码
3. 安装 transformers 和 datasets 依赖
4. 重新训练或下载新的预训练权重

### 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ 或 ROCm 6.4+
- 显存: 12GB+ (训练), 6GB+ (推理)（因 BERT 增加）
- 存储: 5GB+（因 BERT 增加）

### 模型规格

- 架构: Encoder-Decoder (BERT + AttnRes)
- 编码器: BERT-base-chinese
- 解码器层数: 6
- d_model: 256
- 注意力头数: 8
- 前馈维度: 512
- 位置编码: BERT 位置编码
- Dropout: 0.1

### 参考文献

- [Attention Residuals](https://arxiv.org/abs/2603.15031) - Kimi Team, arXiv:2603.15031 (2026)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

---

# SudenMind v3.0.0 Release Notes

> **English** | [中文](#中文)

---

## English

### ✨ Highlights

- **AttnRes Architecture**: Implements cross-layer residual attention from Kimi Team paper (arXiv:2603.15031)
- **Quality Data**: 1000+ high-quality Chinese dialogue pairs (50+ chars each), removed low-quality encyclopedia entries
- **Training Optimized**: Mixed precision + Cosine Annealing + Label Smoothing for better convergence
- **ONNX Export**: Export support with Netron visualization
- **Full Type Hints**: Strict Pylance compatibility

### ⚠️ Important Notice

**Pre-trained model NOT included in source code!**

You need to:
1. Create `model/` directory manually
2. Place `sudenmind.pth` into `model/` directory  
3. Then run `python chat.py`

### Directory Structure

```
SudenMind/
├── model/              # Create this folder manually
│   └── sudenmind.pth  # Put your pre-trained model here
├── data/
│   ├── corpus.txt
│   └── vocab.json
├── model.py
├── train.py
├── chat.py
└── ...
```

### Quick Start

```bash
# 1. Prepare model directory
mkdir model
# Copy your sudenmind.pth into model/

# 2. Start chatting
python chat.py

# Or train from scratch
python process.py
python train.py
```

### Breaking Changes from v2.x

| Component | v2.x | v3.0.0 | Compatible? |
|-----------|------|--------|-------------|
| Architecture | Standard Transformer | AttnRes | ❌ No |
| Position Encoding | Sinusoidal | Learnable | ❌ No |
| Tensor Format | (seq_len, batch) | (batch, seq_len) | ❌ No |
| Weights | v2.x checkpoint | v3.0.0 checkpoint | ❌ No |

**Migration Guide**:
1. Backup old model (if any)
2. Pull v3.0.0 code
3. Re-run `process.py` for new data format
4. Retrain or download new pre-trained weights

### System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ or ROCm 6.4+
- VRAM: 8GB+ (training), 4GB+ (inference)
- Storage: 2GB+

### Model Specifications

- Architecture: AttnRes Decoder
- Layers: 6
- d_model: 256
- nhead: 8
- d_ffn: 512
- Position Encoding: Learnable
- Dropout: 0.1

### References

- [Attention Residuals](https://arxiv.org/abs/2603.15031) - Kimi Team, arXiv:2603.15031 (2026)

---

## 中文

### ✨ 主要更新

- **AttnRes 架构**：参考 Kimi Team 论文 (arXiv:2603.15031) 实现跨层残差注意力机制
- **数据全面升级**：重写 1000+ 条高质量中文对话（每对 50+ 字），删除低质量百科条目
- **训练优化**：混合精度训练 + Cosine Annealing 学习率调度 + Label Smoothing
- **ONNX 导出**：支持导出 ONNX 格式，可用 Netron 可视化模型结构
- **完整类型注解**：通过 Pylance 严格模式检查

### ⚠️ 重要提示

**源码中不包含预训练模型！**

使用前请按以下步骤操作：
1. 手动创建 `model/` 文件夹
2. 将 `sudenmind.pth` 模型文件放入 `model/` 目录
3. 运行 `python chat.py` 开始对话

### 目录结构

```
SudenMind/
├── model/              # 需手动创建此文件夹
│   └── sudenmind.pth  # 放入预训练模型文件
├── data/               # 数据目录
│   ├── corpus.txt     # 训练语料
│   └── vocab.json     # 词表（训练后生成）
├── model.py           # 模型定义
├── train.py           # 训练脚本
├── chat.py            # 对话接口
└── ...
```

### 快速开始

```bash
# 1. 准备模型目录
mkdir model
# 将 sudenmind.pth 复制到 model/ 目录

# 2. 开始对话
python chat.py

# 或从头训练
python process.py  # 数据预处理
python train.py    # 训练模型
```

### v2.x 迁移说明（破坏性变更）

| 组件 | v2.x | v3.0.0 | 兼容性 |
|------|------|--------|--------|
| 架构 | Standard Transformer | AttnRes | ❌ 不兼容 |
| 位置编码 | 正弦编码 | 可学习编码 | ❌ 不兼容 |
| 张量格式 | (seq_len, batch) | (batch, seq_len) | ❌ 不兼容 |
| 权重文件 | v2.x 检查点 | v3.0.0 检查点 | ❌ 不兼容 |

**迁移步骤**：
1. 备份旧模型（如有）
2. 拉取 v3.0.0 代码
3. 重新运行 `process.py` 生成新格式数据
4. 重新训练或下载新的预训练权重

### 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ 或 ROCm 6.4+
- 显存: 8GB+ (训练), 4GB+ (推理)
- 存储: 2GB+

### 模型规格

- 架构: AttnRes Decoder
- 层数: 6
- d_model: 256
- 注意力头数: 8
- 前馈维度: 512
- 位置编码: 可学习 (Learnable)
- Dropout: 0.1

### 参考文献

- [Attention Residuals](https://arxiv.org/abs/2603.15031) - Kimi Team, arXiv:2603.15031 (2026)

---

## Assets | 下载资源

- Source code (zip)
- Source code (tar.gz)
- sudenmind.pth (optional | 可选)

## Full Changelog

Compare with v3.x: [View changes](https://github.com/yourusername/SudenMind/compare/v3.0.0...v4.0.0)
Compare with v2.x: [View changes](https://github.com/yourusername/SudenMind/compare/v2.x...v3.0.0)