# SudenMind-BERT

基于BERT编码器的中文对话生成模型，集成Attention Residuals (AttnRes) 和 Mixture of Experts (MoE) 架构。

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
│   ├── raw/                    # 原始数据
│   │   └── corpus.txt         # 对话语料
│   └── processed/              # 处理后的数据
│       ├── vocab.json         # BERT词表信息
│       └── chat_data.json     # 训练数据
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
- **参数冻结**: BERT参数完全冻结，不参与训练
- **维度适配**: BERT 768维 → 模型256维
- **WordPiece分词**: 比jieba更细粒度的分词

### 2. 模型架构
- **Attention Residuals (AttnRes)**: 跨层残差注意力机制
- **Mixture of Experts (MoE)**: 4个专家网络，top_k=2
- **Decoder-Only**: 自回归生成架构
- **可学习位置编码**: 已移除，使用BERT位置编码

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

### 2. 使用LCCC-base数据
```bash
# 数据将在训练时自动从Hugging Face加载
# LCCC-base数据集包含6,820,506个高质量中文对话
# 支持多轮对话，最大历史长度可配置
```

### 4. 训练模型
```bash
# 开始训练 (BERT参数已冻结)
python src/train.py
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
    "d_model": 256,              # 模型内部维度
    "d_fnn": 512,               # MoE隐藏层维度
    "nhead": 8,                 # 注意力头数
    "n_layers": 6,              # AttnRes层数
    "dropout": 0.1,             # Dropout概率
    "num_experts": 4,           # MoE专家数量
    "top_k": 2,                 # 每个token激活的专家数
    "aux_loss_coef": 0.01,      # MoE辅助损失系数
    "bert_model_name": "bert-base-chinese",  # BERT模型
    "freeze_bert": true,        # 冻结BERT参数
    "bert_hidden_dim": 768      # BERT隐藏层维度
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
- **BERT参数**: 102,267,648 (冻结)
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
- `src/model.py`: BERT集成的主模型架构
- `src/datasets.py`: LCCC-base数据加载和多轮对话处理
- `src/train.py`: 训练脚本，支持混合精度
- `src/chat.py`: 交互式对话脚本

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
**版本**: 3.0 (BERT集成版)
**状态**: ✅ 集成完成，测试通过