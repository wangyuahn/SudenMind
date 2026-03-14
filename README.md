# 中文聊天机器人

一个基于 PyTorch 和 Transformer 模型的中文聊天机器人项目。

[View English README](./README_EN.md)

## 项目结构

```
SudenMind/
├── __pycache__/        # Python 编译缓存
├── model/              # 模型存储目录
│   ├── chat_model.pth  # 训练好的模型
│   └── prechat_model.pth  # 训练好的预训练模型
├── chat.py             # 聊天交互主文件
├── corpus.txt          # 原始语料库
├── model.py            # 模型定义
├── process.py          # 数据处理
├── processed_data.json # 处理后的数据
├── pretrain.py         # 模型预训练
├── knowledge.json      # 知识库
├── train.py            # 模型训练
└── vocab.json          # 词汇表
```

## 功能特性

- 基于 Transformer+Attention 模型的序列到序列生成
- 使用 jieba 进行中文分词
<!-- - 支持上下文感知的对话生成 -->
- 集成知识库进行知识增强
- 可通过 temperature 参数控制回复多样性
- 分层学习率优化训练效果

## 技术栈

- Python 3.10+
- PyTorch
- jieba (中文分词)
- JSON (数据存储)

## 安装步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/wangyuahn/SudenMind.git
   cd SudenMind
   ```

2. **安装依赖**

   ```bash
   pip install torch jieba
   ```

## 使用方法

### 1. 数据处理

如果需要使用自己的语料库，可修改 `corpus.txt` 文件，然后运行：

```bash
python process.py
```

这将生成 `processed_data.json` 和 `vocab.json` 文件。

**警告**：更改语料库后想应用更改必须重新训练所有模型。

### 2. 模型训练

如果需要重新训练模型，运行：

```bash
python pretrain.py
python train.py
```

训练完成后，预训练模型将保存在 `model/prechat_model.pth`，最终模型将保存在 `model/chat_model.pth`。

### 3. 启动聊天机器人

运行以下命令启动聊天机器人：

```bash
python chat.py
```

可以修改 `chat.py` 中的 `temperature` 参数来控制回复的多样性：

```python
output_ids = generate_response(model, input_tensor, temperature=0.6)
```

输入 `exit` 退出聊天。

## 模型参数

在 `chat.py` 和 `train.py` 中，模型参数设置如下：

```python
EMBED_SIZE = 256      # 词嵌入维度
HIDDEN_SIZE = 1024     # 隐藏层维度
NUM_LAYERS = 2       # Transformer 层数
DROPOUT = 0.5        # dropout 率
```

## 语料库格式

`corpus.txt` 文件采用以下格式（使用制表符分隔问题和回答）：

```
你好	你好，有什么可以帮助你的吗？
今天天气怎么样？
今天天气很好，适合外出。
```

## 知识库格式

`knowledge.json` 文件采用以下格式：

```json
[
    {
        "id": 1,
        "text": "量子力学是描述微观粒子运动规律的理论..."
    },
    {
        "id": 2,
        "text": "相对论是爱因斯坦提出的关于时空和引力的理论..."
    }
]
```

## 项目文件说明

- **chat.py**：聊天交互主文件，负责加载模型和处理用户输入
- **model.py**：定义 Seq2Seq 模型，使用 Transformer+Attention 架构
- **process.py**：处理原始语料库和知识库，生成训练数据和词汇表
- **pretrain.py**：在知识库上预训练模型，提高知识理解能力
- **train.py**：在对话数据上训练模型，优化对话生成能力
- **corpus.txt**：原始语料库，包含问答对
- **vocab.json**：词汇表，映射词与 ID
- **processed_data.json**：处理后的数据，用于模型训练
- **knowledge.json**：知识库，提供知识增强

## 注意事项

1. 模型训练需要一定的计算资源，建议在 GPU 环境下运行
2. 语料库和知识库质量直接影响模型性能，建议使用高质量、多样化的语料
3. 模型参数可根据实际情况进行调整以获得更好的性能
4. 语料库中问题和回答由 **\t** 分隔
5. 训练时会自动循环使用知识库数据进行预训练

## 扩展建议

1. 增加更多领域的知识库，提高模型的知识覆盖范围
2. 实现更复杂的对话管理策略，支持多轮对话
3. 添加情感分析以生成更符合语境的回复
4. 集成外部API，提供实时信息查询能力

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！
