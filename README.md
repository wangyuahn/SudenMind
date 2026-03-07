# 中文聊天机器人

一个基于 PyTorch 和 Seq2Seq 模型的中文聊天机器人项目。

## 项目结构

```
MYCHATBOT/
├── __pycache__/        # Python 编译缓存
├── model/              # 模型存储目录
│   └── chat_model.pth  # 训练好的模型
├── chat.py             # 聊天交互主文件
├── corpus.txt          # 原始语料库
├── model.py            # 模型定义
├── process.py          # 数据处理
├── processed_data.json # 处理后的数据
├── train.py            # 模型训练
└── vocab.json          # 词汇表
```

## 功能特性

- 基于 Seq2Seq 模型的序列到序列生成
- 使用 jieba 进行中文分词
- 支持上下文无关的对话生成
- 可通过添加更多语料库进行模型优化

## 技术栈

- Python 3.10+
- PyTorch
- jieba (中文分词)
- JSON (数据存储)

## 安装步骤

1. **克隆项目**

   ```bash
   git clone <项目地址>
   cd MYCHATBOT
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

### 2. 模型训练

如果需要重新训练模型，运行：

```bash
python train.py
```

训练完成后，模型将保存在 `model/chat_model.pth`。

### 3. 启动聊天机器人

运行以下命令启动聊天机器人：

```bash
python chat.py
```

输入 `exit` 退出聊天。

 dui hua。

## 模型参数

在 `chat.py` 中，模型参数设置如下：

```python
EMBED_SIZE = 64      # 词嵌入维度
HIDDEN_SIZE = 32     # 隐藏层维度
NUM_LAYERS = 2       # RNN 层数
DROPOUT = 0.4        #  dropout 率
```

## 语料库格式

`corpus.txt` 文件采用以下格式：

```
问：你好
答：你好，有什么可以帮助你的吗？
问：今天天气怎么样？
答：今天天气很好，适合外出。
```

## 项目文件说明

- **chat.py**：聊天交互主文件，负责加载模型和处理用户输入
- **model.py**：定义 Seq2Seq 模型，包括编码器和解码器
- **process.py**：处理原始语料库，生成训练数据和词汇表
- **train.py**：训练模型并保存结果
- **corpus.txt**：原始语料库，包含问答对
- **vocab.json**：词汇表，映射词与 ID
- **processed_data.json**：处理后的数据，用于模型训练

## 注意事项

1. 模型训练需要一定的计算资源，建议在 GPU 环境下运行
2. 语料库质量直接影响模型性能，建议使用高质量、多样化的语料
3. 模型参数可根据实际情况进行调整以获得更好的性能
4. 模型语料库输入和回答由\t分隔

<!-- ## 扩展建议

1. 添加注意力机制（Attention）以提高模型性能
2. 使用预训练词向量（如 Word2Vec、GloVe）
3. 实现更复杂的对话管理策略
4. 添加情感分析以生成更符合语境的回复 -->

<!-- ## 许可证

本项目采用 MIT 许可证。 -->

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！
