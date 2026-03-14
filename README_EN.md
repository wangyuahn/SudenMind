# Chinese Chatbot

A Chinese chatbot project based on PyTorch and Transformer model.

[查看中文README](./README.md)

## Project Structure

```
SudenMind/
├── __pycache__/        # Python compilation cache
├── model/              # Model storage directory
│   ├── chat_model.pth  # Trained model
│   └── prechat_model.pth  # Pretrained model
├── chat.py             # Main chat interaction file
├── corpus.txt          # Raw corpus
├── model.py            # Model definition
├── process.py          # Data processing
├── processed_data.json # Processed data
├── pretrain.py         # Model pretraining
├── knowledge.json      # Knowledge base
├── train.py            # Model training
└── vocab.json          # Vocabulary
```

## Features

- Sequence-to-sequence generation based on Transformer+Attention model
- Chinese word segmentation using jieba
<!-- - Context-aware dialogue generation -->
- Knowledge base integration for knowledge enhancement
- Temperature parameter to control response diversity
- Layered learning rate optimization for training

## Tech Stack

- Python 3.10+
- PyTorch
- jieba (Chinese word segmentation)
- JSON (data storage)

## Installation

1. **Clone the project**

   ```bash
   git clone https://github.com/wangyuahn/SudenMind.git
   cd SudenMind
   ```

2. **Install dependencies**

   ```bash
   pip install torch jieba
   ```

## Usage

### 1. Data Processing

If you need to use your own corpus, modify the `corpus.txt` file, then run:

```bash
python process.py
```

This will generate `processed_data.json` and `vocab.json` files.

**Warning**: After changing the corpus, you must retrain all models to apply the changes.

### 2. Model Training

If you need to retrain the model, run:

```bash
python pretrain.py
python train.py
```

After training, the pretrained model will be saved in `model/prechat_model.pth`, and the final model will be saved in `model/chat_model.pth`.

### 3. Start the Chatbot

Run the following command to start the chatbot:

```bash
python chat.py
```

You can modify the `temperature` parameter in `chat.py` to control the diversity of responses:

```python
output_ids = generate_response(model, input_tensor, temperature=0.6)
```

Type `exit` to quit the chat.

## Model Parameters

In `chat.py` and `train.py`, the model parameters are set as follows:

```python
EMBED_SIZE = 256      # Embedding dimension
HIDDEN_SIZE = 1024     # Hidden layer dimension
NUM_LAYERS = 2       # Transformer layers
DROPOUT = 0.5        # Dropout rate
```

## Corpus Format

The `corpus.txt` file uses the following format (using tab to separate questions and answers):

```
Hello	Hello, how can I help you?
What's the weather today?
The weather is nice today, suitable for going out.
```

## Knowledge Base Format

The `knowledge.json` file uses the following format:

```json
[
    {
        "id": 1,
        "text": "Quantum mechanics is a theory that describes the motion of microscopic particles..."
    },
    {
        "id": 2,
        "text": "Relativity is a theory about space-time and gravity proposed by Einstein..."
    }
]
```

## Project File Description

- **chat.py**：Main chat interaction file, responsible for loading the model and processing user input
- **model.py**：Defines the Seq2Seq model using Transformer+Attention architecture
- **process.py**：Processes raw corpus and knowledge base, generates training data and vocabulary
- **pretrain.py**：Pretrains the model on the knowledge base to improve knowledge understanding
- **train.py**：Trains the model on dialogue data to optimize dialogue generation
- **corpus.txt**：Raw corpus containing question-answer pairs
- **vocab.json**：Vocabulary mapping words to IDs
- **processed_data.json**：Processed data for model training
- **knowledge.json**：Knowledge base for knowledge enhancement

## Notes

1. Model training requires certain computing resources, it is recommended to run in a GPU environment
2. The quality of corpus and knowledge base directly affects model performance, it is recommended to use high-quality, diverse corpus
3. Model parameters can be adjusted according to actual conditions to obtain better performance
4. Questions and answers in the corpus are separated by **\t**
5. Knowledge base data will be automatically cycled for pretraining during training

## Extension Suggestions

1. Add more domain knowledge bases to improve the model's knowledge coverage
2. Implement more complex dialogue management strategies to support multi-turn conversations
3. Add sentiment analysis to generate more context-appropriate responses
4. Integrate external APIs to provide real-time information query capabilities

## Contribution

Welcome to submit Issues and Pull Requests to improve this project!
