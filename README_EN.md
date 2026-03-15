## English Version
[English](README_EN.md) | [中文版](README.md)

### 1. Project Overview

**SudenMind** is an end-to-end chatbot based on the native PyTorch Transformer architecture. The project implements a **Pre-LN (norm_first)** structure to ensure rapid convergence and utilizes a **Q+A concatenation strategy** during preprocessing to enhance the model's grasp of contextual logic.

### 2. Key Features

* **Architectural Optimization**: Uses Pre-LayerNorm to solve the "vanishing gradient" issue in deep Transformers where Loss fails to drop.
* **Efficient Preprocessing**: Sequence concatenation logic is offloaded to `process.py`, significantly reducing CPU overhead during training.
* **Flexible Generation**: Supports `temperature` adjustment to balance response stability and creativity.
* **Training Reliability**: Includes a Linear Warmup learning rate scheduler.

### 3. Repository Structure

* `process.py`: Tokenization, vocabulary building, and data preprocessing.
* `model.py`: Definitions for `SudenMind` architecture and Positional Encoding.
* `datasets.py`: Lightweight data loading utilities.
* `train.py`: Core training loop with Warmup and Early Stopping.
* `chat.py`: Interactive chat interface.

### 4. Quick Start

1. **Prepare Data**: Place your corpus in `data/corpus.txt` (Format: `Question\tAnswer`).
2. **Data Preprocessing**:
```bash
python process.py

```


3. **Model Training**:
```bash
python train.py

```


4. **Inference/Chat**:
```bash
python chat.py

```