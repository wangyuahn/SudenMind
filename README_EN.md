# SudenMind

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
</p>

<p align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/📖-中文版本-2ea44f?style=for-the-badge">
  </a>
</p>

SudenMind is a Chinese dialogue generation model based on the **AttnRes (Attention with Residual)** [[1]](#references) architecture. It features a Decoder-Only design with cross-layer residual connections, learnable positional encodings, and optimizations for Chinese conversation tasks.

---

## ✨ Key Features

- **AttnRes Architecture**: Each layer can dynamically attend to outputs from all previous layers, enabling richer information flow
- **Decoder-Only Design**: Standard autoregressive generation suitable for dialogue tasks
- **Learnable Positional Encoding**: More flexible than fixed sinusoidal encoding, adapts to different sequence lengths
- **Mixed Precision Training**: FP16 acceleration, saves memory, supports AMD ROCm
- **ONNX Export**: Export to ONNX format for deployment and visualization with Netron
- **Batch-First Format**: Follows PyTorch standards, easy to understand and use

---

## 🏗️ Model Architecture

Based on [Attention Residuals](https://arxiv.org/abs/2603.15031) Decoder-Only architecture:

```
Input (batch, seq_len)
    ↓
Embedding + Learnable Positional Encoding
    ↓
AttnRes Decoder × 6
  ├─ Self-Attention (with causal mask)
  ├─ Cross-Layer Residual Attention (AttnRes) ← Core Innovation
  │   └─ Each layer dynamically selects from all previous layer outputs via softmax attention
  └─ Feed-Forward Network
    ↓
Linear → Softmax
    ↓
Output (batch, seq_len, vocab_size)
```

**AttnRes Core Idea**:
Unlike standard Transformer's fixed residual connection (`output = x + f(x)`), AttnRes allows layer i to selectively aggregate outputs from all previous layers with learned attention weights:
```python
# Standard residual
output = fnn_out + res_out

# AttnRes: Dynamic weighted aggregation of previous layers
attn_weights = softmax(scores)  # Learn importance of each previous layer
res_out = sum(attn_weights[i] * prev_outputs[i] for i in range(n))
output = fnn_out + res_out
```

**Key Parameters** (modifiable):
- `d_model`: 256 (embedding dimension)
- `nhead`: 8 (number of attention heads)
- `d_fnn`: 512 (feed-forward network dimension)
- `n_layers`: 6 (number of AttnRes layers)
- `dropout`: 0.1

---

## 📁 File Structure

```
SudenMind/
├── data/
│   ├── corpus.txt          # Raw corpus (Question\tAnswer)
│   ├── chat_data.json      # Processed training data
│   └── vocab.json          # Vocabulary
├── model/
│   └── sudenmind.pth       # Trained model
├── model.py                # AttnRes model definition
├── datasets.py             # Dataset and data loading
├── process.py              # Data preprocessing and vocab building
├── train.py                # Training script (supports mixed precision)
├── chat.py                 # Interactive chat interface
└── README_EN.md            # This file
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create environment
conda create -n sudenmind python=3.10
conda activate sudenmind

# Install dependencies (AMD ROCm version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Or CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install jieba onnx onnxruntime netron
```

### 2. Prepare Data

Place your dialogue corpus in `data/corpus.txt`, format: **Question\tAnswer** (tab-separated)

```
Hello\tHello! Nice to meet you.
How's the weather today\tIt's quite warm today, perfect for going out.
```

### 3. Data Preprocessing

```bash
python process.py
```

This generates:
- `data/vocab.json`: Vocabulary
- `data/chat_data.json`: Training data

### 4. Train Model

```bash
python train.py
```

**Training Features**:
- Automatic GPU usage (CUDA/ROCm)
- Mixed precision training (FP16)
- Cosine Annealing learning rate scheduling
- Label Smoothing
- Early Stopping mechanism

**Modify training parameters** (in `train.py`):
```python
# Learning rate, batch size, etc. are set in the Trainer class
trainer = Trainer(model, chat_data, device=device, vocab_size=vocab_size, lr=5e-4)
```

### 5. Chat Interface

```bash
python chat.py
```

---

## 📊 ONNX Export & Visualization

After training, the model is automatically exported to ONNX format. You can also manually view it:

```bash
# Install Netron
pip install netron

# Launch visualization
netron model/sudenmind.onnx
```

Your browser will automatically open, displaying the complete model structure.

---

## 🎯 Optimization Tips

### Improving Generation Quality

1. **Increase Data**: At least 2000+ high-quality dialogue pairs
2. **Lower Learning Rate**: If loss oscillates, reduce from `5e-4` to `1e-4`
3. **Increase Model Capacity**: Increase `d_model` to 512, `n_layers` to 8 (requires more VRAM)
4. **Extend Training**: Set `target_loss` below 0.2

### Memory Optimization

If VRAM is insufficient (< 8GB):
- Reduce `batch_size` to 32 or 16
- Reduce `d_model` to 128
- Reduce `seq_len` (modify default parameter in `export_to_onnx` in `train.py`)

---

## 📄 License

MIT License

---

## 📚 References

[1] Kimi Team, et al. "Attention Residuals." arXiv preprint arXiv:2603.15031 (2026). https://arxiv.org/abs/2603.15031

## 🙏 Acknowledgements

- Uses [jieba](https://github.com/fxsjy/jieba) for Chinese word segmentation
- Built with PyTorch
- Implements cross-layer residual attention mechanism based on Kimi Team's Attention Residuals paper
