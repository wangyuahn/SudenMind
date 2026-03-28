# SudenMind

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
</p>

SudenMind is a Chinese dialogue generation model based on the **AttnRes (Attention with Residual)** architecture. It features a custom Encoder-Decoder design with AttnResEncoder (2 layers) and AttnResDecoder (6 layers), cross-layer residual connections, ChatGLM tokenizer integration, and optimizations for Chinese conversation tasks.

---

## ✨ Key Features

- **AttnRes Architecture**: Each layer can dynamically attend to outputs from all previous layers, enabling richer information flow
- **Custom AttnResEncoder**: 6-layer encoder with bidirectional attention, each layer accessing all previous layer outputs
- **MoE (Mixture of Experts) Integration**: All FFN layers replaced with MoE layers (8 experts, top_k=2) for increased model capacity
- **ChatGLM Tokenizer**: Compatible with ChatGLM-6B tokenizer (vocab size 65024)
- **Mixed Precision Training**: FP16 acceleration, memory saving, AMD ROCm support
- **ONNX Export**: Supports exporting to ONNX format, viewable with Netron
- **Batch First**: Follows PyTorch standards, easy to understand and use

---

## 🏗️ Model Architecture

Based on the [Attention Residuals](https://arxiv.org/abs/2603.15031) paper with ChatGLM integration:

```
Input (batch, seq_len)
    ↓
Token Embedding + Position Encoding
    ↓
AttnRes Encoder × 2 (Bidirectional Attention)
  ├─ Layer 0: Bidirectional Self-Attn → MoE → output_0
  ├─ Layer 1: Bidirectional Self-Attn → MoE → AttnRes([output_0]) → output_1
  └─ ...Each layer can access all previous layer outputs
    ↓
AttnRes Decoder × 6 (Causal Attention)
  ├─ Self-Attention (with causal mask)
  ├─ Cross-layer Residual Attention (AttnRes) ← Core innovation
  │   └─ Each layer dynamically selects outputs from all previous layers via softmax attention
  └─ MoE Feed-Forward Network (8 experts, top_k=2)
    ↓
Linear → Softmax
    ↓
Output (batch, seq_len, vocab_size=65024)
```

**AttnRes Core Idea**:
Unlike standard Transformer's fixed residual connections (`output = x + f(x)`), AttnRes allows layer i to selectively aggregate outputs from all previous layers through learned attention weights:

```python
# Standard residual
output = fnn_out + res_out

# AttnRes: Dynamic weighted aggregation of previous layer outputs
attn_weights = softmax(scores)  # Learn importance of each previous layer
res_out = sum(attn_weights[i] * prev_outputs[i] for i in range(n))
output = fnn_out + res_out
```

**Key Parameters** (modifiable in config.json):
- `vocab_size`: 65024 (ChatGLM vocabulary)
- `d_model`: 512 (embedding dimension)
- `d_fnn`: 1024 (feed-forward network dimension)
- `nhead`: 8 (number of attention heads)
- `n_layers`: 6 (number of encoder/decoder layers)
- `dropout`: 0.1
- `num_experts`: 8 (MoE expert count)
- `top_k`: 2 (experts activated per token)
- `aux_loss_coef`: 0.01 (MoE auxiliary loss coefficient)
- `tokenizer_name`: "THUDM/chatglm-6b" (ChatGLM tokenizer)

---

## 📁 File Structure

```
SudenMind/
├── config.json             # ⭐ Hyperparameter configuration
├── data/
│   └── cache/              # Dataset cache directory
│       └── lccc_chatglm_base_train.json  # LCCC dataset cache (ChatGLM format)
├── model/                  # Model save directory
│   ├── sudenmind.pth       # Trained model
│   └── sudenmind.onnx      # ONNX model
├── src/                    # Source code
│   ├── model.py            # AttnRes model (custom 6-layer encoder + 6-layer decoder)
│   ├── data_utils.py       # Dataset and data loading (ChatGLM tokenizer)
│   ├── train.py            # Training script (mixed precision support)
│   ├── chat.py             # Interactive dialogue
│   ├── moe.py              # MoE module
│   └── view_module.py      # Model visualization
├── tests/                  # Test files
│   ├── test_integration.py     # Full integration test
│   ├── quick_test.py           # Quick test
│   └── test_onnx_export.py     # ONNX export test
├── docs/                   # Documentation
│   └── README_EN.md        # English documentation
└── README.md               # This file (Chinese)
```

---

## ⚙️ Configuration

All hyperparameters are managed centrally in `config.json`, no code modification needed:

```json
{
  "model": {
    "vocab_size": 65024,   // ChatGLM vocabulary size
    "d_model": 512,        // Embedding dimension
    "d_fnn": 1024,         // Feed-forward network dimension
    "nhead": 8,            // Attention heads
    "n_layers": 6,         // Encoder/Decoder layers. Encoder layer count = layer//3
    "dropout": 0.1,
    "num_experts": 8,      // MoE expert count
    "top_k": 2,            // Experts activated per token
    "aux_loss_coef": 0.01,  // MoE auxiliary loss coefficient
    "tokenizer_name": "THUDM/chatglm-6b"  // ChatGLM tokenizer
  },
  "training": {
    "lr": 0.0003,          // Learning rate
    "batch_size": 8,       // Batch size
    "max_epochs": 200,     // Max training epochs
    "patience": 30,        // Early stopping patience
    "target_loss": 0.03,   // Target loss
    "label_smoothing": 0,
    "use_amp": true        // Use mixed precision
  },
  "generation": {
    "max_length": 100,     // Max generation length
    "temperature": 0.3,    // Sampling temperature
    "top_k": 50
  },
  "data": {
    "max_seq_len": 256,    // Max sequence length
    "max_history": 4       // Max conversation history turns
  }
}
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create environment
conda create -n sudenmind python=3.10
conda activate sudenmind

# Install PyTorch (AMD ROCm version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Or CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install transformers datasets sentencepiece onnx onnxruntime netron
```

### 2. Data Preparation

The project uses the **LCCC (Large-scale Cleaned Chinese Conversation)** dataset, automatically loaded from Hugging Face and cached to `data/cache/` directory.

### 3. Train Model

```bash
python src/train.py
```

**Training Features**:
- Automatically load LCCC dataset from Hugging Face
- Automatically cache dataset to `data/cache/` directory
- Automatic GPU usage (CUDA/ROCm)
- Mixed precision training (FP16)
- Cosine Annealing learning rate scheduling
- Early stopping

### 4. Chat Test

```bash
python src/chat.py
```

**Test ChatGLM Tokenizer**:
```bash
python src/chat.py --test-tokenizer
```

---

## 📊 ONNX Export & Visualization

After training, the model is automatically exported to ONNX format. You can also manually view:

```bash
# Install Netron
pip install netron

# Launch visualization
netron model/sudenmind.onnx
```

The browser will automatically open, displaying the complete model structure.

---

## 🎯 Optimization Tips

### Improve Generation Quality

1. **Increase Data**: At least 2000+ high-quality dialogue pairs
2. **Reduce Learning Rate**: If loss oscillates, reduce from `5e-4` to `1e-4`
3. **Increase Model Capacity**: Increase `d_model` to 1024, `n_layers` to 8 (requires more VRAM)
4. **Extend Training**: Set `target_loss` below 0.03

### VRAM Optimization

If VRAM is insufficient (< 8GB):
- Reduce `batch_size` to 4 or 2
- Reduce `d_model` to 256
- Reduce `max_seq_len` in `config.json`

---

## 📄 License

MIT License

---

## 📚 References

[1] Kimi Team, et al. "Attention Residuals." arXiv preprint arXiv:2603.15031 (2026). https://arxiv.org/abs/2603.15031
[2] Du, Zhengxiao, et al. "GLM: General Language Model Pretraining with Autoregressive Blank Infilling." arXiv preprint arXiv:2103.10360 (2021).

## 🙏 Acknowledgements

- Uses [transformers](https://github.com/huggingface/transformers) library's ChatGLM tokenizer
- Based on [Attention Residuals](https://arxiv.org/abs/2603.15031) paper for cross-layer residual attention mechanism
- Based on [GLM](https://github.com/THUDM/GLM) architecture
- Built with PyTorch

---
**Version**: 5.0 (ChatGLM + Custom AttnResEncoder)
**Status**: ✅ Transformation Complete

## 🔄 Changelog

### v5.0 (Current)
- **Removed BERT encoder**, replaced with custom 2-layer AttnResEncoder
- **Encoder layers can access all previous layer outputs** (AttnRes mechanism)
- **ChatGLM tokenizer integration** (vocab size 65024)
- **Data format changed to ChatGLM format**: `[gMASK] [sop] input [eos]`
