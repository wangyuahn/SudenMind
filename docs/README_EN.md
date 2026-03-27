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

SudenMind is a Chinese dialogue generation model based on the **AttnRes (Attention with Residual)**  architecture. It features an Encoder-Decoder design with BERT as encoder, cross-layer residual connections, and optimizations for Chinese conversation tasks.

---

## ✨ Key Features

- **AttnRes Architecture**: Each layer can dynamically attend to outputs from all previous layers, enabling richer information flow
- **MoE (Mixture of Experts) Integration**: All FFN layers replaced with MoE layers, 4 experts, top_k=2, increasing model capacity
- **Encoder-Decoder Design**: Uses BERT as encoder and AttnRes as decoder, enhancing semantic understanding capability
- **Learnable Positional Encoding**: More flexible than fixed sinusoidal encoding, adapts to different sequence lengths
- **Mixed Precision Training**: FP16 acceleration, saves memory, supports AMD ROCm
- **ONNX Export**: Export to ONNX format for deployment and visualization with Netron
- **Batch-First Format**: Follows PyTorch standards, easy to understand and use

---

## 🏗️ Model Architecture

Based on [Attention Residuals](https://arxiv.org/abs/2603.15031) Encoder-Decoder architecture:

```
Input (batch, seq_len)
    ↓
BERT Encoder (optional frozen)
  └─ Pre-trained BERT model for semantic feature extraction
    ↓
BERT Adapter (768 → embedding_dim)
    ↓
AttnRes Decoder × 6
  ├─ Self-Attention (with causal mask)
  ├─ Cross-Layer Residual Attention (AttnRes) ← Core Innovation
  │   └─ Each layer dynamically selects from all previous layer outputs via softmax attention
  └─ MoE Feed-Forward Network (4 experts, top_k=2) ← Forced MoE Integration
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

**MoE (Mixture of Experts) Integration**:
All FFN layers are forcibly replaced with MoE layers, each containing 4 expert networks, with each token activating only top_k=2 experts:
```python
# Original FFN layer
output = ffn(x)

# New MoE layer
router_output = router(x)  # Compute expert weights
selected_experts = top_k(router_output, k=2)  # Select top-2 experts
output = sum(selected_experts[i] * expert_i(x) for i in range(2))
aux_loss = load_balancing_loss(router_output)  # Auxiliary loss for expert load balancing
```
MoE increases model capacity through sparse activation without significantly increasing computational cost, suitable for large-scale language models.

**Key Parameters** (modifiable):
- `d_model`: 256 (embedding dimension)
- `d_fnn`: 512 (feed-forward network dimension)
- `nhead`: 8 (number of attention heads)
- `n_layers`: 6 (number of AttnRes layers)
- `dropout`: 0.1
- `num_experts`: 4 (MoE number of experts)
- `top_k`: 2 (number of experts activated per token)
- `aux_loss_coef`: 0.01 (MoE auxiliary loss coefficient)
- `bert_model_name`: "bert-base-chinese" (BERT model name)
- `freeze_bert`: True (whether to freeze BERT parameters)
- `not_freeze_bert_num_layers`: 3 (number of BERT layers not frozen)

---

## 📁 File Structure

```
SudenMind/
├── config.json             # ⭐ Centralized hyperparameter configuration
├── data/
│   └── cache/              # Dataset cache directory
│       └── thu_coai_lccc_base_train.json  # LCCC dataset cache
├── model/
│   └── sudenmind.pth       # Trained model
├── src/
│   ├── model.py            # AttnRes model definition (with BERT encoder)
│   ├── data_utils.py       # Dataset and data loading (LCCC dataset)
│   ├── train.py            # Training script (supports mixed precision)
│   ├── chat.py             # Interactive chat interface
│   ├── moe.py              # MoE module
│   └── view_module.py      # Model visualization
└── README_EN.md            # This file
```

---

## ⚙️ Configuration

All hyperparameters are managed in `config.json`. **No code modification needed**:

```json
{
  "model": {
    "d_model": 256,        // Embedding dimension
    "d_fnn": 512,          // Feed-forward dimension
    "nhead": 8,            // Number of attention heads
    "n_layers": 6,         // Number of layers
    "dropout": 0.1,
    "num_experts": 4,      // MoE number of experts
    "top_k": 2,            // Number of experts activated per token
    "aux_loss_coef": 0.01,  // MoE auxiliary loss coefficient
    "bert_model_name": "bert-base-chinese",  // BERT model name
    "freeze_bert": true,   // Whether to freeze BERT parameters
    "not_freeze_bert_num_layers": 3  // Number of BERT layers not frozen
  },
  "training": {
    "lr": 0.001,           // Learning rate
    "batch_size": 64,      // Batch size
    "max_epochs": 500,     // Max training epochs
    "patience": 30,        // Early stopping patience
    "target_loss": 0.2,    // Target loss
    "label_smoothing": 0.05,
    "use_amp": true        // Enable mixed precision
  },
  "generation": {
    "max_length": 100,     // Max generation length
    "temperature": 0.6,    // Sampling temperature
    "top_k": 5
  },
  "data": {
    "max_seq_len": 512,    // Max sequence length
    "max_history": 5       // Max history turns
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

# Install dependencies (AMD ROCm version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Or CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install jieba onnx onnxruntime netron transformers
```

### 2. Data Preparation

The project uses the **LCCC (Large-scale Cleaned Chinese Conversation)** dataset, which is automatically loaded from Hugging Face and cached to the `data/cache/` directory.

### 3. Train Model

```bash
python src/train.py
```

**Training Features**:
- Automatically loads LCCC dataset from Hugging Face
- Automatically caches dataset to `data/cache/` directory
- Automatic GPU usage (CUDA/ROCm)
- Mixed precision training (FP16)
- Cosine Annealing learning rate scheduling
- Label Smoothing
- Early Stopping mechanism

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

### 4. Chat Interface

```bash
python src/chat.py
```

**Test BERT Tokenizer**:
```bash
python src/chat.py --test-tokenizer
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