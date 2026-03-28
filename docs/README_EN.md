# SudenMind

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
</p>

<p align="center">
  <a href="../README.md">
    <img src="https://img.shields.io/badge/📖-Chinese Version-2ea44f?style=for-the-badge">
  </a>
</p>

SudenMind is a Chinese dialogue generation model based on the **AttnRes (Attention with Residual)** architecture. It features an Encoder-Decoder design with custom 2-layer AttnRes encoder + 6-layer AttnRes decoder, cross-layer residual connections, **ShareGPT/ChatML** industry-standard dialogue format, ChatGLM tokenizer integration, and optimizations for Chinese conversation tasks.

---

## ✨ Key Features

- **AttnRes Architecture**: Each layer can dynamically attend to outputs from all previous layers, enabling richer information flow
- **Custom Encoder-Decoder**: 6-layer AttnResEncoder + 6-layer AttnResDecoder, each layer accessing all previous layer outputs
- **MoE (Mixture of Experts) Integration**: All FFN layers replaced with MoE layers, 8 experts, top_k=2
- **ShareGPT/ChatML Format**: Uses industry-standard dialogue format, compatible with OpenAI/Qwen/ChatGLM2/3
- **ChatGLM Tokenizer**: Compatible with ChatGLM-6B tokenizer (vocab size 65024)
- **Mixed Precision Training**: FP16 acceleration, memory saving, AMD ROCm support
- **ONNX Export**: Supports exporting to ONNX format, viewable with Netron

---

## 🏗️ Model Architecture

Based on [Attention Residuals](https://arxiv.org/abs/2603.15031) paper with Encoder-Decoder architecture:

```
Input (batch, seq_len)
    ↓
Token Embedding + Position Encoding
    ↓
AttnRes Encoder × 2 ← Custom encoder (bidirectional attention)
  ├─ Layer 0: Bidirectional Self-Attention → MoE → output_0
  ├─ Layer 1: Bidirectional Self-Attention → MoE → AttnRes([output_0]) → output_1
  └─ ...Each layer can access all previous layer outputs
    ↓
AttnRes Decoder × 6 ← Decoder (causal attention)
  ├─ Self-Attention (with causal mask)
  ├─ Cross-layer Residual Attention (AttnRes)
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
├── model/                  # Model save directory
│   ├── sudenmind.pth       # Trained model
│   └── sudenmind.onnx      # ONNX model
├── src/                    # Source code
│   ├── model.py            # AttnRes model definition
│   ├── data_utils.py       # ShareGPT/ChatML format data processing
│   ├── train.py            # Training script
│   ├── chat.py             # Interactive dialogue
│   ├── moe.py              # MoE module
│   └── view_module.py      # Model visualization
├── docs/                   # Documentation
│   └── README_EN.md
└── README.md               # This file
```

---

## ⚙️ Configuration

All hyperparameters are managed centrally in `config.json`:

```json
{
  "model": {
    "vocab_size": 65024,
    "d_model": 512,
    "d_fnn": 1024,
    "nhead": 8,
    "n_layers": 6,
    "num_experts": 8,
    "top_k": 2,
    "tokenizer_name": "THUDM/chatglm-6b"
  },
  "training": {
    "lr": 0.0003,
    "batch_size": 8,
    "max_epochs": 200,
    "use_amp": true
  },
  "data": {
    "format": "sharegpt",
    "max_seq_len": 512,
    "max_history": 5,
    "role_markers": {
      "user": "user",
      "assistant": "assistant"
    },
    "special_tokens": {
      "im_start": "<|im_start|>",
      "im_end": "<|im_end|>",
      "gmask_id": 64790,
      "bos_id": 64792,
      "eos_id": 2
    }
  }
}
```

---

## 📝 ShareGPT/ChatML Dialogue Format

SudenMind uses the industry-standard **ShareGPT/ChatML** format:

### Training Format

```
[gMASK] [BOS] <|im_start|>user
Hello
<|im_end|>
<|im_start|>assistant
Hello! I'm happy to help you.
<|im_end|>
<|im_start|>user
How's the weather today?
<|im_end|>
<|im_start|>assistant
The weather is great today!
<|im_end|>
[EOS]
```

### Inference Format

```
[gMASK] [BOS] <|im_start|>user
Hello
<|im_end|>
<|im_start|>assistant
Hello! I'm happy to help you.
<|im_end|>
<|im_start|>user
How's the weather today?
<|im_end|>
<|im_start|>assistant
```

The model generates responses until `<|im_end|>` or `[EOS]` is encountered.

### Comparison with Industry Models

| Model | Format | Features |
|-------|--------|----------|
| **SudenMind** | ``<|im_start|>user\n{content}<|im_end|>`` | Uses ShareGPT/ChatML standard format |
| OpenAI GPT-4 | ``<|im_start|>user<|im_sep|>{content}<|im_end|>`` | Uses `<|im_sep|>` separator |
| Qwen | ``<|im_start|>user\n{content}<|im_end|>`` | Compatible with SudenMind format |
| ChatGLM2/3 | ``[gMASK] [BOS] <|user|>\n{content}<|assistant|>`` | Uses special markers and role labels |

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n sudenmind python=3.10
conda activate sudenmind

# AMD ROCm version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Or CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install transformers datasets sentencepiece onnx onnxruntime netron
```

### 2. Train Model

```bash
python src/train.py
```

**Training Features**:
- LCCC dataset auto-loading
- ShareGPT/ChatML format auto-conversion
- Mixed precision training (FP16)
- Cosine Annealing learning rate scheduling
- Early stopping

### 3. Dialogue Test

```bash
python src/chat.py
```

Dialogue example:
```
==================================================
SudenMind Dialogue System (ShareGPT/ChatML format)
==================================================
Commands: 'quit' to exit | 'clear' to clear history | 'history' to view history
--------------------------------------------------

You: Hello
Assistant: Hello! I'm happy to help you.

You: How's the weather today?
Assistant: The weather is great today, sunny and warm!
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

[1] Kimi Team, et al. "Attention Residuals." arXiv preprint arXiv:2603.15031 (2026).
[2] Du, Zhengxiao, et al. "GLM: General Language Model Pretraining with Autoregressive Blank Infilling." arXiv preprint arXiv:2103.10360 (2021).

## 🙏 Acknowledgements

- Uses [transformers](https://github.com/huggingface/transformers) library for BERT model
- Implements cross-layer residual attention mechanism based on Kimi Team's Attention Residuals paper
- Built with PyTorch

---
**Version**: 6.0 (ShareGPT/ChatML format version)
**Status**: ✅ Reconstruction Complete

## 🔄 Changelog

### v6.0 (Current version)
- **Fully reconstructed to ShareGPT/ChatML format**
- **Removed all Chinese role prefixes** ("用户:" / "助手:")
- **Using industry-standard format**: `<|im_start|>user/assistant`
- **Compatible with OpenAI/Qwen/ChatGLM2/3** and other mainstream model formats

### v5.0
- Removed BERT encoder, replaced with custom 2-layer AttnResEncoder
- ChatGLM tokenizer integration (vocab size 65024)