# SudenMind

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
  <img src="https://img.shields.io/badge/Website-ai.sufun.space-blue.svg">
</p>

<p align="center">
  <a href="../README.md">
    <img src="https://img.shields.io/badge/📖-Chinese%20Version-2ea44f?style=for-the-badge">
  </a>
</p>

SudenMind is a Chinese dialogue generation model based on the **AttnRes (Attention Residuals)** architecture. It features an Encoder-Decoder design with custom 2-layer AttnRes encoder + 6-layer AttnRes decoder, cross-layer residual connections, **ShareGPT/ChatML** industry-standard dialogue format, ChatGLM tokenizer integration, and optimizations for Chinese conversation tasks. Enhanced with gating mechanism (Gate) and KV cache inference acceleration.

---

## ✨ Key Features

- **AttnRes Architecture**: Each layer can dynamically attend to outputs from all previous layers, enabling richer information flow
- **Custom Encoder-Decoder**: 2-layer AttnResEncoder + 6-layer AttnResDecoder, each layer accessing all previous layer outputs
- **Gating Mechanism (Gate)**: Innovative gating network dynamically balances MoE output with cross-layer residual output, adapting feature fusion ratios based on input characteristics
- **KV Cache Inference**: Supports key-value caching for significantly accelerated autoregressive generation
- **MoE (Mixture of Experts) Integration**: All FFN layers replaced with MoE layers, 8 experts, top_k=2
- **ShareGPT/ChatML Format**: Uses industry-standard dialogue format, compatible with OpenAI/Qwen/ChatGLM2/3
- **ChatGLM Tokenizer**: Compatible with ChatGLM-6B tokenizer (vocab size 65024)
- **Mixed Precision Training**: FP16 acceleration, memory saving, AMD ROCm support

## 📝 ShareGPT/ChatML Dialogue Format

SudenMind uses ChatML format managed by `src/data_utils.py` with these rules:

- Start with `[BOS]`, end with `[EOS]`.
- Each turn is wrapped as `<|im_start|>{role}\n...<|im_end|>`.
- Only assistant text in `<|im_start|>assistant ... <|im_end|>` participates in loss; all role headers, `<|im_end|>`, newline separators are masked as `-100`.

### Training example

```text
[BOS] <|im_start|>user
Hello
<|im_end|>
<|im_start|>assistant
Hi there!
<|im_end|>
<|im_start|>user
How are you?
<|im_end|>
<|im_start|>assistant
I'm fine, thank you.
<|im_end|>
[EOS]
```

### Inference example

```text
[BOS] <|im_start|>user
Hello
<|im_end|>
<|im_start|>assistant
Hi there!
<|im_end|>
<|im_start|>user
How are you?
<|im_end|>
<|im_start|>assistant
```

Model continues generating assistant content until `<|im_end|>` or `[EOS]`.

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
  ├─ Gated Feature Fusion (Gate)
  └─ MoE Feed-Forward Network (8 experts, top_k=2)
    ↓
Linear → Softmax
    ↓
Output (batch, seq_len, vocab_size=65024)
```

### AttnRes Gating Mechanism Explained

In each AttnRes layer, the gating network computes:

```
Gate = Sigmoid([MoE_out; Res_out] * W_g)
Output = (1 - Gate) ⊗ MoE_out + Gate ⊗ Res_out
```

Where:
- MoE_out: Mixture of Experts feed-forward network output
- Res_out: Cross-layer residual attention output
- W_g: Learnable gating weight matrix
- ⊗: Element-wise multiplication

This mechanism allows the model to dynamically decide, based on input semantic features, whether to emphasize innovative features (MoE output) or retain historical information (residual output), significantly enhancing modeling flexibility.

### KV Cache Inference

To accelerate autoregressive generation, the model caches historical key-value pairs (Key-Value pairs) during decoding:

- Encoder output key-value pairs are computed and cached during the first forward pass
- At each decoding step, only the current token's key-value pairs need to be computed and appended to the cache
- This avoids redundant computation of historical token key-value pairs, reducing time complexity from O(n²) to O(n)

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
│   ├── model.py            # AttnRes model definition (with gating mechanism and KV cache)
│   ├── data_utils.py       # ShareGPT/ChatML format data processing
│   ├── train.py            # Training script
│   ├── chat.py             # Interactive dialogue (with KV cache inference)
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
    "d_model": 512,
    "d_fnn": 1024,
    "nhead": 8,
    "n_layers": 6,
    "dropout": 0.1,
    "max_position_embeddings": 5000,
    "num_experts": 8,
    "top_k": 2,
    "aux_loss_coef": 0.01,
    "tokenizer_name": "THUDM/chatglm-6b"
  },
  "training": {
    "lr": 0.0003,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "eps": 1e-08,
    "warmup_ratio": 0.05,
    "max_epochs": 200,
    "label_smoothing": 0,
    "ignore_index": -100,
    "batch_size": 1,
    "max_norm": 5.0,
    "patience": 30,
    "target_loss": 0.03,
    "use_amp": false
  },
  "data": {
    "config": "base",
    "split": "train",
    "max_history": 5,
    "max_seq_len": 512
  },
  "generation": {
    "max_length": 200,
    "temperature": 0.3,
    "top_k": 50,
    "onnx_seq_len": 32,
    "onnx_opset": 11
  },
  "paths": {}
}
```

> Note: The current config no longer explicitly stores special token markers (`<|im_start|>`, `<|im_end|>`)—they are handled via tokenizer logic in `src/data_utils.py`.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n sudenmind python=3.10
conda activate sudenmind

# AMD ROCm version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Or CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Other dependencies (hard version requirements)
pip install transformers>=4.30.0 datasets>=2.0.0 sentencepiece>=0.1.99 onnx>=1.14.0 onnxruntime>=1.15.0 netron>=7.0.0
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

### 3. Dialogue Test (with KV Cache Inference)

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

---

## 🙏 Acknowledgements

- Uses [transformers](https://github.com/huggingface/transformers) library for BERT model
- Implements cross-layer residual attention mechanism based on Kimi Team's Attention Residuals paper
- Built with PyTorch

---

## 🔗 Project Links

- **Official Website**: https://ai.sufun.space
- **GitHub**: https://github.com/wangyuahn/SudenMind
- **Documentation**: https://ai.sufun.space/docs

---

## 📄 Version Information

**Version**: 6.1 (AttnRes Gate & KV Cache Version)  
**Status**: ✅ Feature Enhancement Complete  
**Last Updated**: 2026-03-29

## 🔄 Changelog

### v6.1 (Current Version)
- **Removed gMASK token**: Simplified dialogue format, keeping only [BOS] as the sequence start token
- **Added Gating Mechanism (Gate)**: Dynamic balancing of MoE output with cross-layer residual output
- **Added KV Cache Inference**: Significant acceleration of autoregressive generation
- **Optimized Model Architecture**: Improved AttnRes layer information flow mechanism
- **Updated Website Link**: Official site migrated to ai.sufun.space

### v6.0
- **Fully reconstructed to ShareGPT/ChatML format**
- **Removed all Chinese role prefixes** ("用户:" / "助手:")
- **Using industry-standard format**: `授user/受assistant`
- **Compatible with OpenAI/Qwen/ChatGLM2/3** and other mainstream model formats

### v5.0
- Removed BERT encoder, replaced with custom 2-layer AttnResEncoder
- ChatGLM tokenizer integration (vocab size 65024)