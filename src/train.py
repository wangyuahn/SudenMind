import warnings
import os
import sys
import logging
import math
import argparse

os.environ["TORCH_ONNX_ENABLE_DYNAMO_EXPORT"] = "0"
warnings.filterwarnings("ignore")

for name in [
    "torch",
    "torch.onnx",
    "torch.onnx._internal",
    "torch._dynamo",
    "torch.fx",
    "onnxscript",
]:
    logging.getLogger(name).disabled = True
    logging.getLogger(name).setLevel(logging.CRITICAL)


class DevNull:
    def write(self, msg):
        pass

    def flush(self):
        pass


_orig_stderr = sys.stderr
sys.stderr = DevNull()

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

sys.stderr = _orig_stderr

import json
from transformers import AutoTokenizer
from data_utils import ConversationDataset, collate_conversation_batch, IM_START, IM_END
from model import SudenMind

cfg = json.load(open("config.json", "r", encoding="utf-8"))
train_cfg = cfg["training"]
model_cfg = cfg["model"]
gen_cfg = cfg["generation"]
data_cfg = cfg["data"]


def export_to_onnx(
    model,
    vocab_size,
    device,
    seq_len=gen_cfg["onnx_seq_len"],
    save_path="model/sudenmind.onnx",
):
    """导出模型为ONNX格式"""
    model.eval()

    dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long).to(
        device
    )
    dummy_mask = torch.ones(1, seq_len, dtype=torch.bool).to(device)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_len"},
        "attention_mask": {0: "batch_size", 1: "sequence_len"},
        "logits": {0: "batch_size", 1: "sequence_len", 2: "vocab_size"},
    }

    print(f"正在导出 ONNX 模型到 {save_path}...")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _orig_stderr = sys.stderr
            sys.stderr = DevNull()

            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (dummy_input, dummy_mask),
                    save_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=gen_cfg["onnx_opset"],
                    dynamo=False,
                    fallback=True,
                    do_constant_folding=True,
                    export_params=True,
                    keep_initializers_as_inputs=False,
                    training=torch.onnx.TrainingMode.EVAL,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                )

            sys.stderr = _orig_stderr

        print(f"✓ ONNX 导出成功: {save_path}")
        print(f"  模型大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"ONNX 导出失败: {e}")
        import traceback

        traceback.print_exc()


class Trainer:
    def __init__(self, model, train_loader, device, vocab_size, lr=train_cfg["lr"]):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.vocab_size = vocab_size

        self.use_amp = torch.cuda.is_available() and train_cfg["use_amp"] == True
        self.scaler = GradScaler("cuda") if self.use_amp else None

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=train_cfg["ignore_index"],
            label_smoothing=train_cfg["label_smoothing"],
        )

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=train_cfg["weight_decay"],
            betas=tuple(train_cfg["betas"]),
        )

        self.total_steps = len(train_loader) * train_cfg["max_epochs"]
        self.warmup_steps = int(self.total_steps * train_cfg["warmup_ratio"])
        self.scheduler = self._get_cosine_schedule()
        self.global_step = 0

    def _get_cosine_schedule(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(
        self,
        epoch_num=train_cfg["max_epochs"],
        patience=train_cfg["patience"],
        target_loss=train_cfg["target_loss"],
    ):
        self.model.train()
        best_loss = float("inf")
        counter = 0

        if not os.path.exists("model"):
            os.makedirs("model")

        for epoch in range(epoch_num):
            epoch_loss = 0
            epoch_aux_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                inputs = batch["input_ids"].to(self.device)
                targets = batch["labels"].to(self.device)
                attention_masks = batch["attention_mask"].to(self.device)
                # 反转掩码：PyTorch MultiheadAttention中 True=mask掉（忽略），False=有效
                key_padding_mask = ~attention_masks
                
                self.optimizer.zero_grad()

                if self.use_amp and self.scaler is not None:
                    scaler = self.scaler
                    with autocast("cuda"):
                        output = self.model(inputs, key_padding_mask=key_padding_mask)
                        main_loss = self.criterion(
                            output.view(-1, output.size(-1)), targets.view(-1)
                        )
                        if torch.isnan(main_loss):
                            print("main_loss is nan")
                            continue
                        aux_loss = self.model.get_aux_loss()
                        loss = main_loss + aux_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=train_cfg["max_norm"]
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    output = self.model(inputs, key_padding_mask=key_padding_mask)
                    main_loss = self.criterion(
                        output.view(-1, output.size(-1)), targets.view(-1)
                    )
                    if torch.isnan(main_loss):
                        print("main_loss is nan")
                        continue
                    aux_loss = self.model.get_aux_loss()
                    loss = main_loss + aux_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=train_cfg["max_norm"]
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.global_step += 1
                epoch_loss += loss.item()
                aux_loss_value = (
                    aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
                )
                epoch_aux_loss += aux_loss_value
                num_batches += 1

                if (batch_idx + 1) % 500 == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  Step {self.global_step} | "
                        f"Total Loss: {loss.item():.4f} | "
                        f"Main Loss: {main_loss.item():.4f} | "
                        f"Aux Loss: {aux_loss_value:.4f} | "
                        f"LR: {current_lr:.2E} | "
                        f"Best Loss: {best_loss:.4f} |"
                        f"Epoch: {epoch + 1}/{epoch_num}"
                    )
                    if main_loss <= target_loss:
                        print(f"✓ 目标损失 {target_loss} 已达到！")
                        export_to_onnx(self.model, self.vocab_size, self.device)
                        self._save(main_loss, best=True)
                        break
                    

            avg_loss = epoch_loss / num_batches
            avg_aux_loss = epoch_aux_loss / num_batches
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epoch_num} - Total Loss: {avg_loss:.4f} | "
                f"Main Loss: {avg_loss - avg_aux_loss:.4f} | "
                f"Aux Loss: {avg_aux_loss:.4f} | "
                f"LR: {current_lr:.2E} | "
                f"Best Loss: {best_loss:.4f}"
            )
            if avg_loss > best_loss:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    export_to_onnx(self.model, self.vocab_size, self.device)
                    break
            elif avg_loss < best_loss:
                        best_loss = avg_loss
                        self._save(avg_loss, best=True)
                        counter = 0

        print("训练完成，导出 ONNX...")
        export_to_onnx(self.model, self.vocab_size, self.device)

        print("\n启动 Netron 可视化...")

    def _save(self, loss, best=False):
        torch.save(self.model.state_dict(), "model/sudenmind.pth")
        if best:
            print(f"  → Best model saved - Loss: {loss:.4f}")


def get_device(device_arg="auto"):
    """获取训练设备。
    
    Args:
        device_arg: "auto" 自动检测，"cpu" 强制使用 CPU，"cuda" 强制使用 GPU
    
    Returns:
        torch.device 对象
    """
    if device_arg == "cpu":
        print("强制使用 CPU 进行训练/测试")
        return torch.device("cpu")
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            print("强制使用 CUDA 进行训练")
            return torch.device("cuda")
        else:
            print("警告: CUDA 不可用，回退到 CPU")
            return torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            print("自动检测: 使用 CUDA 进行训练")
            return torch.device("cuda")
        else:
            print("自动检测: 使用 CPU 进行训练")
            return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SudenMind 训练脚本")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备: auto(自动检测), cpu(强制CPU), cuda(强制GPU)。默认: auto"
    )
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"使用设备: {device}")

    # 加载ChatGLM tokenizer
    tokenizer_name = model_cfg.get("tokenizer_name", "THUDM/chatglm-6b")
    print(f"正在加载ChatGLM tokenizer: {tokenizer_name}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )
    
    # 添加ShareGPT特殊token（与data_utils保持一致）
    tokenizer.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
    
    # 从tokenizer获取真实词表大小
    vocab_size = len(tokenizer)
    print(f"使用词表大小: {vocab_size} (原始: {vocab_size - 2}, 特殊token: 2)")

    # 初始化SudenMind模型
    model = SudenMind(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        d_fnn=model_cfg["d_fnn"],
        nhead=model_cfg.get("nhead", 8),
        dropout=model_cfg.get("dropout", 0.1),
        n_layers=model_cfg.get("n_layers", 6),
        num_experts=model_cfg.get("num_experts", 8),
        top_k=model_cfg.get("top_k", 2),
        aux_loss_coef=model_cfg.get("aux_loss_coef", 0.01),
        max_position_embeddings=model_cfg.get("max_position_embeddings", 5000),
    ).to(device)

    # 尝试加载预训练权重
    try:
        state_dict = torch.load("model/sudenmind.pth", map_location=device)

        # 检查词表大小是否匹配（处理添加特殊token后的情况）
        embedding_weight = state_dict.get("token_embedding.weight")
        if embedding_weight is not None and embedding_weight.shape[0] != vocab_size:
            print(
                f"词表大小不匹配: 权重={embedding_weight.shape[0]}, 模型={vocab_size}"
            )
            print("重新初始化 token embedding 层...")
            # 删除不匹配的权重，让模型重新初始化
            del state_dict["token_embedding.weight"]
            del state_dict["fc.3.weight"]
            del state_dict["fc.3.bias"]

        model.load_state_dict(state_dict, strict=False)
        print("成功加载预训练模型，继续训练...")
    except Exception as e:
        print(f"未找到预训练模型或加载失败: {e}")
        print("开始全新训练")

    # 加载数据集（ShareGPT格式）
    dataset = ConversationDataset(
        split=data_cfg.get("split", "train"),
        config=data_cfg.get("config", "base"),
        max_history=data_cfg.get("max_history", 5),
        max_length=data_cfg.get("max_seq_len", 512),
        tokenizer_name=tokenizer_name,
    )

    chat_data = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_conversation_batch,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=4,
    )

    trainer = Trainer(
        model, chat_data, device=device, vocab_size=vocab_size, lr=train_cfg["lr"]
    )
    trainer.train_epoch(
        epoch_num=train_cfg["max_epochs"],
        patience=train_cfg["patience"],
        target_loss=train_cfg["target_loss"],
    )
