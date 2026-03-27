import warnings
import os
import sys
import logging
import math

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
from data_utils import LCCCDataset, collate_lccc_batch
from model import SudenMind

cfg = json.load(open("config.json", "r", encoding="utf-8"))
train_cfg = cfg["training"]
model_cfg = cfg["model"]
gen_cfg = cfg["generation"]


def export_to_onnx(
    model,
    vocab_size,
    device,
    seq_len=gen_cfg["onnx_seq_len"],
    save_path="model/sudenmind.onnx",
):
    model.eval()

    dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long).to(
        device
    )

    input_names = ["input_ids"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_len"},
        "logits": {0: "batch_size", 1: "sequence_len", 2: "vocab_size"},
    }

    print(f"正在导出 ONNX 模型到 {save_path}...")
    print("提示: 导出完成后可用 Netron 打开查看模型结构")
    print("  安装: pip install netron")
    print("  使用: netron model/sudenmind.onnx")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _orig_stderr = sys.stderr
            sys.stderr = DevNull()

            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (dummy_input,),
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
                # attention_masks = None

                self.optimizer.zero_grad()

                if self.use_amp and self.scaler is not None:
                    scaler = self.scaler
                    with autocast("cuda"):
                        output = self.model(inputs, key_padding_mask=attention_masks)
                        main_loss = self.criterion(
                            output.view(-1, output.size(-1)), targets.view(-1)
                        )
                        # nan 检查
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
                    output = self.model(inputs, key_padding_mask=attention_masks)
                    main_loss = self.criterion(
                        output.view(-1, output.size(-1)), targets.view(-1)
                    )
                    # nan 检查
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

                if (batch_idx + 1) % 50 == 0:
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
                    elif main_loss < best_loss:
                        best_loss = main_loss
                        self._save(main_loss, best=True)
                        counter = 0

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
            # 平均损失大于最佳损失，早停代码
            if avg_loss > best_loss:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    export_to_onnx(self.model, self.vocab_size, self.device)
                    break

        print("训练完成，导出 ONNX...")
        export_to_onnx(self.model, self.vocab_size, self.device)

        print("\n启动 Netron 可视化...")

    def _save(self, loss, best=False):
        torch.save(self.model.state_dict(), "model/sudenmind.pth")
        if best:
            print(f"  → Best model saved - Loss: {loss:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_size = 21128
    print(f"BERT词表大小: {vocab_size}")

    bert_model_name = model_cfg.get("bert_model_name", "bert-base-chinese")
    freeze_bert = model_cfg.get("freeze_bert", True)

    model = SudenMind(
        embedding_dim=model_cfg["d_model"],
        hidden_dim=model_cfg["d_fnn"],
        output_dim=vocab_size,
        bert_model_name=bert_model_name,
        freeze_bert=freeze_bert,
        not_freeze_bert_num_layers=model_cfg.get("not_freeze_bert_num_layers", 4),
        num_experts=model_cfg.get("num_experts", 4),
        top_k=model_cfg.get("top_k", 2),
        aux_loss_coef=model_cfg.get("aux_loss_coef", 0.01),
    ).to(device)

    try:
        model.load_state_dict(torch.load("model/sudenmind.pth", map_location=device))
        print("成功加载预训练模型，继续训练...")
    except:
        print("未找到预训练模型，开始全新训练")

    data_cfg = cfg["data"]
    dataset = LCCCDataset(
        split=data_cfg.get("split", "train"),
        config=data_cfg.get("config", "base"),
        max_history=data_cfg.get("max_history", 5),
        max_length=data_cfg.get("max_seq_len", 512),
        tokenizer_name=bert_model_name,
    )

    chat_data = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_lccc_batch,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=4,
    )

    trainer = Trainer(
        model, chat_data, device=device, vocab_size=vocab_size, lr=train_cfg["lr"]
    )
    trainer.train_epoch(
        epoch_num=train_cfg["max_epochs"],
        patience=train_cfg["patience"],
        target_loss=train_cfg["target_loss"]
    )
