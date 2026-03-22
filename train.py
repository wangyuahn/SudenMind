"""
SudenMind 训练脚本

支持混合精度训练、早停机制、ONNX 导出等功能。
集成了 MoE (混合专家) 训练，包括辅助损失计算。

特性：
1. 混合精度训练 (FP16)：节省显存，加速训练
2. Cosine Annealing 学习率调度
3. 标签平滑：防止过拟合
4. 梯度裁剪：防止梯度爆炸
5. 早停机制：防止过拟合
6. ONNX 导出：支持模型部署和可视化
7. MoE 辅助损失：专家负载均衡

作者：SudenMind 团队
版本：2.0 (集成 MoE)
"""

import warnings
import os
import sys
import logging
import math

# ========== 环境配置 ==========
# 禁用 ONNX Dynamo 导出警告
os.environ["TORCH_ONNX_ENABLE_DYNAMO_EXPORT"] = "0"

# 忽略所有警告
warnings.filterwarnings("ignore")

# 禁用特定模块的日志，减少输出噪音
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
    """
    空输出类

    用于重定向标准错误输出，减少训练过程中的噪音信息。
    """

    def write(self, msg):
        pass

    def flush(self):
        pass


# ========== 重定向标准错误 ==========
# 临时重定向 stderr 以减少 PyTorch 导入时的警告
_orig_stderr = sys.stderr
sys.stderr = DevNull()

# ========== 导入 PyTorch 相关模块 ==========
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# ========== 恢复标准错误 ==========
sys.stderr = _orig_stderr

# ========== 导入其他模块 ==========
import json
from datasets import ChatDataset, collate_ChatDataset_batch
from model import SudenMind

# ========== 加载配置文件 ==========
# 所有超参数都集中在 config.json 中管理
cfg = json.load(open("config.json", "r", encoding="utf-8"))
train_cfg = cfg["training"]  # 训练配置
model_cfg = cfg["model"]  # 模型配置
gen_cfg = cfg["generation"]  # 生成配置


def export_to_onnx(
    model,
    vocab_size,
    device,
    seq_len=gen_cfg["onnx_seq_len"],
    save_path="model/sudenmind.onnx",
):
    """
    导出模型为 ONNX 格式

    ONNX (Open Neural Network Exchange) 是一种开放的模型格式，
    支持跨框架部署和可视化。导出的模型可以用 Netron 工具可视化。

    参数：
        model: 要导出的 PyTorch 模型
        vocab_size: 词表大小
        device: 计算设备
        seq_len: ONNX 导出的序列长度
        save_path: 保存路径

    注意：
        导出的 ONNX 模型包含完整的计算图结构，
        可以用 Netron (https://github.com/lutzroeder/netron) 可视化。
    """
    model.eval()  # 切换到评估模式

    # 创建虚拟输入 (用于跟踪计算图)
    dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long).to(
        device
    )

    # 输入输出名称 (便于 Netron 识别)
    input_names = ["input_ids"]  # 输入: token IDs (batch, seq_len)
    output_names = ["logits"]  # 输出: logits (batch, seq_len, vocab_size)

    # 动态轴说明 (支持可变批次大小和序列长度)
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
            warnings.simplefilter("ignore")  # 忽略所有警告
            _orig_stderr = sys.stderr
            sys.stderr = DevNull()  # 重定向 stderr 减少输出噪音

            # 使用 torch.onnx.export 导出模型
            with torch.no_grad():  # 禁用梯度计算
                torch.onnx.export(
                    model,
                    (dummy_input,),  # 模型输入
                    save_path,  # 保存路径
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=gen_cfg["onnx_opset"],  # ONNX opset 版本
                    dynamo=False,  # 禁用 Dynamo
                    fallback=True,  # 启用回退机制
                    do_constant_folding=True,  # 常量折叠，简化计算图
                    export_params=True,  # 导出模型参数
                    keep_initializers_as_inputs=False,  # 参数作为初始化器，图更清晰
                )

            sys.stderr = _orig_stderr

        print(f"✓ ONNX 导出成功: {save_path}")
        print(f"  模型大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"ONNX 导出失败: {e}")
        import traceback

        traceback.print_exc()


class Trainer:
    """
    训练器类

    负责管理整个训练过程，包括：
    1. 混合精度训练
    2. 学习率调度
    3. 梯度裁剪
    4. 早停机制
    5. MoE 辅助损失计算

    参数：
        model: 要训练的模型
        train_loader: 训练数据加载器
        device: 计算设备
        vocab_size: 词表大小
        lr: 学习率
    """

    def __init__(self, model, train_loader, device, vocab_size, lr=train_cfg["lr"]):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.vocab_size = vocab_size

        # 混合精度训练设置 (如果可用)
        self.use_amp = torch.cuda.is_available() and train_cfg["use_amp"] == True
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # 损失函数: 交叉熵损失 + 标签平滑
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=train_cfg["ignore_index"],  # 忽略填充 token
            label_smoothing=train_cfg["label_smoothing"],  # 标签平滑防止过拟合
        )

        # 优化器: AdamW (Adam with Weight Decay)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=train_cfg["weight_decay"],  # 权重衰减
            betas=tuple(train_cfg["betas"]),  # Adam 的 beta 参数
        )

        # Cosine Annealing 学习率调度 (带 Warmup)
        self.total_steps = len(train_loader) * train_cfg["max_epochs"]
        self.warmup_steps = int(self.total_steps * train_cfg["warmup_ratio"])
        self.scheduler = self._get_cosine_schedule()
        self.global_step = 0  # 全局训练步数

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
            num_batches = 0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # Mixed Precision Training
                if self.use_amp and self.scaler is not None:
                    scaler = self.scaler  # type: ignore
                    with autocast("cuda"):
                        output = self.model(inputs)
                        main_loss = self.criterion(
                            output.view(-1, output.size(-1)), targets.view(-1)
                        )
                        # 添加MoE辅助损失
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
                    output = self.model(inputs)
                    main_loss = self.criterion(
                        output.view(-1, output.size(-1)), targets.view(-1)
                    )
                    # 添加MoE辅助损失
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
                num_batches += 1

                if (batch_idx + 1) % 50 == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    aux_loss_value = (
                        aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
                    )
                    print(
                        f"  Step {self.global_step} | Loss: {loss.item():.4f} "
                        f"(Main: {main_loss.item():.4f}, Aux: {aux_loss_value:.4f}) | "
                        f"LR: {current_lr:.2E}"
                    )

            avg_loss = epoch_loss / num_batches
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epoch_num} - Loss: {avg_loss:.4f} - LR: {current_lr:.2E}"
            )

            if avg_loss <= target_loss:
                print(f"✓ 目标损失 {target_loss} 已达到！")
                export_to_onnx(self.model, self.vocab_size, self.device)
                self._save(avg_loss, best=True)
                break
            elif avg_loss < best_loss:
                best_loss = avg_loss
                self._save(avg_loss, best=True)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    export_to_onnx(self.model, self.vocab_size, self.device)
                    break

        # 训练结束，导出最终模型
        print("训练完成，导出 ONNX...")
        export_to_onnx(self.model, self.vocab_size, self.device)

        # 启动 Netron 可视化模型结构
        print("\n启动 Netron 可视化...")

    def _save(self, loss, best=False):
        torch.save(self.model.state_dict(), "model/sudenmind.pth")
        if best:
            print(f"  → Best model saved - Loss: {loss:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("data/vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    vocab_size = len(vocab_data["word2id"])

    model = SudenMind(
        vocab_size,
        model_cfg["d_model"],
        model_cfg["d_fnn"],
        vocab_size,
        num_experts=model_cfg.get("num_experts", 4),
        top_k=model_cfg.get("top_k", 2),
        aux_loss_coef=model_cfg.get("aux_loss_coef", 0.01),
    ).to(device)

    try:
        model.load_state_dict(torch.load("model/sudenmind.pth", map_location=device))
        print("成功加载预训练模型，继续训练...")
    except:
        print("未找到预训练模型，开始全新训练")

    chat_data = DataLoader(
        ChatDataset("data/chat_data.json"),
        batch_size=train_cfg["batch_size"],  # 从配置读取
        shuffle=True,
        collate_fn=collate_ChatDataset_batch,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    trainer = Trainer(
        model, chat_data, device=device, vocab_size=vocab_size, lr=train_cfg["lr"]
    )
    trainer.train_epoch(epoch_num=500, patience=30, target_loss=0.2)
