import warnings
import os
import sys
import logging

# 禁用 PyTorch Dynamo 导出
os.environ['TORCH_ONNX_ENABLE_DYNAMO_EXPORT'] = '0'

# 屏蔽所有警告
warnings.filterwarnings('ignore')

# 完全禁用 torch 相关日志
for name in ['torch', 'torch.onnx', 'torch.onnx._internal', 'torch._dynamo', 'torch.fx', 'onnxscript']:
    logging.getLogger(name).disabled = True
    logging.getLogger(name).setLevel(logging.CRITICAL)

# 重定向 stderr 来屏蔽无法通过 logging 的输出
class DevNull:
    def write(self, msg): pass
    def flush(self): pass

_orig_stderr = sys.stderr
sys.stderr = DevNull()

import torch

# 恢复 stderr
sys.stderr = _orig_stderr

import json
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import ChatDataset, collate_ChatDataset_batch
from model import SudenMind


def export_to_onnx(
    model: SudenMind,
    vocab_size: int,
    device: torch.device,
    seq_len: int = 32,
    save_path: str = 'model/sudenmind.onnx'
) -> None:
    """
    将模型导出为 ONNX 格式供 Ollama 使用
    """
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long).to(device)
    
    input_names = ['input_ids']
    output_names = ['logits']
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }
    
    print(f"正在导出 ONNX 模型到 {save_path}...")
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _orig_stderr = sys.stderr
            sys.stderr = DevNull()
            # 使用 dynamo=False 禁用新版导出
            torch.onnx.export(
                model, (dummy_input,), save_path,
                input_names=input_names, output_names=output_names,
                dynamic_axes=dynamic_axes, opset_version=14,
                dynamo=False,
                fallback=True
            )
            sys.stderr = _orig_stderr
        print(f"ONNX 导出成功: {save_path}")
    except Exception as e:
        print(f"ONNX 导出失败: {e}")


class Trainer:
    def __init__(self, model, train_loader: torch.utils.data.DataLoader, device: torch.device, vocab_size: int, lr: float=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.vocab_size = vocab_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        warmup_steps = 200
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

    def train_epoch(self, epoch_num: int, patience: int=10, target_loss: float=0.5):
        self.model.train()
        best_loss = float('inf')
        counter = 0
        total_steps = 0
        
        if not os.path.exists('model'):
            os.makedirs('model')
        
        for epoch in range(epoch_num):
            total_loss = 0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if total_steps < 200:
                    self.warmup_scheduler.step()
                total_steps += 1
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            self.scheduler.step(avg_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epoch_num} - Loss: {avg_loss:.4f} - LR: {current_lr:.2E}")
            
            if avg_loss <= target_loss:
                print(f"目标损失 {target_loss} 已达到，提前停止训练。")
                torch.save(self.model.state_dict(), 'model/sudenmind.pth')
                print(f"-> Best model saved - Loss: {avg_loss:.4f}")
                export_to_onnx(self.model, self.vocab_size, self.device)
                break
            elif avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'model/sudenmind.pth')
                print(f"-> Best model saved - Loss: {best_loss:.4f}")
                # export_to_onnx(self.model, self.vocab_size, self.device)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    export_to_onnx(self.model, self.vocab_size, self.device)
                    break

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  # 输出当前使用的设备
    with open('data/vocab.json', 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    word2id = vocab_data['word2id']
    vocab_size = len(word2id)
    
    embedding_dim = 256
    hidden_dim = 512
    output_dim = vocab_size

    model = SudenMind(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    
    try:
        model.load_state_dict(torch.load('model/sudenmind.pth', map_location=device))
        print("成功加载预训练模型")
    except:
        print("未找到预训练模型，开始全新训练")
    export_to_onnx(model, vocab_size, device)
    chat_data = DataLoader(
        ChatDataset('data/chat_data.json'),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_ChatDataset_batch
    )
    
    trainer = Trainer(model, chat_data, device=device, vocab_size=vocab_size)
    trainer.train_epoch(500, patience=100, target_loss=0.5)
