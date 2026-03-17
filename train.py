import torch
import json
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import ChatDataset, collate_ChatDataset_batch
from model import SudenMind
from safetensors.torch import save_file

class Trainer:
    def __init__(self, model, train_loader: torch.utils.data.DataLoader, device: torch.device, lr: float=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # 使用 1e-4 的学习率
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 增加 Warmup 预热机制
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
                output = self.model(inputs) # (seq_len, batch, vocab_size)
                
                # 计算损失
                loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 更新 Warmup
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
                tensors = model.state_dict()
                save_file(tensors, "model/sudenmind.safetensors")
                print(f"-> Best model saved - Loss: {avg_loss:.4f}")
                break
            elif avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'model/sudenmind.pth')
                tensors = model.state_dict()
                save_file(tensors, "model/sudenmind.safetensors")
                print(f"-> Best model saved - Loss: {best_loss:.4f}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('data/vocab.json', 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    word2id = vocab_data['word2id']
    vocab_size = len(word2id)
    
    # 保持你原来的参数设置
    embedding_dim = 256
    hidden_dim = 512
    output_dim = vocab_size

    model = SudenMind(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    
    try:
        model.load_state_dict(torch.load('model/sudenmind.pth', map_location=device))
        print("成功加载预训练模型")
    except:
        print("未找到预训练模型，开始全新训练")

    chat_data = DataLoader(
        ChatDataset('data/chat_data.json'),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_ChatDataset_batch
    )
    
    trainer = Trainer(model, chat_data, device=device)
    trainer.train_epoch(200)