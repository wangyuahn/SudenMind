import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Seq2Seq, Encoder, Decoder

# 复用预训练数据集定义
import jieba
def tokenize_chinese(text):
    return list(jieba.cut(text, HMM=True))

def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ChatDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item['input'], dtype=torch.long), torch.tensor(item['target'], dtype=torch.long)

def collate_batch(batch):
    inputs  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs_padded  = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded

class Trainer:
    def __init__(self, model: Seq2Seq, dataloader: DataLoader,
                 encoder_lr: float = 1e-5, decoder_lr: float = 5e-4,
                 device: torch.device = device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 分层学习率
        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())
        self.optimizer = optim.RMSprop([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': decoder_params, 'lr': decoder_lr}
        ], weight_decay=1e-5)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)
        self.best_loss = float('inf')
        self.patience = 100
        self.counter = 0

    def train(self, epochs):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for inp, tgt in self.dataloader:
                inp, tgt = inp.to(self.device), tgt.to(self.device)

                # 问答任务前向
                self.optimizer.zero_grad()
                output = self.model(inp, tgt)
                total_loss_batch = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))

                # 总损失
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                # 累加标量值
                total_loss += total_loss_batch.item()

            self.scheduler.step()
            avg_loss = total_loss / len(self.dataloader)
        
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {avg_loss:.4f}")

            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.counter = 0
                torch.save(self.model.state_dict(), 'model/prechat_model.pth')
                print(f"  -> Best model saved with total loss {avg_loss:.4f}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("早停触发，停止训练。")
                    break

if __name__ == '__main__':
    # ---------- 1. 加载词汇表 ----------
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    word2id = vocab_data['word2id']
    vocab_size = len(word2id)
    print(f"词汇表大小: {vocab_size}")

    # ---------- 2. 加载问答数据集 ----------
    dataset = ChatDataset('processed_data.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

    # ---------- 3. 初始化模型 ----------
    embedding_dim = 256
    hidden_dim = 1024
    num_layers = 1
    dropout = 0.5

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
    model = Seq2Seq(encoder, decoder, device)

    # 加载预训练权重
    pretrained_path = 'model/prechat_model.pth'  
    try:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"已加载预训练模型: {pretrained_path}")
    except FileNotFoundError:
        print(f"未找到预训练模型 {pretrained_path}，将从头开始训练。")

    # 可选：冻结编码器（这里不冻结，用分层学习率）
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # ---------- 4. 初始化训练器 ----------
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        encoder_lr=1e-3,
        decoder_lr=1e-4,
        device=device
    )

    # ---------- 6. 开始训练 ----------
    num_epochs = 500
    trainer.train(num_epochs)