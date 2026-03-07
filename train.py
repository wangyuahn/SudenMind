import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Seq2Seq, Encoder, Decoder   # 确保 model.py 中包含您的 LSTM 模型定义

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
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
    def __init__(self, model, dataloader, learning_rate=0.001, device=device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 <PAD>
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        # self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.best_loss = float('inf')  # 用于保存最佳模型

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inp, tgt in self.dataloader:
                inp, tgt = inp.to(self.device), tgt.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(inp, tgt)  # (batch, tgt_len, vocab_size)
                loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                torch.save(self.model.state_dict(), 'model/chat_model.pth')
                print(f"  -> Best model saved with loss {avg_loss:.4f}")

if __name__ == '__main__':
    # 1. 加载词汇表
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    word2id = vocab_data['word2id']
    vocab_size = len(word2id)
    print(f"词汇表大小: {vocab_size}")

    # 2. 加载数据集
    dataset = ChatDataset('processed_data.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

    # 3. 初始化模型
    embedding_dim = 512
    hidden_dim = 512
    num_layers = 1
    dropout = 0.3   # 如果需要 dropout，在 Encoder/Decoder 中启用

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device)

    # 4. 初始化训练器
    trainer = Trainer(model, dataloader, learning_rate=0.001)

    # 5. 开始训练（传入总 epoch 数）
    num_epochs = 200
    trainer.train(num_epochs)