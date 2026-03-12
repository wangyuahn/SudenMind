import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
import itertools  # 新增：用于循环迭代器
import random      # 新增：用于随机选择索引
from model import Seq2Seq, Encoder, Decoder

# 复用预训练数据集定义
import jieba
def tokenize_chinese(text):
    return list(jieba.cut(text, HMM=True))

class KnowledgeTrainDataset(Dataset):
    def __init__(self, knowledge_list, word2id):
        self.knowledge_list = knowledge_list
        self.word2id = word2id
    
    def __len__(self):
        return len(self.knowledge_list)
    
    def __getitem__(self, idx):
        k = self.knowledge_list[idx]
        tokens = tokenize_chinese(k)
        ids = [2] + [self.word2id.get(tok, self.word2id['<UNK>']) for tok in tokens] + [3]
        return torch.tensor(ids, dtype=torch.long)

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
                 knowledge_train_dataloader: Optional[DataLoader] = None,
                 knowledge_train_loss_weight: float = 0.2,
                 device: torch.device = device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.knowledge_train_dataloader = knowledge_train_dataloader
        self.knowledge_train_loss_weight = knowledge_train_loss_weight
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

        # 使用 itertools.cycle 创建无限循环的知识训练数据迭代器（如果提供）
        if self.knowledge_train_dataloader is not None:
            knowledge_train_cycle = itertools.cycle(self.knowledge_train_dataloader)
        else:
            knowledge_train_cycle = None

        for epoch in range(epochs):
            total_loss = 0.0
            total_qa_loss = 0.0
            total_knowledge_train_loss = 0.0

            for inp, tgt in self.dataloader:
                inp, tgt = inp.to(self.device), tgt.to(self.device)

                # 问答任务前向
                self.optimizer.zero_grad()
                output = self.model(inp, tgt)
                loss_qa = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))

                # 知识训练任务前向（如果启用）
                loss_knowledge_train = torch.tensor(0.0, device=self.device)
                if knowledge_train_cycle is not None:
                    knowledge_train_batch = next(knowledge_train_cycle)  # 自动循环，不会 StopIteration
                    knowledge_train_batch = knowledge_train_batch.to(self.device)
                    # 构造输入和目标：输入为前 3 个 token，目标为后面的 token
                    idx = random.randint(3, 15)
                    # len_seq = random.randint(3, 10)
                    input_knowledge_train = knowledge_train_batch[:, :idx]  # 随机截取一段作为输入
                    target_knowledge_train = knowledge_train_batch[:, idx:]  # 目标为输入后一个 token 开始所有 token
                    output_knowledge_train = self.model(input_knowledge_train, target_knowledge_train)
                    loss_knowledge_train = self.criterion(
                        output_knowledge_train.reshape(-1, output_knowledge_train.size(-1)),
                        target_knowledge_train.reshape(-1)
                    )

                # 总损失
                total_loss_batch = loss_qa + self.knowledge_train_loss_weight * loss_knowledge_train
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                # 累加标量值
                total_loss += total_loss_batch.item()
                total_qa_loss += loss_qa.item()
                total_knowledge_train_loss += loss_knowledge_train.item()

            self.scheduler.step()
            avg_loss = total_loss / len(self.dataloader)
            avg_qa_loss = total_qa_loss / len(self.dataloader)
            avg_knowledge_train_loss = total_knowledge_train_loss / len(self.dataloader) if knowledge_train_cycle else 0.0

            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {avg_loss:.4f} | QA Loss: {avg_qa_loss:.4f} | Knowledge Train Loss: {avg_knowledge_train_loss:.4f}")

            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.counter = 0
                torch.save(self.model.state_dict(), 'model/chat_model.pth')
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

    # ---------- 3. 加载知识训练数据集（知识文本） ----------
    knowledge_list = []
    with open('knowledge.json', 'r', encoding='utf-8') as f:
        knowledge = json.load(f)
        for item in knowledge:
            knowledge_list.append(item["text"])
    knowledge_train_dataset = KnowledgeTrainDataset(knowledge_list, word2id)
    knowledge_train_dataloader = DataLoader(knowledge_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # ---------- 4. 初始化模型 ----------
    embedding_dim = 256
    hidden_dim = 1024
    num_layers = 1
    dropout = 0.5

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
    model = Seq2Seq(encoder, decoder, device)

    # 加载预训练权重
    pretrained_path = 'model/chat_model.pth'  
    try:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"已加载预训练模型: {pretrained_path}")
    except FileNotFoundError:
        print(f"未找到预训练模型 {pretrained_path}，将从头开始训练。")

    # 可选：冻结编码器（这里不冻结，用分层学习率）
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # ---------- 5. 初始化训练器 ----------
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        encoder_lr=1e-3,
        decoder_lr=1e-4,
        knowledge_train_dataloader=knowledge_train_dataloader,
        knowledge_train_loss_weight=1,  # 知识训练损失权重较小，主要微调问答任务
        device=device
    )

    # ---------- 6. 开始训练 ----------
    num_epochs = 500
    trainer.train(num_epochs)