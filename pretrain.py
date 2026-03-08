import torch
import json
import jieba
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Seq2Seq, Encoder, Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize_chinese(text):
    """返回分词后的列表"""
    return list(jieba.cut(text, HMM=True))

class PretrainDataset(Dataset):
    def __init__(self, knowledge_list):
        self.knowledge_list = knowledge_list
        with open('vocab.json', 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.word2id = vocab_data['word2id']
    
    def __len__(self):
        return len(self.knowledge_list)
    
    def __getitem__(self, idx):
        k = self.knowledge_list[idx]
        k_ids = torch.tensor(self.encode_sentence(k), dtype=torch.long)
        return k_ids
    
    def encode_sentence(self, sentence):
        ids = [2]+[self.word2id.get(tok, self.word2id['<UNK>']) for tok in tokenize_chinese(sentence)]+[3]
        return ids

def collate_fn(batch):
    """自定义的collate_fn函数，用于处理不同长度的序列"""
    # 使用pad_sequence对序列进行填充，padding_value=0对应<PAD>标记
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

def pretrain(model, dataloader, epochs=10, learning_rate=0.001):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # 忽略<PAD>标记的损失计算
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            input_seq: torch.Tensor = batch[:, :-3]  # 输入序列，去掉最后三个标记
            target_seq: torch.Tensor = batch[:, 3:]   # 目标序列，去掉前三个标记
            optimizer.zero_grad()
            output: torch.Tensor = model(input_seq, target_seq)  # 输入和目标都是知识文本
            loss = criterion(output.reshape(-1, output.size(-1)), target_seq.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if avg_loss < 0.5:
            print("-> Loss < 0.5, stop pretraining.")
            torch.save(model.state_dict(), 'model/pretrained_model.pth')
            print("-> Save best model with Loss: {:.4f}".format(best_loss))
            break
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'model/pretrained_model.pth')
            print("-> Save best model with Loss: {:.4f}".format(best_loss))
        
if __name__ == "__main__":
    knowledge_list = []
    with open('knowledge.json', 'r', encoding='utf-8') as f:
        knowledge = json.load(f)
        for item in knowledge:
            knowledge_list.append(item["text"])
    dataset = PretrainDataset(knowledge_list)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # 初始化模型
    encoder = Encoder(vocab_size=len(dataset.word2id), embed_size=256, 
                      hidden_size=1024, num_layers=1, dropout=0.2).to(device)
    
    decoder = Decoder(vocab_size=len(dataset.word2id), embed_size=256, 
                      hidden_size=1024, num_layers=1, dropout=0.2).to(device)
    
    model = Seq2Seq(encoder, decoder, device).to(device)

    pretrain(model, dataloader, epochs=200, learning_rate=1e-4)

