import torch
import json
import jieba
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict


class ChatDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['input'], item['target']


def collate_ChatDataset_batch(batch):
    # 因为 process.py 已经做好了错位，这里只需要转 Tensor 并 Pad 即可
    inputs = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    targets = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    
    # 保持 (batch, seq_len) 格式 - batch_first=True
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs_padded, targets_padded


def sentence_to_ids(sentence: str, word2id: Dict[str, int]) -> List[int]:
    """将句子转换为 ID 序列"""
    return [word2id.get(w, word2id.get('<UNK>', 1)) for w in jieba.cut(sentence, HMM=True)]


def ids_to_words(token_ids: List[int], id2word: Dict[int, str]) -> List[str]:
    """将 ID 序列转换为词列表"""
    return [id2word.get(i, '<UNK>') for i in token_ids]
