import torch
import json
import jieba
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Optional

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
    inputs  = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    targets = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    
    # 保持 (seq_len, batch) 格式
    inputs_padded  = pad_sequence(inputs, batch_first=False, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=False, padding_value=0)
    
    return inputs_padded, targets_padded

def transform_token(token: Optional[list]=None, sentence: Optional[str]=None,
                    id2word: Optional[dict]=None, word2id: Optional[dict]=None,
                    to: str='word'):
    if to == 'word' and token is not None and id2word is not None:
        return "".join([id2word.get(str(i), '<UNK>') for i in token])
    elif to == 'id' and sentence is not None and word2id is not None:
        return [word2id.get(w, word2id.get('<UNK>', 1)) for w in jieba.cut(sentence, HMM=True)]
    else:
        raise ValueError("错误的参数组合")