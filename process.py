import jieba
import json

def tokenize_chinese(text):
    """返回分词后的列表"""
    return list(jieba.cut(text))

class Vocab:
    def __init__(self, sentences, min_freq=1):
        # 统计词频
        word_freq = {}
        for sent in sentences:
            for token in tokenize_chinese(sent):
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # 初始化特殊标记
        self.word2id = {'<PAD>':0, '<UNK>':1, '<SOS>':2, '<EOS>':3}
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        
        idx = 4
        for word, freq in word_freq.items():
            if freq >= min_freq:
                self.word2id[word] = idx
                self.id2word[idx] = word
                idx += 1
    
    def __len__(self):
        return len(self.word2id)
    
    def encode(self, sentence, add_special=True):
        tokens = tokenize_chinese(sentence)
        ids = [self.word2id.get(tok, self.word2id['<UNK>']) for tok in tokens]
        if add_special:
            ids = [self.word2id['<SOS>']] + ids + [self.word2id['<EOS>']]
        return ids
    
    def decode(self, ids, skip_special=True):
        words = []
        for i in ids:
            w = self.id2word.get(i, '<UNK>')
            if skip_special and w in ('<PAD>','<UNK>','<SOS>','<EOS>'):
                continue
            words.append(w)
        return ''.join(words)

pairs = []
with open('MYCHATBOT/corpus.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        q, a = line.split('\t')
        pairs.append((q, a))

questions = [q for q,_ in pairs]
answers   = [a for _,a in pairs]
all_sentences = questions + answers

vocab = Vocab(all_sentences, min_freq=0)   # 过滤低频词
print(f"词汇表大小: {len(vocab)}")

processed_data = []
for q, a in pairs:
    input_ids  = vocab.encode(q, add_special=True)   # 含 <SOS> 和 <EOS>
    target_ids = vocab.encode(a, add_special=True)   # 含 <SOS> 和 <EOS>
    processed_data.append({
        'input': input_ids,
        'target': target_ids
    })


with open('MYCHATBOT/processed_data.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

# 同时保存词汇表，方便以后加载
vocab_dict = {
    'word2id': vocab.word2id,
    'id2word': {str(k):v for k,v in vocab.id2word.items()}  # JSON要求键为字符串
}
with open('MYCHATBOT/vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)