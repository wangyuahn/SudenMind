"""
SudenMind 数据预处理模块

负责原始语料的预处理，包括：
1. 中文分词
2. 词表构建
3. 数据格式转换
4. 训练数据生成

输入：data/corpus.txt (问题\t回答 格式)
输出：data/vocab.json, data/chat_data.json

作者：SudenMind 团队
版本：2.0
"""

import jieba
import json
import os
import sys
from typing import List

# 添加项目根目录到路径，以便导入 datasets 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import sentence_to_ids, ids_to_words


def tokenize_chinese(text):
    """返回分词后的列表"""
    return list(jieba.cut(text, HMM=True))


class Vocab:
    def __init__(self, sentences, min_freq=1):
        # 统计词频
        word_freq = {}
        for sent in sentences:
            for token in tokenize_chinese(sent):
                word_freq[token] = word_freq.get(token, 0) + 1

        # 初始化特殊标记
        self.word2id = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "<SEP>": 4}
        self.id2word = {idx: word for word, idx in self.word2id.items()}

        idx = 5
        for word, freq in word_freq.items():
            if freq >= min_freq:
                self.word2id[word] = idx
                self.id2word[idx] = word
                idx += 1

    def __len__(self):
        return len(self.word2id)

    def encode(self, sentence, add_special=True) -> List[int]:
        """将句子编码为 ID 序列"""
        ids: List[int] = sentence_to_ids(sentence, self.word2id)
        if add_special:
            ids = [self.word2id["<SOS>"]] + ids + [self.word2id["<EOS>"]]
        return ids

    def decode(self, ids, skip_special=True):
        """将 ID 序列解码为句子"""
        # 过滤特殊字符
        filtered_ids = ids
        if skip_special:
            special_ids = {
                self.word2id.get(t)
                for t in ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<SEP>"]
                if self.word2id.get(t) is not None
            }
            filtered_ids = [i for i in ids if i not in special_ids]
        words = ids_to_words(filtered_ids, self.id2word)
        return "".join(words)


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    pairs = []
    # 读取原始语料
    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            q, a = line.split("\t")
            pairs.append((q, a))

    questions = [q for q, _ in pairs]
    answers = [a for _, a in pairs]
    all_sentences = questions + answers

    vocab = Vocab(all_sentences, min_freq=1)
    print(f"词汇表大小: {len(vocab)}")

    processed_data = []
    for q, a in pairs:
        # 这里 encode 时不加特殊符号，手动拼接
        q_ids = vocab.encode(q, add_special=False)
        a_ids = vocab.encode(a, add_special=False)

        # 完整序列: [SOS] Q [SEP] A [EOS]
        full_seq = (
            [vocab.word2id["<SOS>"]]
            + q_ids
            + [vocab.word2id["<SEP>"]]
            + a_ids
            + [vocab.word2id["<EOS>"]]
        )

        # 直接在这里完成错位，彻底解放 datasets.py
        input_ids = full_seq[:-1]
        target_ids = full_seq[1:]

        processed_data.append({"input": input_ids, "target": target_ids})

    with open("data/chat_data.json", "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    vocab_dict = {
        "word2id": vocab.word2id,
        "id2word": {str(k): v for k, v in vocab.id2word.items()},
    }
    with open("data/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
