#!/usr/bin/env python3
"""
Seq2Seq 模型实现
用于聊天机器人的序列到序列转换
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    编码器类
    将输入序列编码为隐藏状态
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout: float=0):
        """
        初始化编码器
        
        参数:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            dropout:  dropout概率
        """
        super().__init__()
        # 词嵌入层，将单词ID转换为向量
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # LSTM层，处理序列数据
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        # 保存参数
        self.num_layers = num_layers
        self.hidden_size = hidden_size


    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列 (batch_size, seq_len)
            
        返回:
            hidden: 最后一层的隐藏状态 (num_layers, batch_size, hidden_size)
            cell: 最后一层的细胞状态 (num_layers, batch_size, hidden_size)
        """
        # 将单词ID转换为词向量
        embedded = self.embedding(x)   # (batch_size, seq_len, embed_size)
        # LSTM处理，只返回最后隐藏状态和细胞状态
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """
    解码器类
    根据编码器的隐藏状态生成输出序列
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout: float=0.0):
        """
        初始化解码器
        
        参数:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            dropout:  dropout概率
        """
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # LSTM层
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        # 全连接层，将隐藏状态映射到词汇表空间
        self.fc = nn.Linear(hidden_size, vocab_size)
        # Softmax层，获取概率分布
        # self.softmax = nn.Softmax(dim=1)


    def forward(self, x, hidden, cell):
        """
        前向传播
        
        参数:
            x: 当前输入词ID (batch_size, 1)
            hidden: 上一步的隐藏状态 (num_layers, batch_size, hidden_size)
            cell: 上一步的细胞状态 (num_layers, batch_size, hidden_size)
            
        返回:
            prediction: 当前步的预测概率分布 (batch_size, vocab_size)
        """
        # 将单词ID转换为词向量
        embedded = self.embedding(x)   # (batch_size, 1, embed_size)
        # LSTM处理
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # 移除序列维度
        output = output.squeeze(1)  # (batch_size, hidden_size)
        # 映射到词汇表空间
        prediction = self.fc(output)  # (batch_size, vocab_size)
        # 应用softmax获取概率分布
        # prediction = self.softmax(prediction)  # (batch_size, vocab_size)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    序列到序列模型
    结合编码器和解码器完成序列转换
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        """
        初始化Seq2Seq模型
        
        参数:
            encoder: 编码器实例
            decoder: 解码器实例
            device: 运行设备 (CPU或GPU)
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(self, source, target, teacher_forcing_ratio=0.8):
        """
        前向传播
        
        参数:
            source: 源序列 (batch_size, source_seq_len)
            target: 目标序列 (batch_size, target_seq_len)
            teacher_forcing_ratio: 教师强制比例
            
        返回:
            outputs: 解码器的输出序列 (batch_size, target_seq_len, vocab_size)
        """
        # 获取批次大小和目标序列长度
        batch_size = source.size(0)
        target_len = target.size(1)
        # 获取词汇表大小
        vocab_size = self.decoder.fc.out_features

        # 初始化输出张量，存储解码器每一步的预测
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)

        # 编码源序列，获取初始隐藏状态和细胞状态
        hidden, cell = self.encoder(source)

        # 解码器的第一个输入是目标序列的第一个标记（通常是<SOS>）
        decoder_input = target[:, 0].unsqueeze(1)  # (batch_size, 1)

        # 循环解码，生成目标序列
        for t in range(1, target_len):
            # 解码器前向传播
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # 存储当前时间步的预测
            outputs[:, t, :] = prediction
            
            # 教师强制：随机决定是否使用真实目标作为下一个输入
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # 获取预测概率最高的单词索引
            top1 = prediction.argmax(1)
            
            # 更新解码器的输入
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
    
        return outputs