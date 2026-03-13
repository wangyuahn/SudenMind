"""
Seq2Seq 模型实现
用于聊天机器人的序列到序列转换
"""

import torch
import torch.nn as nn
import math


class Encoder(nn.Module):
    """编码器：将输入序列编码为隐藏状态"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout: float=0):
        """
        参数:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏层维度
            num_layers: Transformer层数
            dropout: dropout概率
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=4,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size

    def forward(self, x):
        """
        参数:
            x: (batch_size, seq_len) 输入序列

        返回:
            hidden: (num_layers, batch_size, embed_size) 最后一层最后一个时间步的输出（重复num_layers次）
            cell:   与hidden相同，为兼容旧接口
        """
        embedded = self.embedding(x)                     # (batch_size, seq_len, embed_size)
        embedded = self.pos_encoding(embedded)
        output = self.transformer_encoder(embedded)      # (batch_size, seq_len, embed_size)

        hidden = output[:, -1, :].unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = output[:, -1, :].unsqueeze(0).repeat(self.num_layers, 1, 1)
        return hidden, cell


class Decoder(nn.Module):
    """解码器：根据编码器隐藏状态生成目标序列"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout: float=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_size, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=4,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x, hidden, cell):
        """
        参数:
            x:      (batch_size, 1) 当前输入词ID
            hidden: (num_layers, batch_size, embed_size) 编码器隐藏状态
            cell:   (num_layers, batch_size, embed_size) 编码器细胞状态（兼容旧接口）

        返回:
            prediction: (batch_size, vocab_size) 当前步预测概率分布
            hidden:     不变，为兼容循环接口
            cell:       不变，为兼容循环接口
        """
        embedded = self.embedding(x)                     # (batch_size, 1, embed_size)
        embedded = self.pos_encoding(embedded)

        memory = hidden[-1].unsqueeze(1)                 # (batch_size, 1, embed_size) 取最后一层
        output = self.transformer_decoder(embedded, memory)  # (batch_size, 1, embed_size)
        output = output.squeeze(1)                        # (batch_size, embed_size)

        prediction = self.fc(output)                      # (batch_size, vocab_size)
        return prediction, hidden, cell


class PositionalEncoding(nn.Module):
    """位置编码：为序列添加位置信息"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)              # (max_len, 1, d_model)
        self.pe = pe  # 注意：未注册为buffer，保持与原代码一致

    def forward(self, x):
        """
        参数:
            x: (batch_size, seq_len, d_model) 输入

        返回:
            添加位置编码后的输出
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1).to(x.device)
        return self.dropout(x)


class Seq2Seq(nn.Module):
    """序列到序列模型：组合编码器和解码器"""
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.8):
        """
        参数:
            source: (batch_size, src_len) 源序列
            target: (batch_size, tgt_len) 目标序列
            teacher_forcing_ratio: 教师强制比例

        返回:
            outputs: (batch_size, tgt_len, vocab_size) 所有时间步的预测
        """
        batch_size = source.size(0)
        target_len = target.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)

        hidden, cell = self.encoder(source)
        decoder_input = target[:, 0].unsqueeze(1)          # 初始输入为<SOS>

        for t in range(1, target_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = prediction

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs