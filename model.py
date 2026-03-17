import torch
from torch import nn
import math

class Position(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(Position, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 对齐 (seq_len, 1, d_model) 以适配 batch_first=False
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.pe: torch.Tensor

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0), :]

class SudenMind(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position = Position(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim,
            batch_first=False,
            dropout=0.3,
            norm_first=True  # 开启 Pre-LN 提升训练稳定性
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (seq_len, batch)
        device = x.device
        seq_len = x.size(0)

        embedded = self.embedding(x)               
        embedded = self.position(embedded)          

        # 显式生成因果掩码
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        padding_mask = (x == 0).transpose(0, 1).to(device) # (batch, seq_len)

        out = self.transformer(
            embedded,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        return self.fc(out)                   

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) == 1, diagonal=1)
        return mask 
    
    def generate(self, input_seq, max_length=100, temperature=1.0, device=torch.device('cpu')):
        self.eval()
        input_seq = input_seq.to(device) # (batch, seq_len)
        generated = input_seq

        with torch.no_grad():
            for _ in range(max_length):
                # 转置输入: (seq_len, batch)
                output = self.forward(generated.transpose(0, 1))   
                logits = output[-1, :, :] / temperature            
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  
                generated = torch.cat([generated, next_token], dim=1) 
                if next_token.item() == 3: # <EOS>
                    break
        return generated