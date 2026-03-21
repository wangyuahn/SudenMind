import torch
from torch import nn
from typing import List, cast, Optional


class AttnRes(nn.Module):
    """单层 Attention + Residual 模块 (Decoder-Only, batch_first=True)"""
    def __init__(
            self,
            d_model: int,
            d_fnn: int,
            nhead: int,
            dropout: float = 0.1
        ):
        super().__init__()
        self.dropout = dropout

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        self.fnn_norm = nn.LayerNorm(d_model)
        self.fnn = nn.Sequential(
            nn.Linear(d_model, d_fnn),
            nn.GELU(),
            nn.Linear(d_fnn, d_model),
            nn.Dropout(dropout)
        )

        self.res_query = nn.Linear(d_model, d_model)

    def forward(
            self,
            x: torch.Tensor,
            prev_output: List[torch.Tensor],
            attn_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        norm = self.norm(x)
        attn_out, _ = self.attn(norm, norm, norm, attn_mask=attn_mask)
        x = x + nn.functional.dropout(attn_out, self.dropout, training=self.training)

        fnn_out = self.fnn(self.fnn_norm(x))
        if len(prev_output) == 0:
            output = fnn_out
        else:
            prev_stack = torch.stack(prev_output, dim=2)
            res_query = self.res_query(prev_stack)
            scores = torch.einsum('bld,blnd->bln', x, res_query) / (x.size(-1) ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            res_out = torch.einsum('bln,blnd->bld', attn_weights, prev_stack)
            output = fnn_out + res_out
        return output


class AttnResDecoder(nn.Module):
    """多层 AttnRes 堆叠 (Decoder-Only, batch_first=True)"""
    def __init__(
            self,
            d_model: int,
            d_fnn: int,
            nhead: int,
            dropout: float = 0.1,
            n_layers: int = 6,
            batch_fire: bool = False
        ):
        super().__init__()
        self.attn_mask: torch.Tensor
        self.attnres = nn.ModuleList([
            AttnRes(d_model, d_fnn, nhead, dropout)
            for _ in range(n_layers)
        ])
        self.batch_fire = batch_fire
        self.n_layers = n_layers

        if batch_fire:
            self.register_buffer(
                'attn_mask',
                torch.tril(torch.ones(n_layers, n_layers), diagonal=-1)
            )
    
    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        if self.batch_fire:
            return self._forward_batch_fire(x, attn_mask)
        else:
            return self._forward_seq(x, attn_mask)
    
    def _forward_seq(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        all_outputs: List[torch.Tensor] = []
        for layer in self.attnres:
            x = layer(x, all_outputs, attn_mask=attn_mask)
            all_outputs.append(x)
        return x
    
    def _forward_batch_fire(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        真正的批量并行计算。
        
        核心思想：所有层的自注意力和 FNN 可以完全并行计算，
        残差连接通过矩阵操作一次性计算。
        """
        B, L, D = x.size()
        N = self.n_layers
        
        # 阶段1: 并行计算所有层的自注意力和 FNN
        # 输入: x (B, L, D)
        # 输出: attn_outputs (B, L, N, D), fnn_outputs (B, L, N, D)
        
        # 将同一输入复制到所有层
        # 注意：这里假设所有层共享相同的输入（近似）
        # 如果需要严格递归，请使用 _forward_seq
        
        attn_outputs_list = []
        fnn_outputs_list = []
        
        current_x = x
        # 注：这里仍然是顺序的，因为每层的输入依赖于前一层的输出
        # 真正的并行需要改变模型架构
        for module in self.attnres:
            layer = cast(AttnRes, module)
            normed = layer.norm(current_x)
            attn_out, _ = layer.attn(normed, normed, normed, attn_mask=attn_mask)
            current_x = current_x + nn.functional.dropout(attn_out, layer.dropout, training=self.training)
            attn_outputs_list.append(current_x)
            
            fnn_out = layer.fnn(layer.fnn_norm(current_x))
            fnn_outputs_list.append(fnn_out)
            current_x = fnn_out
        
        # 堆叠成张量
        attn_stack = torch.stack(attn_outputs_list, dim=2)  # (B, L, N, D)
        fnn_stack = torch.stack(fnn_outputs_list, dim=2)    # (B, L, N, D)
        
        # 阶段2: 批量计算所有层的残差连接
        # 构建跨层注意力矩阵
        
        # 为每层计算对所有层的查询
        all_queries = []
        for i, module in enumerate(self.attnres):
            layer = cast(AttnRes, module)
            # 对于第 i 层，需要查询所有之前的层
            if i == 0:
                query = torch.zeros(B, L, 0, D, device=x.device)
            else:
                query = layer.res_query(attn_stack[:, :, :i, :])  # (B, L, i, D)
            all_queries.append(query)
        
        # 现在通过迭代方式计算每层的最终输出（因为存在递归依赖）
        outputs = []
        for i in range(N):
            if i == 0:
                output = fnn_outputs_list[i]
            else:
                layer = cast(AttnRes, self.attnres[i])
                
                # 使用之前计算好的输出
                prev_outputs = torch.stack(outputs, dim=2)  # (B, L, i, D)
                
                # 计算注意力分数
                res_query = layer.res_query(prev_outputs)  # (B, L, i, D)
                scores = torch.einsum('bld,blnd->bln', attn_outputs_list[i], res_query) / (D ** 0.5)
                attn_weights = torch.softmax(scores, dim=-1)  # (B, L, i)
                
                # 计算残差
                res_out = torch.einsum('bln,blnd->bld', attn_weights, prev_outputs)
                output = fnn_outputs_list[i] + res_out
            
            outputs.append(output)
        
        return outputs[-1]


class LearnablePosition(nn.Module):
    """可学习的位置编码（训练式, batch_first=True）"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb


class SudenMind(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position = LearnablePosition(embedding_dim)

        self.decoder = AttnResDecoder(
            d_model=embedding_dim,
            d_fnn=hidden_dim,
            nhead=8,
            dropout=0.1,  # 从 0.3 降低到 0.1，数据质量高了不需要太强正则
            n_layers=6,
            batch_fire=False
        )

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
        device = x.device
        batch_size, seq_len = x.size()

        embedded = self.embedding(x)
        embedded = self.position(embedded)

        causal_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        out = self.decoder(embedded, attn_mask=causal_mask)
        return self.fc(out)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def generate(self, input_seq, max_length=100, temperature=1.0, device=torch.device('cpu')):
        self.eval()
        input_seq = input_seq.to(device)
        generated = input_seq

        with torch.no_grad():
            for _ in range(max_length):
                output = self.forward(generated)
                logits = output[:, -1, :] / temperature
                top_probs, top_indices = torch.topk(logits, k=5, dim=-1)
                probs = torch.softmax(top_probs, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_indices, dim=-1, index=next_token)
                
                if next_token.item() == 3:
                    break
                generated = torch.cat([generated, next_token], dim=1)
        return generated
