import torch
from torch import nn
from typing import List, Optional
from moe import MoELayer


class AttnRes(nn.Module):
    """
    AttnRes (Attention with Residual) 模块
    允许当前层通过注意力机制动态选择之前所有层的输出
    """

    def __init__(
        self,
        d_model: int,
        d_fnn: int,
        nhead: int,
        dropout: float = 0.1,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_experts = num_experts
        self.top_k = top_k

        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        self.fnn_norm = nn.LayerNorm(d_model)
        self.moe = MoELayer(
            d_model=d_model,
            d_ff=d_fnn,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            use_aux_loss=True,
            aux_loss_coef=aux_loss_coef,
        )

        self.res_query = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        prev_output: List[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm = self.norm(x)
        attn_out, _ = self.attn(
            norm, norm, norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = x + nn.functional.dropout(attn_out, self.dropout, training=self.training)

        moe_input = self.fnn_norm(x)
        moe_out = self.moe(moe_input)

        if len(prev_output) == 0:
            output = moe_out
        else:
            prev_stack = torch.stack(prev_output, dim=2)
            res_query = self.res_query(prev_stack)
            scores = torch.einsum("bld,blnd->bln", x, res_query) / (x.size(-1) ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            res_out = torch.einsum("bln,blnd->bld", attn_weights, prev_stack)
            output = moe_out + res_out

        return output

    def get_aux_loss(self) -> float:
        if hasattr(self.moe, "aux_loss") and self.moe.aux_loss is not None:
            return self.moe.aux_loss
        return 0.0


class AttnResEncoderLayer(nn.Module):
    """
    AttnRes 编码器层
    使用双向自注意力，每层可以访问之前所有层的输出
    """

    def __init__(
        self,
        d_model: int,
        d_fnn: int,
        nhead: int,
        dropout: float = 0.1,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_experts = num_experts
        self.top_k = top_k

        # 双向自注意力（编码器使用双向注意力）
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # MoE 前馈网络
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(
            d_model=d_model,
            d_ff=d_fnn,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            use_aux_loss=True,
            aux_loss_coef=aux_loss_coef,
        )

        # AttnRes 残差注意力查询投影
        self.res_query = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        prev_outputs: List[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            prev_outputs: 之前所有层的输出列表，每个元素为 [batch, seq_len, d_model]
            key_padding_mask: 填充掩码 [batch, seq_len]
        Returns:
            output: 当前层输出 [batch, seq_len, d_model]
        """
        # 1. 双向自注意力 + 残差连接
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
        )
        x = x + nn.functional.dropout(attn_out, self.dropout, training=self.training)

        # 2. MoE 前馈网络
        moe_input = self.norm2(x)
        moe_out = self.moe(moe_input)

        # 3. AttnRes: 动态聚合之前所有层的输出
        if len(prev_outputs) == 0:
            # 第一层，没有之前的输出
            output = moe_out
        else:
            # 将之前所有层的输出堆叠: [batch, seq_len, num_prev_layers, d_model]
            prev_stack = torch.stack(prev_outputs, dim=2)

            # 生成查询: [batch, seq_len, num_prev_layers, d_model]
            res_query = self.res_query(prev_stack)

            # 计算注意力分数: [batch, seq_len, num_prev_layers]
            scores = torch.einsum("bld,blnd->bln", x, res_query) / (x.size(-1) ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)

            # 加权聚合之前的输出: [batch, seq_len, d_model]
            res_out = torch.einsum("bln,blnd->bld", attn_weights, prev_stack)

            # 残差连接
            output = moe_out + res_out

        return output

    def get_aux_loss(self) -> float:
        if hasattr(self.moe, "aux_loss") and self.moe.aux_loss is not None:
            return self.moe.aux_loss
        return 0.0


class AttnResEncoder(nn.Module):
    """
    AttnRes 编码器
    6层AttnRes结构，每层都是AttnResEncoderLayer
    每层可以访问之前所有层的输出
    """

    def __init__(
        self,
        d_model: int,
        d_fnn: int,
        nhead: int,
        dropout: float = 0.1,
        n_layers: int = 6,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.n_layers = n_layers

        # 6层AttnRes编码器层
        self.layers = nn.ModuleList(
            [
                AttnResEncoderLayer(
                    d_model, d_fnn, nhead, dropout, num_experts, top_k, aux_loss_coef
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入嵌入 [batch, seq_len, d_model]
            key_padding_mask: 填充掩码 [batch, seq_len]
        Returns:
            output: 编码器输出 [batch, seq_len, d_model]
        """
        all_outputs: List[torch.Tensor] = []

        for layer in self.layers:
            x = layer(x, all_outputs, key_padding_mask=key_padding_mask)
            all_outputs.append(x)

        return x

    def get_aux_loss(self) -> float:
        aux_loss = 0.0
        moe_layer_count = 0

        for layer in self.layers:
            if hasattr(layer, "get_aux_loss"):
                layer_loss = layer.get_aux_loss()
                if layer_loss > 0:
                    aux_loss += layer_loss
                    moe_layer_count += 1

        return aux_loss / max(moe_layer_count, 1)


class AttnResDecoder(nn.Module):
    """
    AttnRes 解码器
    使用因果自注意力（带掩码）
    """

    def __init__(
        self,
        d_model: int,
        d_fnn: int,
        nhead: int,
        dropout: float = 0.1,
        n_layers: int = 6,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        self.attnres = nn.ModuleList(
            [
                AttnRes(
                    d_model, d_fnn, nhead, dropout, num_experts, top_k, aux_loss_coef
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        all_outputs: List[torch.Tensor] = []
        for layer in self.attnres:
            x = layer(
                x, all_outputs, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )
            all_outputs.append(x)
        return x

    def get_aux_loss(self) -> float:
        aux_loss = 0.0
        moe_layer_count = 0

        for layer in self.attnres:
            if hasattr(layer, "get_aux_loss"):
                layer_loss = layer.get_aux_loss()
                if layer_loss > 0:
                    aux_loss += layer_loss
                    moe_layer_count += 1

        return aux_loss / max(moe_layer_count, 1)


class LearnablePosition(nn.Module):
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
    """
    SudenMind 模型
    使用自定义AttnResEncoder（6层）和AttnResDecoder（6层）
    适配ChatGLM tokenizer格式
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_fnn: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        n_layers: int = 6,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        max_position_embeddings: int = 5000,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.position_encoding = LearnablePosition(d_model, max_position_embeddings)

        # 6层AttnRes编码器（双向注意力）
        self.encoder = AttnResEncoder(
            d_model=d_model,
            d_fnn=d_fnn,
            nhead=nhead,
            dropout=dropout,
            n_layers=n_layers // 3,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )

        # 6层AttnRes解码器（因果注意力）
        self.decoder = AttnResDecoder(
            d_model=d_model,
            d_fnn=d_fnn,
            nhead=nhead,
            dropout=dropout,
            n_layers=n_layers,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )

        # 输出投影层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_fnn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fnn, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入token IDs [batch, seq_len]
            key_padding_mask: 填充掩码 [batch, seq_len]，True表示有效token，False表示padding
        Returns:
            output: 输出logits [batch, seq_len, vocab_size]
        """
        device = x.device
        batch_size, seq_len = x.size()

        # Token Embedding + 位置编码
        embedded = self.token_embedding(x)
        embedded = self.position_encoding(embedded)

        # 编码器（双向注意力）
        encoder_out = self.encoder(
            embedded,
            key_padding_mask=(key_padding_mask == False)
            if key_padding_mask is not None
            else None,
        )

        # 解码器（因果注意力）
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        decoder_out = self.decoder(
            encoder_out,
            attn_mask=causal_mask,
            key_padding_mask=(key_padding_mask == False)
            if key_padding_mask is not None
            else None,
        )

        # 输出投影
        return self.fc(decoder_out)

    def get_aux_loss(self) -> float:
        """获取编码器和解码器的辅助损失之和"""
        encoder_aux_loss = (
            self.encoder.get_aux_loss()
            if hasattr(self.encoder, "get_aux_loss")
            else 0.0
        )
        decoder_aux_loss = (
            self.decoder.get_aux_loss()
            if hasattr(self.decoder, "get_aux_loss")
            else 0.0
        )
        return encoder_aux_loss + decoder_aux_loss

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成因果掩码（上三角为True）"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def generate(
        self,
        input_seq: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        device: torch.device = torch.device("cpu"),
        use_eos_stop: bool = False,
    ) -> torch.Tensor:
        self.eval()
        input_seq = input_seq.to(device)
        generated = input_seq
        attention_mask = torch.ones_like(generated, dtype=torch.bool)
        eos_token_id = 64790

        with torch.no_grad():
            for _ in range(max_length):
                output = self.forward(generated, key_padding_mask=attention_mask)
                logits = output[:, -1, :] / temperature
                top_probs, top_indices = torch.topk(logits, k=5, dim=-1)
                probs = torch.softmax(top_probs, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_indices, dim=-1, index=next_token)

                generated = torch.cat([generated, next_token], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token, dtype=torch.bool)],
                    dim=1,
                )

                if use_eos_stop and next_token.item() == eos_token_id:
                    break

        return generated
