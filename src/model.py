import torch
from torch import nn
from typing import List, Optional
from moe import MoELayer


class AttnRes(nn.Module):
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

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        prev_output: List[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_kv=None,
        use_cache=False,
    ):
        norm = self.norm(x)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, norm], dim=1)
            v = torch.cat([past_v, norm], dim=1)
        else:
            k = v = norm

        attn_out, _ = self.attn(
            norm,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        new_kv = (k, v) if use_cache else None

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

            gate = self.gate(torch.cat([moe_out, res_out], dim=-1))
            output = (1 - gate) * moe_out + gate * res_out

        return output, new_kv


    def get_aux_loss(self):
        return self.moe.aux_loss if hasattr(self.moe, "aux_loss") else 0.0


class AttnResEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_fnn,
        nhead,
        dropout=0.1,
        n_layers=6,
        num_experts=4,
        top_k=2,
        aux_loss_coef=0.01,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            AttnRes(d_model, d_fnn, nhead, dropout,
                    num_experts, top_k, aux_loss_coef)
            for _ in range(n_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        all_outputs = []

        for layer in self.layers:
            x, _ = layer(x, all_outputs, key_padding_mask=key_padding_mask)
            all_outputs.append(x)

        return x

    def get_aux_loss(self):
        return sum(l.get_aux_loss() for l in self.layers)


class AttnResDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_fnn,
        nhead,
        dropout=0.1,
        n_layers=6,
        num_experts=4,
        top_k=2,
        aux_loss_coef=0.01,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            AttnRes(d_model, d_fnn, nhead, dropout,
                    num_experts, top_k, aux_loss_coef)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None,
        past_kvs=None,
        use_cache=False,
    ):
        all_outputs = []
        new_kvs = []

        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None

            x, new_kv = layer(
                x,
                all_outputs,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                past_kv=past_kv,
                use_cache=use_cache,
            )

            all_outputs.append(x)
            if use_cache:
                new_kvs.append(new_kv)

        return x, new_kvs

    def get_aux_loss(self):
        return sum(l.get_aux_loss() for l in self.layers)


class LearnablePosition(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x, start_pos=0):
        """添加位置编码
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            start_pos: 序列起始位置（推理时需传入累计长度）
        """
        seq_len = x.size(1)
        pos = torch.arange(start_pos, start_pos + seq_len, device=x.device).unsqueeze(0)
        pos = torch.clamp(pos, max=self.max_len - 1)
        return x + self.pos_embedding(pos)


class SudenMind(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        d_fnn=512,
        nhead=8,
        dropout=0.1,
        n_layers=6,
        num_experts=8,
        top_k=2,
        aux_loss_coef=0.01,
        max_position_embeddings=5000,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = LearnablePosition(d_model, max_position_embeddings)

        self.encoder = AttnResEncoder(
            d_model, d_fnn, nhead, dropout,
            n_layers=n_layers // 2,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )

        self.decoder = AttnResDecoder(
            d_model, d_fnn, nhead, dropout,
            n_layers=n_layers,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_fnn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fnn, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, key_padding_mask=None, past_kvs=None, use_cache=False, start_pos=0):
        x = self.token_embedding(x)
        x = self.position_encoding(x, start_pos=start_pos)

        x = self.encoder(x, key_padding_mask)

        x, new_kvs = self.decoder(
            x,
            key_padding_mask=key_padding_mask,
            past_kvs=past_kvs,
            use_cache=use_cache,
        )

        logits = self.fc(x)

        return (logits, new_kvs) if use_cache else logits

    def get_aux_loss(self):
        return self.encoder.get_aux_loss() + self.decoder.get_aux_loss()

    def generate(
        self,
        input_seq,
        max_length=100,
        temperature=1.0,
        device=torch.device("cpu"),
        use_eos_stop=False,
        eos_token_id=None,
    ):
        self.eval()
        input_seq = input_seq.to(device)

        generated = input_seq
        past_kvs = None
        current_seq_len = input_seq.size(1)  # 追踪已生成的序列长度

        if eos_token_id is None:
            eos_token_id = 150005

        with torch.no_grad():
            # 首次推理：输入完整的prompt
            logits, past_kvs = self.forward(
                generated, 
                use_cache=True,
                start_pos=0
            )

            for step in range(max_length):
                logits_step = logits[:, -1, :] / temperature
                probs = torch.softmax(logits_step, dim=-1)
                next_token = torch.multinomial(probs, 1)

                generated = torch.cat([generated, next_token], dim=1)
                current_seq_len += 1

                # 后续推理：只输入新token，但告诉模型当前位置
                logits, past_kvs = self.forward(
                    next_token,
                    past_kvs=past_kvs,
                    use_cache=True,
                    start_pos=current_seq_len - 1  # 新token的位置
                )

                if use_eos_stop and (next_token == eos_token_id).all():
                    break

        return generated