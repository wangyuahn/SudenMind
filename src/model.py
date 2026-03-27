import torch
from torch import nn
from typing import List, cast, Optional
from moe import MoELayer
from transformers import BertModel


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


class AttnResDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_fnn: int,
        nhead: int,
        dropout: float = 0.1,
        n_layers: int = 6,
        batch_fire: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.attn_mask: torch.Tensor
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
        self.batch_fire = batch_fire
        self.n_layers = n_layers

        if batch_fire:
            self.register_buffer(
                "attn_mask", torch.tril(torch.ones(n_layers, n_layers), diagonal=-1)
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.batch_fire:
            return self._forward_batch_fire(
                x, attn_mask, key_padding_mask=key_padding_mask
            )
        else:
            return self._forward_seq(x, attn_mask, key_padding_mask=key_padding_mask)

    def _forward_seq(
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

    def _forward_batch_fire(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.size()
        N = self.n_layers

        attn_outputs_list = []
        fnn_outputs_list = []

        current_x = x
        for module in self.attnres:
            layer = cast(AttnRes, module)

            normed = layer.norm(current_x)
            attn_out, _ = layer.attn(
                normed,
                normed,
                normed,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            current_x = current_x + nn.functional.dropout(
                attn_out, layer.dropout, training=self.training
            )
            attn_outputs_list.append(current_x)

            fnn_out = layer.moe(layer.fnn_norm(current_x))
            fnn_outputs_list.append(fnn_out)
            current_x = fnn_out

        attn_stack = torch.stack(attn_outputs_list, dim=2)
        fnn_stack = torch.stack(fnn_outputs_list, dim=2)

        all_queries = []
        for i, module in enumerate(self.attnres):
            layer = cast(AttnRes, module)
            if i == 0:
                query = torch.zeros(B, L, 0, D, device=x.device)
            else:
                query = layer.res_query(attn_stack[:, :, :i, :])
            all_queries.append(query)

        outputs = []
        for i in range(N):
            if i == 0:
                output = fnn_outputs_list[i]
            else:
                layer = cast(AttnRes, self.attnres[i])
                prev_outputs = torch.stack(outputs, dim=2)
                res_query = layer.res_query(prev_outputs)
                scores = torch.einsum(
                    "bld,blnd->bln", attn_outputs_list[i], res_query
                ) / (D**0.5)
                attn_weights = torch.softmax(scores, dim=-1)
                res_out = torch.einsum("bln,blnd->bld", attn_weights, prev_outputs)
                output = fnn_outputs_list[i] + res_out

            outputs.append(output)

        return outputs[-1]

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
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        bert_model_name="bert-base-chinese",
        freeze_bert=True,
        not_freeze_bert_num_layers=3,
        num_experts=4,
        top_k=2,
        aux_loss_coef=0.01,
    ):
        super().__init__()

        self.bert_model_name = bert_model_name
        self.freeze_bert = freeze_bert
        self.bert_hidden_dim = 768

        self.bert = BertModel.from_pretrained(bert_model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print(f"BERT参数已冻结 ({bert_model_name})")

            num_layers = len(self.bert.encoder.layer)
            start_layer = num_layers - not_freeze_bert_num_layers
            for i in range(start_layer, num_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
                print(f"BERT参数已解冻第 {i} 层")

        self.bert_adapter = nn.Linear(self.bert_hidden_dim, embedding_dim)

        self.decoder = AttnResDecoder(
            d_model=embedding_dim,
            d_fnn=hidden_dim,
            nhead=8,
            dropout=0.1,
            n_layers=6,
            batch_fire=False,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and p.requires_grad:
                if not name.startswith("bert."):
                    nn.init.xavier_uniform_(p)
                    # print(f"初始化: {name} ({p.shape})")

    def forward(self, x, key_padding_mask=None):
        device = x.device
        batch_size, seq_len = x.size()

        bert_input_ids = x

        bert_outputs = self.bert(
            bert_input_ids,
            attention_mask=key_padding_mask.float()
            if key_padding_mask is not None
            else None,
        )
        bert_embeddings = bert_outputs.last_hidden_state

        embedded = self.bert_adapter(bert_embeddings)

        causal_mask = self.generate_square_subsequent_mask(seq_len).to(device)

        out = self.decoder(
            embedded,
            attn_mask=causal_mask,
            key_padding_mask=(key_padding_mask == False)
            if key_padding_mask is not None
            else None,
        )

        return self.fc(out)

    def get_aux_loss(self) -> float:
        if hasattr(self.decoder, "get_aux_loss"):
            return self.decoder.get_aux_loss()
        return 0.0

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def generate(
        self, input_seq, max_length=100, temperature=1.0, device=torch.device("cpu")
    ):
        self.eval()
        input_seq = input_seq.to(device)
        generated = input_seq

        # 获取 [SEP] token ID (BERT 的 sep_token_id 是 102)
        sep_token_id = 102  # BERT 中文模型的 [SEP] token ID

        with torch.no_grad():
            for _ in range(max_length):
                output = self.forward(generated)
                logits = output[:, -1, :] / temperature
                top_probs, top_indices = torch.topk(logits, k=5, dim=-1)
                probs = torch.softmax(top_probs, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_indices, dim=-1, index=next_token)

                # 使用正确的 [SEP] token ID 作为停止条件
                if next_token.item() == sep_token_id:
                    break

                generated = torch.cat([generated, next_token], dim=1)

        return generated
