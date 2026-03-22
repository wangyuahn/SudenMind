"""
SudenMind 模型架构
基于 Attention Residuals (AttnRes) 的 Decoder-Only 架构，集成了 Mixture of Experts (MoE)

核心特性：
1. AttnRes 架构：每层可以动态关注之前所有层的输出
2. MoE 集成：所有 FFN 层被强制替换为 MoE 层
3. 可学习位置编码：比固定正弦编码更灵活
4. 批量优先格式：符合 PyTorch 标准

作者：SudenMind 团队
版本：2.0 (集成 MoE)
"""

import torch
from torch import nn
from typing import List, cast, Optional
from moe import MoELayer


class AttnRes(nn.Module):
    """
    单层 Attention with Residual (AttnRes) 模块

    这是 SudenMind 的核心构建块，包含：
    1. 多头自注意力机制 (带因果掩码)
    2. MoE 前馈网络 (强制替换标准 FFN)
    3. 跨层残差注意力 (AttnRes 核心创新)

    参数：
        d_model: 模型隐藏维度
        d_fnn: MoE 专家网络的隐藏维度
        nhead: 注意力头数
        dropout: Dropout 概率
        num_experts: MoE 专家数量 (默认 4)
        top_k: 每个 token 激活的专家数 (默认 2)
        aux_loss_coef: MoE 辅助损失系数 (默认 0.01)
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

        # 多头自注意力层 (batch_first=True 符合 PyTorch 标准)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)  # 注意力前的层归一化

        # MoE 前馈网络 (强制替换标准 FFN)
        self.fnn_norm = nn.LayerNorm(d_model)  # MoE 前的层归一化
        self.moe = MoELayer(
            d_model=d_model,
            d_ff=d_fnn,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            use_aux_loss=True,
            aux_loss_coef=aux_loss_coef,
        )

        # 跨层残差注意力的查询变换
        self.res_query = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        prev_output: List[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)
            prev_output: 之前所有层的输出列表
            attn_mask: 注意力掩码 (用于因果注意力)

        返回：
            output: 当前层的输出张量
        """
        # 1. 多头自注意力
        norm = self.norm(x)  # 层归一化
        attn_out, _ = self.attn(norm, norm, norm, attn_mask=attn_mask)  # 自注意力
        x = x + nn.functional.dropout(
            attn_out, self.dropout, training=self.training
        )  # 残差连接

        # 2. MoE 前馈网络 (强制替换标准 FFN)
        moe_input = self.fnn_norm(x)  # MoE 前的层归一化
        moe_out = self.moe(moe_input)  # MoE 层前向传播

        # 3. 跨层残差注意力 (AttnRes 核心)
        if len(prev_output) == 0:
            # 第一层没有之前的层可参考
            output = moe_out
        else:
            # 将之前层的输出堆叠成张量 (batch_size, seq_len, num_prev_layers, d_model)
            prev_stack = torch.stack(prev_output, dim=2)

            # 计算查询向量
            res_query = self.res_query(prev_stack)

            # 计算注意力分数：当前层输出与之前层查询的点积
            scores = torch.einsum("bld,blnd->bln", x, res_query) / (x.size(-1) ** 0.5)

            # Softmax 归一化得到注意力权重
            attn_weights = torch.softmax(scores, dim=-1)

            # 加权求和得到残差输出
            res_out = torch.einsum("bln,blnd->bld", attn_weights, prev_stack)

            # 最终输出 = MoE 输出 + 残差输出
            output = moe_out + res_out

        return output

    def get_aux_loss(self) -> float:
        """
        获取 MoE 辅助损失 (负载均衡损失)

        辅助损失用于平衡专家之间的负载，防止某些专家被过度使用而其他专家被忽略。
        这是 MoE 训练中的重要组成部分。

        返回：
            aux_loss: MoE 辅助损失值，如果不存在则返回 0.0
        """
        if hasattr(self.moe, "aux_loss") and self.moe.aux_loss is not None:
            return self.moe.aux_loss
        return 0.0


class AttnResDecoder(nn.Module):
    """
    多层 AttnRes 解码器

    将多个 AttnRes 层堆叠在一起，支持两种计算模式：
    1. 顺序模式 (_forward_seq): 标准递归计算
    2. 批量优先模式 (_forward_batch_fire): 尝试并行计算 (实验性)

    参数：
        d_model: 模型隐藏维度
        d_fnn: MoE 专家网络的隐藏维度
        nhead: 注意力头数
        dropout: Dropout 概率
        n_layers: AttnRes 层数
        batch_fire: 是否启用批量优先模式 (实验性)
        num_experts: MoE 专家数量
        top_k: 每个 token 激活的专家数
        aux_loss_coef: MoE 辅助损失系数
    """

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
        self.attn_mask: torch.Tensor  # 类型注解
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        # 创建 n_layers 个 AttnRes 层
        self.attnres = nn.ModuleList(
            [
                AttnRes(
                    d_model, d_fnn, nhead, dropout, num_experts, top_k, aux_loss_coef
                )
                for _ in range(n_layers)
            ]
        )
        self.batch_fire = batch_fire  # 批量优先模式标志
        self.n_layers = n_layers

        # 如果启用批量优先模式，注册层间注意力掩码
        if batch_fire:
            self.register_buffer(
                "attn_mask", torch.tril(torch.ones(n_layers, n_layers), diagonal=-1)
            )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 (根据模式选择计算方法)

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)
            attn_mask: 注意力掩码 (用于因果注意力)

        返回：
            output: 解码器输出张量
        """
        if self.batch_fire:
            return self._forward_batch_fire(x, attn_mask)  # 批量优先模式
        else:
            return self._forward_seq(x, attn_mask)  # 顺序模式

    def _forward_seq(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        顺序前向传播 (标准递归计算)

        这是默认的计算模式，每层的输入依赖于前一层的输出。
        计算复杂度: O(n_layers)

        参数：
            x: 输入张量
            attn_mask: 注意力掩码

        返回：
            x: 最后一层的输出
        """
        all_outputs: List[torch.Tensor] = []  # 存储所有层的输出
        for layer in self.attnres:
            # 当前层接收输入 x 和之前所有层的输出
            x = layer(x, all_outputs, attn_mask=attn_mask)
            all_outputs.append(x)  # 将当前层输出添加到列表
        return x  # 返回最后一层的输出

    def _forward_batch_fire(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        批量优先前向传播 (实验性并行计算)

        尝试将部分计算并行化，但注意：由于层间依赖关系，真正的完全并行需要改变架构。
        当前实现仍然是顺序的，但将中间结果存储起来供后续批量计算使用。

        核心思想：
        1. 阶段1: 计算所有层的自注意力和 MoE 输出
        2. 阶段2: 批量计算跨层残差注意力

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)
            attn_mask: 注意力掩码

        返回：
            output: 最后一层的输出
        """
        B, L, D = x.size()  # batch_size, seq_len, d_model
        N = self.n_layers  # 层数

        # ========== 阶段1: 计算所有层的自注意力和 MoE 输出 ==========
        # 输入: x (B, L, D)
        # 输出: attn_outputs_list (N 个张量, 每个 B×L×D)
        #       fnn_outputs_list (N 个张量, 每个 B×L×D)

        attn_outputs_list = []  # 存储每层的注意力输出
        fnn_outputs_list = []  # 存储每层的 MoE 输出

        current_x = x
        # 注意：这里仍然是顺序计算，因为每层的输入依赖于前一层的输出
        # 真正的并行需要改变模型架构（如使用更复杂的并行策略）
        for module in self.attnres:
            layer = cast(AttnRes, module)

            # 自注意力计算
            normed = layer.norm(current_x)
            attn_out, _ = layer.attn(normed, normed, normed, attn_mask=attn_mask)
            current_x = current_x + nn.functional.dropout(
                attn_out, layer.dropout, training=self.training
            )
            attn_outputs_list.append(current_x)  # 保存注意力输出

            # MoE 前馈网络计算
            fnn_out = layer.moe(layer.fnn_norm(current_x))  # 使用 MoE 替换 FFN
            fnn_outputs_list.append(fnn_out)  # 保存 MoE 输出
            current_x = fnn_out

        # 将列表转换为堆叠张量
        attn_stack = torch.stack(attn_outputs_list, dim=2)  # (B, L, N, D)
        fnn_stack = torch.stack(fnn_outputs_list, dim=2)  # (B, L, N, D)

        # ========== 阶段2: 批量计算跨层残差注意力 ==========
        # 为每层计算对所有之前层的查询向量
        all_queries = []
        for i, module in enumerate(self.attnres):
            layer = cast(AttnRes, module)
            if i == 0:
                # 第一层没有之前的层可查询
                query = torch.zeros(B, L, 0, D, device=x.device)
            else:
                # 查询所有之前层的注意力输出
                query = layer.res_query(attn_stack[:, :, :i, :])  # (B, L, i, D)
            all_queries.append(query)

        # 迭代计算每层的最终输出（由于递归依赖，无法完全并行）
        outputs = []
        for i in range(N):
            if i == 0:
                # 第一层：只有 MoE 输出，没有残差
                output = fnn_outputs_list[i]
            else:
                layer = cast(AttnRes, self.attnres[i])

                # 获取之前所有层的输出
                prev_outputs = torch.stack(outputs, dim=2)  # (B, L, i, D)

                # 计算跨层注意力
                res_query = layer.res_query(prev_outputs)  # (B, L, i, D)
                scores = torch.einsum(
                    "bld,blnd->bln", attn_outputs_list[i], res_query
                ) / (D**0.5)  # 缩放点积注意力
                attn_weights = torch.softmax(scores, dim=-1)  # (B, L, i)

                # 计算残差输出
                res_out = torch.einsum("bln,blnd->bld", attn_weights, prev_outputs)

                # 最终输出 = MoE 输出 + 残差输出
                output = fnn_outputs_list[i] + res_out

            outputs.append(output)

        return outputs[-1]  # 返回最后一层的输出

    def get_aux_loss(self) -> float:
        """
        收集所有 MoE 层的辅助损失并计算平均值

        遍历所有 AttnRes 层，收集每个 MoE 层的辅助损失，
        然后计算平均损失。这有助于平衡所有专家的负载。

        返回：
            avg_aux_loss: 平均 MoE 辅助损失
        """
        aux_loss = 0.0
        moe_layer_count = 0

        for layer in self.attnres:
            if hasattr(layer, "get_aux_loss"):
                layer_loss = layer.get_aux_loss()
                if layer_loss > 0:
                    aux_loss += layer_loss
                    moe_layer_count += 1

        # 计算平均损失，避免除零错误
        return aux_loss / max(moe_layer_count, 1)


class LearnablePosition(nn.Module):
    """
    可学习的位置编码

    与 Transformer 的固定正弦位置编码不同，这种位置编码是可学习的，
    通过训练来适应特定的任务和数据分布。

    参数：
        d_model: 模型隐藏维度
        max_len: 最大序列长度 (默认 5000)
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)  # 可学习的位置嵌入
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码到输入

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)

        返回：
            x + pos_emb: 添加位置编码后的张量
        """
        seq_len = x.size(1)  # 获取序列长度

        # 创建位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # 获取位置嵌入并添加到输入
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, d_model)
        return x + pos_emb  # 广播到整个批次


class SudenMind(nn.Module):
    """
    SudenMind 主模型类

    完整的对话生成模型，包含：
    1. 词嵌入层
    2. 可学习位置编码
    3. AttnResDecoder (多层 AttnRes + MoE)
    4. 输出投影层

    参数：
        vocab_size: 词表大小
        embedding_dim: 词嵌入维度
        hidden_dim: 隐藏层维度 (MoE 专家网络维度)
        output_dim: 输出维度 (通常等于 vocab_size)
        num_experts: MoE 专家数量 (默认 4)
        top_k: 每个 token 激活的专家数 (默认 2)
        aux_loss_coef: MoE 辅助损失系数 (默认 0.01)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_experts=4,
        top_k=2,
        aux_loss_coef=0.01,
    ):
        super().__init__()

        # 1. 词嵌入层 (使用 padding_idx=0 处理填充 token)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. 可学习位置编码
        self.position = LearnablePosition(embedding_dim)

        # 3. AttnResDecoder (核心解码器)
        self.decoder = AttnResDecoder(
            d_model=embedding_dim,
            d_fnn=hidden_dim,
            nhead=8,  # 注意力头数固定为 8
            dropout=0.1,  # 从 0.3 降低到 0.1，数据质量高了不需要太强正则
            n_layers=6,  # 6 层 AttnRes
            batch_fire=False,  # 使用顺序模式
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )

        # 4. 输出投影层 (将隐藏状态映射到词表空间)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),  # 扩展维度
            nn.LeakyReLU(),  # 非线性激活
            nn.Linear(hidden_dim, output_dim),  # 映射到输出维度
        )

        # 5. 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        使用 Xavier 均匀分布初始化所有权重

        这种初始化方法有助于保持梯度在深度网络中的流动，
        防止梯度消失或爆炸问题。
        """
        for p in self.parameters():
            if p.dim() > 1:  # 只初始化权重矩阵，不初始化偏置
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入 token IDs (batch_size, seq_len)

        返回：
            logits: 输出 logits (batch_size, seq_len, output_dim)
        """
        device = x.device
        batch_size, seq_len = x.size()

        # 1. 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # 2. 添加位置编码
        embedded = self.position(embedded)

        # 3. 生成因果注意力掩码 (防止看到未来信息)
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(device)

        # 4. 通过 AttnResDecoder
        out = self.decoder(
            embedded, attn_mask=causal_mask
        )  # (batch_size, seq_len, embedding_dim)

        # 5. 投影到词表空间
        return self.fc(out)  # (batch_size, seq_len, output_dim)

    def get_aux_loss(self) -> float:
        """
        获取 MoE 辅助损失

        从解码器收集所有 MoE 层的辅助损失。
        这个损失用于训练时的负载均衡。

        返回：
            aux_loss: MoE 辅助损失值
        """
        if hasattr(self.decoder, "get_aux_loss"):
            return self.decoder.get_aux_loss()
        return 0.0

    def generate_square_subsequent_mask(self, sz):
        """
        生成因果注意力掩码 (上三角矩阵)

        用于防止解码器在生成时看到未来的 token。
        例如，对于序列长度 4:
        [[False,  True,  True,  True],
         [False, False,  True,  True],
         [False, False, False,  True],
         [False, False, False, False]]

        参数：
            sz: 序列长度

        返回：
            mask: 布尔掩码张量 (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def generate(
        self, input_seq, max_length=100, temperature=1.0, device=torch.device("cpu")
    ):
        """
        自回归文本生成

        使用 top-k 采样策略生成文本，支持温度控制。

        参数：
            input_seq: 输入序列 (batch_size, seq_len)
            max_length: 最大生成长度
            temperature: 采样温度 (越高越随机，越低越确定)
            device: 计算设备

        返回：
            generated: 生成的完整序列 (包含输入)
        """
        self.eval()  # 切换到评估模式
        input_seq = input_seq.to(device)
        generated = input_seq  # 初始化为输入序列

        with torch.no_grad():  # 禁用梯度计算
            for _ in range(max_length):
                # 前向传播获取当前序列的 logits
                output = self.forward(
                    generated
                )  # (batch_size, current_len, vocab_size)

                # 获取最后一个 token 的 logits 并应用温度
                logits = output[:, -1, :] / temperature

                # Top-k 采样 (k=5)
                top_probs, top_indices = torch.topk(logits, k=5, dim=-1)
                probs = torch.softmax(top_probs, dim=-1)  # 归一化概率

                # 从 top-k 中采样下一个 token
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_indices, dim=-1, index=next_token)

                # 检查是否生成结束 token (EOS token id=3)
                if next_token.item() == 3:  # EOS token
                    break

                # 将新 token 添加到序列中
                generated = torch.cat([generated, next_token], dim=1)

        return generated
