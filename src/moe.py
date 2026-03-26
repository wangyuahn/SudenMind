import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: 输入张量 (任意形状，最后一维为 d_model)

        返回：
            output: 专家输出 (与输入形状相同)
        """
        return self.net(x)


class Router(nn.Module):
    """
    路由器网络

    负责将每个输入 token 路由到最合适的专家。
    路由器学习为每个 token 分配专家权重，实现稀疏激活。

    参数：
        d_model: 输入维度
        num_experts: 专家数量
        top_k: 每个 token 激活的 top-k 专家 (控制稀疏性)
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由器是一个线性层，将输入映射到专家 logits
        # 不使用偏置项，让路由器专注于学习专家选择模式
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)

        返回：
            router_probs: 路由概率 (batch_size, seq_len, num_experts)
            router_logits: 路由 logits (batch_size, seq_len, num_experts)
        """
        # 计算每个专家的得分 (logits)
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]

        # 使用 softmax 将 logits 转换为概率分布
        # 使用 float32 确保数值稳定性
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        return router_probs, router_logits


class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) 层

    完整的 MoE 实现，包含：
    1. 多个专家网络
    2. 路由器网络
    3. 稀疏激活机制 (top-k 选择)
    4. 负载均衡辅助损失

    参数：
        d_model: 输入/输出维度
        d_ff: 专家网络的中间层维度
        num_experts: 专家数量 (默认 8)
        top_k: 每个 token 激活的 top-k 专家 (默认 2)
        dropout: Dropout 概率
        use_aux_loss: 是否使用辅助损失 (负载均衡)
        aux_loss_coef: 辅助损失系数 (默认 0.01)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_aux_loss: bool = True,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        self.aux_loss_coef = aux_loss_coef

        # 创建专家列表 (每个专家是一个独立的 MLP)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

        # 路由器网络
        self.router = Router(d_model, num_experts, top_k)

        # 记录辅助损失 (在训练过程中计算)
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)

        返回：
            output: 输出张量 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # ========== 1. 路由：决定将每个 token 发送给哪些专家 ==========
        router_probs, router_logits = self.router(x)
        # router_probs: [batch_size, seq_len, num_experts]
        # router_logits: [batch_size, seq_len, num_experts]

        # ========== 2. 选择 top-k 专家 (稀疏激活) ==========
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # top_k_probs: [batch_size, seq_len, top_k] - 选中的专家概率
        # top_k_indices: [batch_size, seq_len, top_k] - 选中的专家索引

        # ========== 3. 重新归一化 top-k 概率 ==========
        # 确保选中的专家概率之和为 1
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # ========== 4. 准备输出张量 ==========
        output = torch.zeros_like(x)  # 初始化为零

        # ========== 5. 计算辅助损失 (负载均衡) ==========
        if self.use_aux_loss and self.training:
            # 计算每个专家被选中的次数 (one-hot 编码)
            expert_mask = torch.nn.functional.one_hot(
                top_k_indices, num_classes=self.num_experts
            ).float()
            # expert_mask: [batch_size, seq_len, top_k, num_experts]

            # 在 top_k 维度求和，得到每个 token 选择的专家
            expert_mask = expert_mask.sum(dim=2)
            # expert_mask: [batch_size, seq_len, num_experts]

            # 计算每个专家被分配到的 token 比例 (批次平均)
            expert_proportion = expert_mask.mean(dim=1)
            # expert_proportion: [batch_size, num_experts]

            # 计算路由概率的平均值
            router_prob_mean = router_probs.mean(dim=1)
            # router_prob_mean: [batch_size, num_experts]

            # 负载均衡损失: 鼓励专家负载均匀分布
            # 公式: aux_loss = mean(expert_proportion * router_prob_mean) * num_experts
            self.aux_loss = (
                expert_proportion * router_prob_mean
            ).mean() * self.num_experts
            self.aux_loss = self.aux_loss * self.aux_loss_coef  # 应用系数
        else:
            self.aux_loss = None  # 推理时不计算辅助损失

        # ========== 6. 处理每个专家的计算 ==========
        # 更高效的实现：批量处理每个专家的计算
        for expert_idx, expert in enumerate(self.experts):
            # 找到使用当前专家的所有 token
            # mask: [batch_size, seq_len, top_k] 表示每个位置是否选择了该专家
            expert_mask = top_k_indices == expert_idx

            if expert_mask.any():
                # 获取选择该专家的所有 token 位置
                batch_indices, seq_indices, k_indices = torch.where(expert_mask)

                if len(batch_indices) > 0:
                    # 收集这些 token 的输入
                    token_indices = batch_indices * seq_len + seq_indices
                    expert_inputs = x.view(-1, d_model)[
                        token_indices
                    ]  # [num_tokens, d_model]

                    # 收集对应的权重
                    expert_weights = top_k_probs[
                        batch_indices, seq_indices, k_indices
                    ]  # [num_tokens]

                    # 批量计算专家输出
                    expert_outputs = expert(expert_inputs)  # [num_tokens, d_model]

                    # 加权输出
                    weighted_outputs = expert_outputs * expert_weights.unsqueeze(
                        1
                    )  # [num_tokens, d_model]

                    # 将结果累加到输出中
                    # 使用 index_add_ 进行高效累加
                    output_flat = output.view(-1, d_model)
                    output_flat.index_add_(0, token_indices, weighted_outputs)

        return output


class MoETransformerLayer(nn.Module):
    """
    支持 MoE 的 Transformer 层 (可选)

    这是一个可选的 Transformer 层实现，支持在 FFN 部分使用 MoE。
    注意：在 SudenMind 中，我们强制使用 MoE，这个类仅供参考。

    参数：
        d_model: 模型隐藏维度
        d_ff: FFN/MoE 中间层维度
        nhead: 注意力头数
        num_experts: MoE 专家数量
        top_k: 每个 token 激活的专家数
        dropout: Dropout 概率
        use_moe: 是否使用 MoE (开关)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        nhead: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_moe: bool = False,  # 开关：是否使用 MoE
    ):
        super().__init__()
        self.use_moe = use_moe

        # 注意力部分
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # FFN 部分 (普通 FFN 或 MoE)
        self.norm2 = nn.LayerNorm(d_model)
        if use_moe:
            # 使用 MoE 层
            self.ffn = MoELayer(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                use_aux_loss=True,
            )
        else:
            # 使用标准 FFN
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None):
        """
        前向传播

        参数：
            x: 输入张量
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码

        返回：
            x: 输出张量
        """
        # 1. 自注意力 + 残差连接
        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + F.dropout(attn_out, self.dropout, training=self.training)

        # 2. FFN/MoE + 残差连接
        ffn_out = self.ffn(self.norm2(x))
        x = x + F.dropout(ffn_out, self.dropout, training=self.training)

        return x

    def get_aux_loss(self):
        """
        获取 MoE 辅助损失

        如果使用 MoE 且 MoE 层有辅助损失，则返回该损失。

        返回：
            aux_loss: MoE 辅助损失值，如果不使用 MoE 则返回 0.0
        """
        if (
            self.use_moe
            and isinstance(self.ffn, MoELayer)
            and self.ffn.aux_loss is not None
        ):
            return self.ffn.aux_loss
        return 0.0
