"""
conversation_dataset.py

ShareGPT/ChatML 格式的对话数据集，适用于 ChatGLM 系列模型的微调训练。

格式:
    [BOS]<|im_start|>user\n{问题}<|im_end|>\n<|im_start|>assistant\n{回答}<|im_end|>...[EOS]

Loss 计算规则:
    - 只对 assistant 轮次的内容 token 计算 loss
    - user / system 轮次及 [BOS] 前缀均设为 -100
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


# ─────────────────────────── 常量 ────────────────────────────

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"


# ─────────────────────────── 工具函数 ────────────────────────────


def clean_chinese_text(text: str) -> str:
    """清洗中文文本：去除汉字间多余空格及标点前后空格。"""
    if not text:
        return text
    # 去除相邻中文字符间的空格
    text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    # 去除中文标点前的空格
    text = re.sub(r'\s+([，。！？；："\'《》【】（）])', r"\1", text)
    # 去除中文标点后的空格
    text = re.sub(r'([，！？；："\'《》【】（）])\s+', r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_response(text: str) -> str:
    """从生成文本中提取最后一段 assistant 回复内容。"""
    if not text:
        return ""
    marker = f"{IM_START}{ROLE_ASSISTANT}\n"
    idx = text.rfind(marker)  # 取最后一个 assistant 块
    if idx == -1:
        return text.strip()
    content = text[idx + len(marker) :]
    return content.split(IM_END)[0].strip()


# ─────────────────────────── 数据集 ────────────────────────────


class ConversationDataset(Dataset):
    """
    加载 LCCC 数据集并编码为 ChatML 格式，供语言模型监督微调（SFT）使用。

    Args:
        split:          数据集分割，如 "train" / "validation"
        config:         LCCC 子集，"base" 或 "large"
        max_history:    保留最近的对话轮数（user+assistant 各算一轮）
        max_length:     token 序列最大长度（含 BOS/EOS）
        tokenizer_name: HuggingFace tokenizer 名称或本地路径
        test_mode:      True 时仅加载前 100 条，用于调试
        cache_dir:      本地缓存目录
        use_cache:      是否使用本地 JSON 缓存
        clean_spaces:   是否清洗中文文本空格
    """

    def __init__(
        self,
        split: str = "train",
        config: str = "base",
        max_history: int = 5,
        max_length: int = 512,
        tokenizer_name: str = "THUDM/chatglm-6b",
        test_mode: bool = False,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        clean_spaces: bool = True,
    ):
        self.split = split
        self.config = config
        self.max_history = max_history
        self.max_length = max_length
        self.test_mode = test_mode
        self.clean_spaces = clean_spaces
        self.cache_dir = cache_dir or "data/cache"
        self.use_cache = use_cache

        self.tokenizer = self._load_tokenizer(tokenizer_name)
        self._setup_special_tokens()
        self._load_data()

    # ── 初始化 ──────────────────────────────────────────────

    @staticmethod
    def _load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
        print(f"加载 tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        # 确保 ChatML 特殊 token 已注册为独立 token
        new_tokens = [
            t
            for t in (IM_START, IM_END)
            if t not in tokenizer.additional_special_tokens
        ]
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        return tokenizer

    def _setup_special_tokens(self):
        """缓存常用 token ID，避免重复查询。"""
        tok = self.tokenizer
        self.bos_id = tok.bos_token_id
        self.eos_id = tok.eos_token_id
        self.pad_id = tok.pad_token_id
        self.im_start_id = tok.convert_tokens_to_ids(IM_START)
        self.im_end_id = tok.convert_tokens_to_ids(IM_END)

        nl = tok.encode("\n", add_special_tokens=False)
        self.newline_id = nl[0] if nl else 10  # fallback: LF 的常见 ID

        print(
            f"词表大小={len(tok)}  BOS={self.bos_id}  EOS={self.eos_id}  "
            f"im_start={self.im_start_id}  im_end={self.im_end_id}"
        )

    def _load_data(self):
        """优先从本地 JSON 缓存加载；缓存不存在时下载并保存。"""
        cache_file = os.path.join(
            self.cache_dir, f"lccc_{self.config}_{self.split}.json"
        )

        if self.use_cache and os.path.exists(cache_file):
            print(f"从缓存加载: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
            print(f"已加载 {len(self.dataset)} 条对话")
            return

        split_str = f"{self.split}[:100]" if self.test_mode else self.split
        print(f"下载数据集: thu-coai/lccc  config={self.config}  split={split_str}")
        raw = load_dataset("thu-coai/lccc", self.config, split=split_str)

        self.dataset = []
        for item in raw:
            dialog = item.get("dialog", [])
            if len(dialog) < 2:
                continue
            if self.clean_spaces:
                dialog = [clean_chinese_text(u) for u in dialog]
            self.dataset.append({"dialog": dialog})

        print(f"已加载 {len(self.dataset)} 条对话")

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.dataset, f, ensure_ascii=False)
            print(f"已缓存到: {cache_file}")

    # ── Dataset 接口 ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dialog = self.dataset[idx]["dialog"]

        if len(dialog) < 2:
            return self._dummy_sample()

        # 截取最近 N 轮并保证偶数（user/assistant 成对）
        dialog = dialog[-(self.max_history * 2) :]
        if len(dialog) % 2 != 0:
            dialog = dialog[:-1]

        # 编码为完整 token 序列: [BOS] <对话> [EOS]
        dialog_tokens = self._build_dialog_tokens(dialog)
        full_tokens = [self.bos_id] + dialog_tokens + [self.eos_id]
        full_tokens = full_tokens[: self.max_length]

        # 按轮次构建 labels（只对 assistant 计算 loss）
        labels = self._build_labels(full_tokens, dialog)

        # 自回归偏移：input = full[:-1]，label = full[1:]
        return {
            "input_ids": torch.tensor(full_tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(labels[1:], dtype=torch.long),
            "attention_mask": torch.ones(len(full_tokens) - 1, dtype=torch.bool),
        }

    # ── Token 构建 ──────────────────────────────────────────

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _role_header_tokens(self, role: str) -> List[int]:
        """生成角色头部: <|im_start|>role\n"""
        return [self.im_start_id] + self._encode(role) + [self.newline_id]

    def _build_dialog_tokens(self, dialog: List[str]) -> List[int]:
        """
        将对话列表编码为 ChatML token 序列。

        格式: <|im_start|>user\n内容<|im_end|>\n<|im_start|>assistant\n内容<|im_end|>...
        """
        tokens: List[int] = []
        for i, content in enumerate(dialog):
            role = ROLE_USER if i % 2 == 0 else ROLE_ASSISTANT
            tokens += self._role_header_tokens(role)
            tokens += self._encode(content)
            tokens += [self.im_end_id]
            if i < len(dialog) - 1:
                tokens += [self.newline_id]
        return tokens

    def _build_labels(self, full_tokens: List[int], dialog: List[str]) -> List[int]:
        """
        构建 labels 列表（与 full_tokens 等长）。

        规则:
        - [BOS] 前缀      → -100
        - user 轮次全部   → -100
        - assistant 轮次内容部分  → 与 full_tokens 相同（参与 loss）
        - 角色头/im_end/newline → -100
        - [EOS]           → 参与 loss

        注意：正确处理 newline 分隔符，确保标签位置与实际 token 序列一致
        """
        labels = [-100] * len(full_tokens)
        pos = 1  # 跳过 BOS

        for i, content in enumerate(dialog):
            role = ROLE_USER if i % 2 == 0 else ROLE_ASSISTANT
            is_assistant = i % 2 == 1

            # 构建该轮的 tokens 并精确追踪各部分位置
            header_tokens = self._role_header_tokens(role)  # [im_start] + [role] + [newline]
            content_tokens = self._encode(content)
            end_token = [self.im_end_id]

            # 角色头部分长度
            header_len = len(header_tokens)
            content_len = len(content_tokens)

            # 该轮 tokens 总长度（不含分隔 newline）
            turn_len = header_len + content_len + len(end_token)

            # 计算该轮在 full_tokens 中的范围
            turn_start = pos
            turn_end = min(pos + turn_len, len(full_tokens))

            # 如果是 assistant 轮，标记内容部分
            if is_assistant:
                content_start = turn_start + header_len
                content_end = min(content_start + content_len, len(full_tokens))
                # 只标记内容部分，不标记头部和 im_end
                if content_start < len(full_tokens):
                    labels[content_start:content_end] = full_tokens[content_start:content_end]

            # 移动位置到该轮末尾
            pos = turn_end

            # 如果不是最后一轮，还要跳过分隔 newline
            if i < len(dialog) - 1 and pos < len(full_tokens):
                pos += 1  # 跳过分隔的 newline

            if pos >= len(full_tokens):
                break

        # EOS token 参与 loss（若存在且未被截断）
        if pos < len(full_tokens) and full_tokens[pos] == self.eos_id:
            labels[pos] = full_tokens[pos]

        return labels

    # ── 工具 ────────────────────────────────────────────────

    def _dummy_sample(self) -> Dict[str, torch.Tensor]:
        """为无效/过短样本生成占位输出。"""
        ids = [self.bos_id, self.eos_id]
        return {
            "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
            "labels": torch.tensor(ids[1:], dtype=torch.long),
            "attention_mask": torch.ones(len(ids) - 1, dtype=torch.bool),
        }


# ─────────────────────────── 批处理 ────────────────────────────


def collate_conversation_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    将长度不一的样本 pad 到批内最大长度。

    - input_ids / attention_mask 用 0 填充
    - labels 用 -100 填充（不计入 loss）
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    def pad(seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
        result = []
        for seq in seqs:
            gap = max_len - seq.size(0)
            if gap > 0:
                seq = torch.cat([seq, seq.new_full((gap,), pad_val)])
            result.append(seq)
        return torch.stack(result)

    return {
        "input_ids": pad([b["input_ids"] for b in batch], 0),
        "labels": pad([b["labels"] for b in batch], -100),
        "attention_mask": pad([b["attention_mask"] for b in batch], False),
    }


# ─────────────────────────── 推理工具 ────────────────────────────


def format_conversation_for_inference(
    history: List[Tuple[str, str]],
    current_input: str,
    max_history: int = 5,
) -> str:
    """
    将历史对话 + 当前输入格式化为推理用的 ChatML 文本。

    生成文本末尾以 <|im_start|>assistant 结尾，留给模型续写。

    Args:
        history:       历史对话，每项为 (role, content)，role 支持中英文
        current_input: 当前用户输入
        max_history:   保留最近 N 条历史

    Returns:
        ChatML 格式字符串（不含 EOS）
    """
    _role_map = {
        "用户": ROLE_USER,
        "user": ROLE_USER,
        "User": ROLE_USER,
        "助手": ROLE_ASSISTANT,
        "assistant": ROLE_ASSISTANT,
    }

    recent = history[-max_history:]
    lines: List[str] = []

    for raw_role, content in recent:
        role = _role_map.get(raw_role, ROLE_USER)
        lines.append(f"{IM_START}{role}\n{clean_chinese_text(content)}{IM_END}")

    current = clean_chinese_text(current_input)
    lines.append(f"{IM_START}{ROLE_USER}\n{current}{IM_END}")
    lines.append(f"{IM_START}{ROLE_ASSISTANT}")

    return "\n".join(lines)


def build_model_input(
    text: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """
    将 ChatML 格式文本编码为模型输入张量。

    格式: [BOS] + ChatML tokens

    Args:
        text:       由 format_conversation_for_inference 生成的文本
        tokenizer:  已加载的 tokenizer（需包含 im_start/im_end 特殊 token）
        max_length: 最大 token 长度（含 BOS）

    Returns:
        包含 input_ids 和 attention_mask 的字典（batch_size=1）
    """
    # 确保特殊 token 已注册
    new_tokens = [
        t for t in (IM_START, IM_END) if t not in tokenizer.additional_special_tokens
    ]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = nl[0] if nl else 10

    def encode(s: str) -> List[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    # 按行解析 ChatML 文本，正确识别特殊 token
    # 注意：内容本身可能包含换行符，因此以 IM_START 为块分隔符
    tokens: List[int] = []
    blocks = text.split(IM_START)  # 每个块: "role\ncontent<|im_end|>..." 或 ""

    for i, block in enumerate(blocks):
        if not block:
            continue
        tokens.append(im_start_id)

        if IM_END in block:
            # 正常块: role\ncontent<|im_end|>后续文本
            inner, rest = block.split(IM_END, 1)
            tokens.extend(encode(inner))
            tokens.append(im_end_id)
            if rest:
                tokens.extend(encode(rest))
        else:
            # 最后一个不完整块（推理时 assistant 开头）
            tokens.extend(encode(block))

    # 截断并添加前缀
    tokens = tokens[: max_length - 1]
    input_ids = [tokenizer.bos_token_id] + tokens

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(input_ids), dtype=torch.bool),
    }


# ─────────────────────────── Tokenizer 单例 ────────────────────────────


class TokenizerHelper:
    """
    轻量级 tokenizer 单例封装，避免重复加载模型。

    Usage:
        helper = TokenizerHelper("THUDM/chatglm-6b")
        ids = helper.encode("你好")
        text = helper.decode(ids)
    """

    _instances: Dict[str, "TokenizerHelper"] = {}

    def __new__(cls, tokenizer_name: str = "THUDM/chatglm-6b") -> "TokenizerHelper":
        if tokenizer_name not in cls._instances:
            instance = super().__new__(cls)
            instance._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
            cls._instances[tokenizer_name] = instance
        return cls._instances[tokenizer_name]

    def encode(self, text: str, clean_spaces: bool = True) -> List[int]:
        if clean_spaces:
            text = clean_chinese_text(text)
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )


# 便捷函数
def sentence_to_ids(
    sentence: str,
    tokenizer_name: str = "THUDM/chatglm-6b",
    clean_spaces: bool = True,
) -> List[int]:
    """句子 → token ID 列表。"""
    return TokenizerHelper(tokenizer_name).encode(sentence, clean_spaces)


def ids_to_sentence(
    token_ids: List[int],
    tokenizer_name: str = "THUDM/chatglm-6b",
    skip_special_tokens: bool = True,
) -> str:
    """token ID 列表 → 句子。"""
    return TokenizerHelper(tokenizer_name).decode(token_ids, skip_special_tokens)
