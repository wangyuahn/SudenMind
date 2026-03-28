import torch
import os
import json
import re
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
from transformers import AutoTokenizer
from datasets import load_dataset

# ShareGPT/ChatML 格式常量
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM = "user", "assistant", "system"


class ConversationDataset(Dataset):
    """
    ShareGPT/ChatML 格式的对话数据集。
    格式: [gMASK] [BOS] <|im_start|>user\n{问题}<|im_end|>\n<|im_start|>assistant\n{回答}<|im_end|> ... [EOS]
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
        self.split, self.config = split, config
        self.max_history, self.max_length = max_history, max_length
        self.test_mode, self.clean_spaces = test_mode, clean_spaces
        self.cache_dir = cache_dir or "data/cache"
        self.use_cache = use_cache

        # 加载 tokenizer
        print(f"加载 tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )

        # 设置特殊 token
        self._setup_special_tokens()

        # 加载数据
        self._load_data()

    def _setup_special_tokens(self):
        """设置 ChatGLM 和 ShareGPT 特殊 token。"""
        # ChatGLM-6B 常量
        self.gmask_id = 64790
        self.bos_id = 64792
        self.eos_id = getattr(self.tokenizer, "eos_token_id", 2)
        self.pad_id = getattr(self.tokenizer, "pad_token_id", 3)

        # 添加 ShareGPT token（确保 <|im_start|> 是一个独立 token）
        try:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [IM_START, IM_END]}
            )
            self.im_start_id = self.tokenizer.convert_tokens_to_ids(IM_START)
            self.im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)
        except Exception as e:
            print(f"警告: 添加特殊 token 失败: {e}")
            self.im_start_id = self.im_end_id = None

        # 缓存换行符 token ID
        newline_tokens = self._encode_text("\n")
        self.newline_id = newline_tokens[0] if newline_tokens else 10

        print(
            f"词表大小: {len(self.tokenizer)}, gMASK: {self.gmask_id}, BOS: {self.bos_id}"
        )

    def _encode_text(self, text: str) -> List[int]:
        """文本编码为 token ID 列表。"""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _load_data(self):
        """从缓存加载或下载数据集。"""
        cache_file = os.path.join(
            self.cache_dir, f"lccc_{self.config}_{self.split}.json"
        )

        if self.use_cache and os.path.exists(cache_file):
            print(f"从缓存加载: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
            print(f"已加载 {len(self.dataset)} 条对话")
            return

        print(f"加载 LCCC: thu-coai/lccc, config={self.config}, split={self.split}")
        split_str = f"{self.split}[:100]" if self.test_mode else self.split
        raw = load_dataset("thu-coai/lccc", self.config, split=split_str)

        self.dataset = [
            {
                "dialog": [self._clean_text(u) for u in item["dialog"]]
                if self.clean_spaces
                else item["dialog"]
            }
            for item in raw
            if len(item.get("dialog", [])) >= 2
        ]

        print(f"已加载 {len(self.dataset)} 条对话")

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.dataset, f, ensure_ascii=False)
            print(f"已缓存到: {cache_file}")

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗中文文本中的空格。"""
        if not text:
            return text
        # 移除中文字符间的空格
        text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
        # 移除中文标点前后的空格
        for pat in [
            r'\s+([，。！？；："\'《》【】（）])',
            r'([，！？；："\'《》【】（）])\s+',
        ]:
            text = re.sub(pat, r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dialog = self.dataset[idx]["dialog"]

        if len(dialog) < 2:
            return self._dummy_sample()

        # 确保偶数轮次并限制历史长度
        dialog = (
            dialog[-self.max_history * 2 :]
            if len(dialog) > self.max_history * 2
            else dialog
        )
        if len(dialog) % 2 != 0:
            dialog = dialog[:-1]

        # 构建完整序列: [gMASK] [BOS] <对话> [EOS]
        full_tokens = (
            [self.gmask_id, self.bos_id]
            + self._build_dialog_tokens(dialog)
            + [self.eos_id]
        )
        full_tokens = full_tokens[: self.max_length]

        # 找到助手回答的开始位置
        assistant_start_pos = self._find_assistant_start_position(full_tokens, dialog)
        
        # 创建labels: 助手回答之前的部分设为-100（不计算损失）
        labels = [-100] * len(full_tokens)
        if assistant_start_pos < len(full_tokens):
            # 助手回答部分正常计算损失
            labels[assistant_start_pos:] = full_tokens[assistant_start_pos:]

        # 自回归训练：input_ids是完整序列（去掉最后一个token）
        # labels是向右偏移一个位置的序列
        return {
            "input_ids": torch.tensor(full_tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(labels[1:], dtype=torch.long),
            "attention_mask": torch.ones(len(full_tokens) - 1, dtype=torch.bool),
        }

    def _build_dialog_tokens(self, dialog: List[str]) -> List[int]:
        """将对话构建为 ShareGPT 格式的 token 列表。

        格式: <|im_start|>user\n内容<|im_end|>\n<|im_start|>assistant\n内容<|im_end|>...
        """
        tokens = []
        for i, content in enumerate(dialog):
            role = ROLE_USER if i % 2 == 0 else ROLE_ASSISTANT

            # 构建: <|im_start|>role\ncontent<|im_end|>\n
            tokens.extend(
                [self.im_start_id] if self.im_start_id else self._encode_text(IM_START)
            )
            tokens.extend(self._encode_text(role))
            tokens.append(self.newline_id)
            tokens.extend(self._encode_text(content))
            tokens.extend(
                [self.im_end_id] if self.im_end_id else self._encode_text(IM_END)
            )
            if i < len(dialog) - 1:
                tokens.append(self.newline_id)

        return tokens
    
    def _find_assistant_start_position(self, full_tokens: List[int], dialog: List[str]) -> int:
        """找到助手回答在token序列中的开始位置。"""
        # 找到最后一个助手回复的开始
        for i in range(len(dialog)-1, -1, -1):
            if i % 2 == 1:  # 助手轮次
                # 构建助手开头的token序列
                role = ROLE_ASSISTANT
                assistant_start = []
                assistant_start.extend([self.im_start_id] if self.im_start_id else self._encode_text(IM_START))
                assistant_start.extend(self._encode_text(role))
                assistant_start.append(self.newline_id)
                
                # 在full_tokens中查找这个模式
                pattern = assistant_start
                for pos in range(len(full_tokens) - len(pattern) + 1):
                    if full_tokens[pos:pos+len(pattern)] == pattern:
                        return pos + len(pattern)  # 返回助手内容开始的位置
        
        # 如果找不到，默认从中间位置开始
        return len(full_tokens) // 2

    def _dummy_sample(self) -> Dict[str, torch.Tensor]:
        """为无效数据创建空样本。"""
        ids = [self.gmask_id, self.bos_id, self.eos_id]
        return {
            "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
            "labels": torch.tensor(ids[1:], dtype=torch.long),
            "attention_mask": torch.ones(len(ids) - 1, dtype=torch.bool),
        }


def collate_conversation_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """批处理填充函数。"""
    max_len = max(len(item["input_ids"]) for item in batch)

    def pad(sequences, pad_value):
        return torch.stack(
            [
                torch.cat(
                    [seq, torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype)]
                )
                if len(seq) < max_len
                else seq
                for seq in sequences
            ]
        )

    return {
        "input_ids": pad([b["input_ids"] for b in batch], 0),
        "labels": pad([b["labels"] for b in batch], -100),
        "attention_mask": pad([b["attention_mask"] for b in batch], False),
    }


def format_conversation_for_inference(
    history: List[Tuple[str, str]],
    current_input: str,
    max_history: int = 5,
) -> str:
    """格式化对话用于推理（ShareGPT 格式）。

    Args:
        history: 历史对话列表，每项为 (角色, 内容)
        current_input: 当前用户输入
        max_history: 最大历史轮数

    Returns:
        格式化后的对话文本（不含 assistant 结尾标记）
    """
    # 清洗并规范化角色名
    history = history[-max_history:] if len(history) > max_history else history
    cleaned = []
    for role, content in history:
        role = ROLE_USER if role in ("用户", "user", "User") else ROLE_ASSISTANT
        cleaned.append((role, ConversationDataset._clean_text(content)))

    current = ConversationDataset._clean_text(current_input)

    # 构建文本
    lines = [f"{IM_START}{r}\n{c}{IM_END}" for r, c in cleaned]
    lines.extend(
        [f"{IM_START}{ROLE_USER}\n{current}{IM_END}", f"{IM_START}{ROLE_ASSISTANT}"]
    )

    return "\n".join(lines)


def build_model_input(
    text: str,
    tokenizer: AutoTokenizer,
    gmask_id: int = 64790,
    bos_id: int = 64792,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """从格式化文本构建模型输入。

    格式: [gMASK] [BOS] text

    注意: 正确处理特殊 token <|im_start|> 和 <|im_end|>
    """
    # 确保特殊 token 已添加
    if IM_START not in getattr(tokenizer, "additional_special_tokens", []):
        tokenizer.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})

    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = newline_tokens[0] if newline_tokens else 10

    # 解析并编码文本
    tokens = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(IM_START):
            tokens.append(im_start_id)
            tokens.extend(
                tokenizer.encode(line[len(IM_START) :], add_special_tokens=False)
            )
        elif line.endswith(IM_END):
            tokens.extend(
                tokenizer.encode(line[: -len(IM_END)], add_special_tokens=False)
            )
            tokens.append(im_end_id)
        elif line == IM_END:
            tokens.append(im_end_id)
        else:
            tokens.extend(tokenizer.encode(line, add_special_tokens=False))
        if i < len(lines) - 1:
            tokens.append(newline_id)

    tokens = tokens[: max_length - 3]
    input_ids = [gmask_id, bos_id] + tokens

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(input_ids), dtype=torch.bool),
    }


def extract_response(text: str) -> str:
    """从生成文本中提取助手回复。"""
    if not text:
        return ""

    assistant_start = f"{IM_START}{ROLE_ASSISTANT}"

    if assistant_start not in text:
        return text.strip()

    idx = text.find(assistant_start) + len(assistant_start)
    remaining = text[idx:]

    if IM_END in remaining:
        response = remaining.split(IM_END)[0]
        if response.strip():
            return response.strip()

    return remaining.strip()


class TokenizerHelper:
    """Tokenizer 辅助类（单例模式，复用 tokenizer）。"""

    _instances: Dict[str, "TokenizerHelper"] = {}

    def __new__(cls, tokenizer_name: str = "THUDM/chatglm-6b"):
        if tokenizer_name not in cls._instances:
            cls._instances[tokenizer_name] = super().__new__(cls)
            cls._instances[tokenizer_name]._initialized = False
        return cls._instances[tokenizer_name]

    def __init__(self, tokenizer_name: str = "THUDM/chatglm-6b"):
        if self._initialized:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self._initialized = True

    def encode(self, text: str, clean_spaces: bool = True) -> List[int]:
        """文本编码为 token ID 列表。"""
        if clean_spaces:
            text = ConversationDataset._clean_text(text)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """token ID 列表解码为文本。"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


# 便捷函数（使用 TokenizerHelper 单例）
def sentence_to_ids(
    sentence: str, tokenizer_name: str = "THUDM/chatglm-6b", clean_spaces: bool = True
) -> List[int]:
    """句子转为 token ID 列表。"""
    return TokenizerHelper(tokenizer_name).encode(sentence, clean_spaces)


def ids_to_sentence(
    token_ids: List[int],
    tokenizer_name: str = "THUDM/chatglm-6b",
    skip_special_tokens: bool = True,
) -> str:
    """token ID 列表转为句子。"""
    return TokenizerHelper(tokenizer_name).decode(token_ids, skip_special_tokens)