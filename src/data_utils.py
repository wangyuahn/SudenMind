import torch
import os
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional
from transformers import BertTokenizer
from datasets import load_dataset


class LCCCDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        config: str = "base",
        max_history: int = 5,
        max_length: int = 512,
        tokenizer_name: str = "bert-base-chinese",
        test_mode: bool = False,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        clean_spaces: bool = True,  # 新增：是否清理空格
    ):
        self.split = split
        self.config = config
        self.max_history = max_history
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.test_mode = test_mode
        self.cache_dir = cache_dir or "data/cache"
        self.use_cache = use_cache
        self.clean_spaces = clean_spaces  # 保存清理空格选项

        # 创建缓存目录
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(
                self.cache_dir, f"thu_coai_lccc_{config}_{split}.json"
            )

            if os.path.exists(cache_file) and self.use_cache:
                print(f"从缓存加载数据集: {cache_file}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.dataset = json.load(f)
                print(f"缓存数据集加载完成，共 {len(self.dataset)} 个对话")
            else:
                print(f"加载LCCC数据集: thu-coai/lccc, config={config}, split={split}")
                if test_mode:
                    # 测试模式下只加载少量数据
                    self.dataset = load_dataset(
                        "thu-coai/lccc", config, split=f"{split}[:100]"
                    )
                    print(f"数据集加载完成（测试模式），共 {len(self.dataset)} 个对话")
                else:
                    self.dataset = load_dataset("thu-coai/lccc", config, split=split)
                    print(f"数据集加载完成，共 {len(self.dataset)} 个对话")

                # 转换为列表并保存到缓存
                self.dataset = [item for item in self.dataset]
                if self.use_cache:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(self.dataset, f, ensure_ascii=False, indent=2)
                    print(f"数据集已缓存到: {cache_file}")
        else:
            print(f"加载LCCC数据集: thu-coai/lccc, config={config}, split={split}")
            if test_mode:
                # 测试模式下只加载少量数据
                self.dataset = load_dataset(
                    "thu-coai/lccc", config, split=f"{split}[:100]"
                )
                print(f"数据集加载完成（测试模式），共 {len(self.dataset)} 个对话")
            else:
                self.dataset = load_dataset("thu-coai/lccc", config, split=split)
                print(f"数据集加载完成，共 {len(self.dataset)} 个对话")

        self.special_tokens = {
            "pad": self.tokenizer.pad_token_id,
            "unk": self.tokenizer.unk_token_id,
            "cls": self.tokenizer.cls_token_id,
            "sep": self.tokenizer.sep_token_id,
            "eos": self.tokenizer.sep_token_id,
        }

    @staticmethod
    def clean_lccc_text(text: str) -> str:
        """
        清理LCCC数据集文本中的空格
        LCCC数据集已经预先分词，每个词之间用空格分隔
        这个方法移除这些分词空格，让文本恢复正常格式
        """
        if not text:
            return text
        cleaned = text

        # 移除中文词之间的空格（中文不需要词间空格）
        # 但保留英文单词之间的空格
        import re
        # 处理中文文本：移除中文字符之间的空格
        # 匹配模式：中文字符 + 空格 + 中文字符
        cleaned = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", cleaned)
        # 处理中文标点：移除标点前的空格
        cleaned = re.sub(r'\s+([，。！？；："\'《》【】（）])', r"\1", cleaned)
        cleaned = re.sub(r'([，！？；："\'《》【】（）])\s+', r"\1", cleaned) # 处理中文标点：移除标点后的空格（除了句号后可能需要空格）
        cleaned = re.sub(r"\s+", " ", cleaned) # 移除多余的空格（多个连续空格变为一个）
        cleaned = cleaned.strip() # 移除首尾空格

        return cleaned

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if isinstance(self.dataset, list):
            dialog = self.dataset[idx]["dialog"]
        else:
            dialog = self.dataset[idx]["dialog"]

        if len(dialog) < 2:
            return self._create_dummy_sample()

        if len(dialog) % 2 != 0:
            dialog = dialog[:-1]

        turns = len(dialog)

        if turns == 2:
            history = []
            question = dialog[0]
            answer = dialog[1]
        else:
            history_turns = min(self.max_history * 2, turns - 2)
            history = dialog[:history_turns]
            question = dialog[history_turns]
            answer = (
                dialog[history_turns + 1] if history_turns + 1 < len(dialog) else ""
            )

        # 清理LCCC数据集中的空格
        if self.clean_spaces:
            history = [self.clean_lccc_text(utt) for utt in history]
            question = self.clean_lccc_text(question)
            answer = self.clean_lccc_text(answer)

        input_text = self._format_input(history, question)
        target_text = answer

        input_ids = self.tokenizer.encode(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            max_length=self.max_length - len(input_ids),
            truncation=True,
            padding=False,
        )

        target_ids = target_ids + [self.special_tokens["eos"]]

        full_seq = input_ids + target_ids
        input_seq = full_seq[:-1]
        target_seq = full_seq[1:]

        return {
            "input_ids": torch.tensor(input_seq, dtype=torch.long),
            "labels": torch.tensor(target_seq, dtype=torch.long),
            "attention_mask": torch.ones(len(input_seq), dtype=torch.bool),
        }

    def _format_input(self, history: List[str], question: str) -> str:
        if not history:
            return f"[CLS]{question}[SEP]"

        formatted = "[CLS]"
        for i, utterance in enumerate(history):
            if i % 2 == 0:
                formatted += f"用户:{utterance}"
            else:
                formatted += f"助手:{utterance}"
            formatted += "[SEP]"

        formatted += f"用户:{question}[SEP]"
        return formatted

    def _create_dummy_sample(self):
        dummy_input = torch.tensor(
            [self.special_tokens["cls"], self.special_tokens["sep"]], dtype=torch.long
        )
        dummy_target = torch.tensor(
            [self.special_tokens["sep"], self.special_tokens["eos"]], dtype=torch.long
        )
        return {
            "input_ids": dummy_input,
            "labels": dummy_target,
            "attention_mask": torch.ones(len(dummy_input), dtype=torch.bool),
        }


def collate_lccc_batch(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )
    # attention_mask: 1 for real tokens, 0 for padding tokens
    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_masks_padded,
    }


def sentence_to_ids(sentence: str, clean_spaces: bool = True) -> List[int]:
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # 清理空格（如果来自LCCC数据集）
    if clean_spaces:
        sentence = LCCCDataset.clean_lccc_text(sentence)
    return tokenizer.encode(sentence, add_special_tokens=False)


def ids_to_sentence(token_ids: List[int], clean_spaces: bool = True) -> str:
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    # 如果需要，进一步清理空格
    if clean_spaces:
        text = LCCCDataset.clean_lccc_text(text)
    return text


def format_conversation_for_inference(
    history: List[Tuple[str, str]], current_question: str, max_history: int = 5
) -> str:
    # 清理输入文本中的空格（如果来自LCCC数据集）
    cleaned_history = []
    for speaker, utterance in history:
        cleaned_utterance = LCCCDataset.clean_lccc_text(utterance)
        cleaned_history.append((speaker, cleaned_utterance))

    cleaned_question = LCCCDataset.clean_lccc_text(current_question)

    if not cleaned_history:
        return f"[CLS]用户:{cleaned_question}[SEP]"

    formatted = "[CLS]"

    history_turns = min(len(cleaned_history), max_history)
    recent_history = cleaned_history[-history_turns:] if history_turns > 0 else []

    for speaker, utterance in recent_history:
        formatted += f"{speaker}: {utterance}[SEP]"

    formatted += f"用户:{cleaned_question}[SEP]"
    return formatted
