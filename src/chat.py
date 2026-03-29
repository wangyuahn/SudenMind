import os
import json
import sys

import torch
from transformers import AutoTokenizer

from model import SudenMind
from data_utils import (
    clean_chinese_text,
    extract_response,
    format_conversation_for_inference,
    build_model_input,
    IM_START,
    IM_END,
    ROLE_USER,
    ROLE_ASSISTANT,
)

DEBUG        = True
USE_EOS_STOP = True


def _load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    print(f"加载 tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    new_tokens = [
        t for t in (IM_START, IM_END)
        if t not in tokenizer.additional_special_tokens
    ]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    print(
        f"词表大小: {len(tokenizer)}  "
        f"BOS={tokenizer.bos_token_id}  EOS={tokenizer.eos_token_id}"
    )
    return tokenizer


def _load_model(model_cfg: dict, vocab_size: int, device: torch.device) -> SudenMind:
    model = SudenMind(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        d_fnn=model_cfg["d_fnn"],
        nhead=model_cfg.get("nhead", 8),
        dropout=model_cfg.get("dropout", 0.1),
        n_layers=model_cfg.get("n_layers", 6),
        num_experts=model_cfg.get("num_experts", 8),
        top_k=model_cfg.get("top_k", 2),
        aux_loss_coef=model_cfg.get("aux_loss_coef", 0.01),
        max_position_embeddings=model_cfg.get("max_position_embeddings", 5000),
    ).to(device)

    model_path = "model/sudenmind.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重: {model_path}，请先运行 train.py")

    state_dict = torch.load(model_path, map_location=device)

    # 处理词表扩展
    emb = state_dict.get("token_embedding.weight")
    if emb is not None and emb.shape[0] != vocab_size:
        print(f"词表不匹配 (权重={emb.shape[0]}, 当前={vocab_size})，重建 embedding")
        del state_dict["token_embedding.weight"]
        del state_dict["fc.3.weight"]
        del state_dict["fc.3.bias"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def chat():
    cfg       = json.load(open("config.json", "r", encoding="utf-8"))
    model_cfg = cfg["model"]
    gen_cfg   = cfg["generation"]
    data_cfg  = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    tokenizer   = _load_tokenizer(model_cfg.get("tokenizer_name", "THUDM/chatglm-6b"))
    model       = _load_model(model_cfg, len(tokenizer), device)
    max_history = data_cfg.get("max_history", 5)

    print("\n" + "=" * 50)
    print("SudenMind 对话系统")
    print("=" * 50)
    print("quit/exit — 退出 | clear — 清空 | history — 历史")
    print("-" * 50)

    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Assistant: 再见！")
            break

        if user_input.lower() in ("clear", "reset"):
            history = []
            print("Assistant: 已清空历史")
            continue

        if user_input.lower() in ("history", "h"):
            print("\n--- 对话历史 ---")
            for role, content in history:
                print(f"{'You' if role == ROLE_USER else 'Assistant'}: {content}")
            print("----------------")
            continue

        # 构造输入
        prompt = format_conversation_for_inference(
            history, user_input, max_history
        )

        input_dict = build_model_input(
            prompt,
            tokenizer,
            max_length=data_cfg.get("max_seq_len", 512),
        )

        # ⭐ 修复：必须有 batch 维
        input_tensor = input_dict["input_ids"].unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model.generate(
                input_seq=input_tensor,
                max_length=gen_cfg["max_length"],
                temperature=gen_cfg["temperature"],
                device=device,
                use_eos_stop=USE_EOS_STOP,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(
            output_tensor[0].tolist(),
            skip_special_tokens=False,
        )

        if DEBUG:
            print(f"[DEBUG] {full_text[:500]}")

        response = clean_chinese_text(extract_response(full_text))
        print(f"Assistant: {response}")

        history.append((ROLE_USER, user_input))
        history.append((ROLE_ASSISTANT, response))

        if len(history) > max_history * 2:
            history = history[-(max_history * 2):]


def test_tokenizer():
    tokenizer = _load_tokenizer("THUDM/chatglm-6b")

    sample = (
        f"{IM_START}user\n你好{IM_END}\n"
        f"{IM_START}assistant\n你好！很高兴为你服务。{IM_END}"
    )

    print("原始:\n", sample)

    token_ids = tokenizer.encode(sample, add_special_tokens=False)
    print(f"\nToken IDs ({len(token_ids)}): {token_ids}")

    full_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
    print("完整 IDs:", full_ids)

    print("\n解码:\n", tokenizer.decode(full_ids, skip_special_tokens=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-tokenizer":
        test_tokenizer()
    else:
        chat()