import torch
import json
import os
from model import SudenMind
from data_utils import (
    format_conversation_for_inference,
    build_model_input,
    extract_response,
    ConversationDataset,
    IM_START,
    IM_END,
    ROLE_USER,
    ROLE_ASSISTANT,
)
from transformers import AutoTokenizer

DEBUG = True
GMASK_ID = 64790
BOS_ID = 64792
USE_EOS_STOP = False


def chat():
    """交互式对话"""
    # 加载配置
    cfg = json.load(open("config.json", "r", encoding="utf-8"))
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用 {device} 进行推理...")

    # 加载ChatGLM tokenizer
    tokenizer_name = model_cfg.get("tokenizer_name", "THUDM/chatglm-6b")
    print(f"正在加载ChatGLM tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )

    # 添加 ShareGPT/ChatML 特殊标记
    try:
        tokenizer.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
        vocab_size = len(tokenizer)
    except Exception as e:
        print(f"添加特殊标记失败: {e}")
        vocab_size = 65024

    print(f"ChatGLM词表大小: {vocab_size}")

    # ChatGLM特殊token
    gmask_token_id = GMASK_ID
    bos_token_id = BOS_ID
    print(f"  [gMASK] ID: {gmask_token_id}")
    print(f"  [BOS] ID: {bos_token_id}")

    # 初始化模型
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
        print(f"错误：未找到模型权重文件 {model_path}，请先运行 train.py 进行训练。")
        return

    # 加载权重，处理词表扩展情况
    state_dict = torch.load(model_path, map_location=device)
    embedding_weight = state_dict.get("token_embedding.weight")
    if embedding_weight is not None and embedding_weight.shape[0] != vocab_size:
        print(f"词表大小不匹配: 权重={embedding_weight.shape[0]}, 当前={vocab_size}")
        print("重新初始化 token embedding 和输出层...")
        del state_dict["token_embedding.weight"]
        del state_dict["fc.3.weight"]
        del state_dict["fc.3.bias"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("\n" + "=" * 50)
    print("SudenMind 对话系统 (ShareGPT/ChatML 格式)")
    print("=" * 50)
    print("命令: 'quit' 退出 | 'clear' 清空历史 | 'history' 查看历史")
    print("-" * 50)

    conversation_history = []  # [(role, content), ...]
    max_history = data_cfg.get("max_history", 5)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nAssistant: 再见！")
            break

        if user_input.lower() in ["clear", "reset"]:
            conversation_history = []
            print("Assistant: 对话历史已清空")
            continue

        if user_input.lower() in ["history", "h"]:
            print("\n--- 对话历史 ---")
            for role, content in conversation_history:
                display_role = "You" if role == ROLE_USER else "Assistant"
                print(f"{display_role}: {content}")
            print("----------------")
            continue

        # 格式化输入（ShareGPT格式）
        prompt = format_conversation_for_inference(
            conversation_history, user_input, max_history
        )

        # 构建模型输入
        input_dict = build_model_input(
            prompt,
            tokenizer,
            gmask_id=gmask_token_id,
            bos_id=bos_token_id,
            max_length=data_cfg.get("max_seq_len", 512),
        )
        input_tensor = input_dict["input_ids"].to(device)

        # 生成回复
        with torch.no_grad():
            output_tensor = model.generate(
                input_seq=input_tensor,
                max_length=gen_cfg["max_length"],
                temperature=gen_cfg["temperature"],
                device=device,
                use_eos_stop=USE_EOS_STOP,
            )

        # 解码生成的文本
        generated_ids = output_tensor[0].tolist()
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        if DEBUG:
            print(f"[DEBUG] full_text: {full_text[:500]}")

        # 提取助手回复
        response_text = extract_response(full_text)
        response_text = ConversationDataset._clean_text(response_text)

        print(f"Assistant: {response_text}")

        # 更新对话历史
        conversation_history.append((ROLE_USER, user_input))
        conversation_history.append((ROLE_ASSISTANT, response_text))

        # 限制历史长度
        max_turns = max_history * 2
        if len(conversation_history) > max_turns:
            conversation_history = conversation_history[-max_turns:]


def test_tokenizer():
    """测试tokenizer"""
    print("=== 测试 ShareGPT/ChatML 格式 ===\n")

    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )

    # 添加特殊标记
    tokenizer.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
    print(f"词表大小（添加特殊标记后）: {tokenizer.vocab_size}")
    print(f"'<|im_start|>' ID: {tokenizer.convert_tokens_to_ids(IM_START)}")
    print(f"'<|im_end|>' ID: {tokenizer.convert_tokens_to_ids(IM_END)}")
    print()

    # 测试文本
    conversation = """<|im_start|>user
你好
<|im_end|>
<|im_start|>assistant
你好！很高兴为你服务。
<|im_end|>"""

    print("原始对话格式:")
    print(conversation)
    print()

    # 编码
    token_ids = tokenizer.encode(conversation, add_special_tokens=False)
    print(f"Token IDs: {token_ids}")
    print(f"Token数量: {len(token_ids)}")
    print()

    # ChatGLM完整格式
    gmask_id = 64790
    bos_id = 64792
    eos_id = 2

    full_ids = [gmask_id, bos_id] + token_ids + [eos_id]
    print(f"完整输入 IDs: {full_ids}")
    print()

    # 解码
    decoded = tokenizer.decode(full_ids, skip_special_tokens=False)
    print("解码结果:")
    print(decoded)

    print("\n=== 特殊Token ===")
    print(f"[gMASK] ID: {gmask_id}")
    print(f"[BOS] ID: {bos_id}")
    print(f"[EOS] ID: {eos_id}")
    print(f"词表大小: {tokenizer.vocab_size}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test-tokenizer":
        test_tokenizer()
    else:
        chat()
