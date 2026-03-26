import torch
import json
import os
from model import SudenMind
from data_utils import (
    format_conversation_for_inference,
    sentence_to_ids,
    ids_to_sentence,
)
from transformers import BertTokenizer


def chat():
    # 加载配置
    cfg = json.load(open("config.json", "r", encoding="utf-8"))
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用 {device} 进行推理...")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    vocab_size = tokenizer.vocab_size

    embedding_dim = model_cfg["d_model"]
    hidden_dim = model_cfg["d_fnn"]
    output_dim = vocab_size

    bert_model_name = model_cfg.get("bert_model_name", "bert-base-chinese")
    freeze_bert = model_cfg.get("freeze_bert", True)
    num_experts = model_cfg.get("num_experts", 4)
    top_k = model_cfg.get("top_k", 2)
    aux_loss_coef = model_cfg.get("aux_loss_coef", 0.01)

    model = SudenMind(
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        bert_model_name=bert_model_name,
        freeze_bert=freeze_bert,
        num_experts=num_experts,
        top_k=top_k,
        aux_loss_coef=aux_loss_coef,
    ).to(device)

    model_path = "model/sudenmind.pth"
    if not os.path.exists(model_path):
        print(f"错误：未找到模型权重文件 {model_path}，请先运行 train.py 进行训练。")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("\n成功加载 SudenMind 模型！现在可以开始聊天了。(输入 'quit' 或 'exit' 退出)")
    print("-" * 50)

    conversation_history = []
    max_history = data_cfg.get("max_history", 5)

    while True:
        user_input = input("你: ")

        if user_input.lower() in ["quit", "exit", "退出"]:
            print("SudenMind: 再见！")
            break

        if user_input.lower() in ["clear", "重置", "清空"]:
            conversation_history = []
            print("SudenMind: 对话历史已清空")
            continue

        if not user_input.strip():
            continue

        # 将对话历史转换为 (speaker, utterance) 格式
        formatted_history = []
        for i, utterance in enumerate(conversation_history):
            speaker = "用户" if i % 2 == 0 else "助手"
            formatted_history.append((speaker, utterance))

        formatted_input = format_conversation_for_inference(
            formatted_history, user_input, max_history
        )

        input_ids = tokenizer.encode(
            formatted_input,
            add_special_tokens=False,
            max_length=data_cfg.get("max_seq_len", 512),
            truncation=True,
        )

        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        with torch.no_grad():
            output_tensor = model.generate(
                input_seq=input_tensor,
                max_length=gen_cfg["max_length"],
                temperature=gen_cfg["temperature"],
                device=device,
            )

        generated_ids = output_tensor[0][len(input_ids) :].tolist()

        eos_token_id = tokenizer.sep_token_id
        if eos_token_id in generated_ids:
            eos_index = generated_ids.index(eos_token_id)
            generated_ids = generated_ids[:eos_index]

        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"SudenMind: {response_text}")

        conversation_history.append(user_input)
        conversation_history.append(response_text)

        if len(conversation_history) > max_history * 2:
            conversation_history = conversation_history[-(max_history * 2) :]


def test_bert_tokenizer():
    print("=== 测试BERT分词器 ===")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    test_text = "你好，今天天气怎么样？"
    print(f"测试文本: {test_text}")

    tokens = tokenizer.tokenize(test_text)
    print(f"分词结果: {tokens}")

    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"Token IDs: {token_ids}")

    decoded = tokenizer.decode(token_ids)
    print(f"解码结果: {decoded}")

    print(f"\n特殊token:")
    print(f"  [CLS]: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"  [SEP]: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"  [PAD]: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  [UNK]: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test-tokenizer":
        test_bert_tokenizer()
    else:
        chat()
