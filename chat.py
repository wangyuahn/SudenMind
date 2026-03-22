"""
SudenMind 交互式对话脚本

支持与训练好的 SudenMind 模型进行实时对话。
使用 top-k 采样策略生成响应，支持温度控制。

特性：
1. 实时交互式对话
2. 支持温度控制 (控制生成随机性)
3. 自动处理特殊 token (SOS, SEP, EOS)
4. 支持 MoE 模型加载

作者：SudenMind 团队
版本：2.0 (集成 MoE)
"""

import torch
import json
import os
from model import SudenMind
from datasets import sentence_to_ids, ids_to_words

# 加载配置文件
cfg = json.load(open("config.json", "r", encoding="utf-8"))
model_cfg = cfg["model"]  # 模型配置
gen_cfg = cfg["generation"]  # 生成配置


def chat():
    """
    交互式对话主函数

    加载训练好的模型和词表，进入交互式对话循环。
    用户输入问题，模型生成回答。

    流程：
    1. 检测可用设备 (GPU/CPU)
    2. 加载词表
    3. 初始化模型 (包含 MoE 参数)
    4. 加载模型权重
    5. 进入交互循环
    """
    # 检测设备 (优先使用 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用 {device} 进行推理...")

    # 1. 加载词表
    vocab_path = "data/vocab.json"
    if not os.path.exists(vocab_path):
        print(f"错误：未找到 {vocab_path}，请先运行 process.py 生成数据和词表。")
        return

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    word2id = vocab_data["word2id"]
    # json 加载后键是字符串，需要转换为 int
    id2word = {int(k): v for k, v in vocab_data["id2word"].items()}
    vocab_size = len(word2id)

    # 2. 初始化模型架构 (参数从 config.json 读取，需与 train.py 保持一致)
    embedding_dim = model_cfg["d_model"]
    hidden_dim = model_cfg["d_fnn"]
    output_dim = vocab_size

    # 添加 MoE 参数
    num_experts = model_cfg.get("num_experts", 4)
    top_k = model_cfg.get("top_k", 2)
    aux_loss_coef = model_cfg.get("aux_loss_coef", 0.01)

    model = SudenMind(
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_experts=num_experts,
        top_k=top_k,
        aux_loss_coef=aux_loss_coef,
    ).to(device)

    # 3. 加载训练好的权重
    model_path = "model/sudenmind.pth"
    if not os.path.exists(model_path):
        print(f"错误：未找到模型权重文件 {model_path}，请先运行 train.py 进行训练。")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式
    print("\n成功加载 SudenMind 模型！现在可以开始聊天了。(输入 'quit' 或 'exit' 退出)")
    print("-" * 50)

    # 4. 交互循环
    while True:
        user_input = input("你: ")

        # 退出指令
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("SudenMind: 再见！")
            break

        # 防止空输入
        if not user_input.strip():
            continue

        # 第一步：分词并转换为 ID
        input_ids = sentence_to_ids(user_input, word2id)

        # 第二步：构造 Prompt 序列 -> [SOS] + 问题 + [SEP]
        sos_id = word2id.get("<SOS>", 2)
        sep_id = word2id.get("<SEP>", 4)
        eos_id = word2id.get("<EOS>", 3)

        prompt_ids = [sos_id] + input_ids + [sep_id]

        # 转换为 Tensor，保持 (batch_size, seq_len) 格式，这里 batch_size=1
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        # 第三步：生成回答
        with torch.no_grad():
            # temperature 控制随机性：越接近0越固定，越接近1越有创意
            output_tensor = model.generate(
                input_seq=input_tensor,
                max_length=gen_cfg["max_length"],
                temperature=gen_cfg["temperature"],
                device=device,
            )

        # 第四步：解码输出
        # output_tensor 的形状是 (1, total_seq_len)，我们要截取掉前面的 prompt 部分
        generated_ids = output_tensor[0][len(prompt_ids) :].tolist()

        # 使用 ids_to_words 解码
        response_words = ids_to_words(generated_ids, id2word)
        response_text = "".join(response_words)
        print(f"SudenMind: {response_text}")


if __name__ == "__main__":
    chat()
