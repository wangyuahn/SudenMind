import torch
import json
import jieba
import os
from model import SudenMind

def chat():
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用 {device} 进行推理...")
    
    # 1. 加载词表
    vocab_path = 'data/vocab.json'
    if not os.path.exists(vocab_path):
        print(f"错误：未找到 {vocab_path}，请先运行 process.py 生成数据和词表。")
        return
        
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        
    word2id = vocab_data['word2id']
    id2word = vocab_data['id2word']
    vocab_size = len(word2id)

    # 2. 初始化模型架构 (参数需与 train.py 保持完全一致)
    embedding_dim = 256
    hidden_dim = 512
    output_dim = vocab_size
    
    model = SudenMind(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    
    # 3. 加载训练好的权重
    model_path = 'model/sudenmind.pth'
    if not os.path.exists(model_path):
        print(f"错误：未找到模型权重文件 {model_path}，请先运行 train.py 进行训练。")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换到评估模式
    print("\n成功加载 SudenMind 模型！现在可以开始聊天了。(输入 'quit' 或 'exit' 退出)")
    print("-" * 50)

    # 4. 交互循环
    while True:
        user_input = input("\n你: ")
        
        # 退出指令
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("SudenMind: 再见！")
            break
            
        # 防止空输入
        if not user_input.strip():
            continue

        # 第一步：分词并转换为 ID
        tokens = list(jieba.cut(user_input, HMM=True))
        input_ids = [word2id.get(tok, word2id.get('<UNK>', 1)) for tok in tokens]
        
        # 第二步：构造 Prompt 序列 -> [SOS] + 问题 + [SEP]
        sos_id = word2id.get('<SOS>', 2)
        sep_id = word2id.get('<SEP>', 4)
        eos_id = word2id.get('<EOS>', 3)
        
        prompt_ids = [sos_id] + input_ids + [sep_id]
        
        # 转换为 Tensor，保持 (batch_size, seq_len) 格式，这里 batch_size=1
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        # 第三步：生成回答
        with torch.no_grad():
            # temperature 控制随机性：越接近0越固定，越接近1越有创意
            output_tensor = model.generate(
                input_seq=input_tensor, 
                max_length=50, 
                temperature=0.8, 
                device=device
            )
        
        # 第四步：解码输出
        # output_tensor 的形状是 (1, total_seq_len)，我们要截取掉前面的 prompt 部分
        generated_ids = output_tensor[0][len(prompt_ids):].tolist()
        
        response_words = []
        for idx in generated_ids:
            if idx == eos_id:  # 遇到 <EOS> 代表模型认为话说完了
                break
                
            word = id2word.get(str(idx), '<UNK>')
            # 过滤掉特殊的控制字符
            if word not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<SEP>']:
                response_words.append(word)
        
        response_text = "".join(response_words)
        print(f"SudenMind: {response_text}")

if __name__ == "__main__":
    chat()