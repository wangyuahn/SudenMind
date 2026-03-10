# chat.py
import torch
import json
import jieba
from model import Seq2Seq, Encoder, Decoder   # 确保 model.py 在同一目录或可导入

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 加载词汇表
def load_vocab(vocab_path: str):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    # 注意：json 中 id2word 的键是字符串，需要转回整数
    word2id = vocab_data['word2id']
    id2word = {int(k): v for k, v in vocab_data['id2word'].items()}
    return word2id, id2word

word2id, id2word = load_vocab('vocab.json')
vocab_size = len(word2id)
print(f"词汇表加载完成，大小: {vocab_size}")

# 特殊标记 ID
PAD_ID = word2id['<PAD>']
UNK_ID = word2id['<UNK>']
SOS_ID = word2id['<SOS>']
EOS_ID = word2id['<EOS>']

# 中文分词函数
def tokenize(text: str):
    return list(jieba.cut(text))

# 将用户输入转换为模型输入 ID 序列
def encode_sentence(sentence: str):
    tokens = tokenize(sentence)
    ids = [word2id.get(token, UNK_ID) for token in tokens]
    # 添加 <SOS> 和 <EOS>
    ids = [SOS_ID] + ids + [EOS_ID]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # 添加 batch 维度

# 将模型输出的 ID 序列转换为可读文本
def decode_ids(ids: list):
    # ids 是一个列表或 1D tensor
    words = []
    for idx in ids:
        idx = idx.item() if torch.is_tensor(idx) else idx
        word = id2word.get(int(idx), '<UNK>')
        if word in ('<PAD>', '<UNK>', '<SOS>', '<EOS>'):
            continue
        words.append(word)
    return ''.join(words)

# 加载模型
def load_model(model_path: str, vocab_size: int, embed_size: int = 256, hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.5):
    encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
    decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 推理生成回答
def generate_response(model: Seq2Seq, input_tensor: torch.Tensor, max_len=1000):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        # 编码器前向
        hidden, cell = model.encoder(input_tensor)
        
        # decoder 初始输入为 <SOS>
        decoder_input = torch.tensor([[SOS_ID]], device=device)  # (1, 1)
        
        generated_ids = []
        for _ in range(max_len):
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            # 应用 softmax 获取概率分布
            output = torch.softmax(output, dim=-1)

            # 获取概率最高的前 k 个词及其概率
            top_probs, top_indices = torch.topk(output, k=3, dim=-1)
            probs = top_probs[0]  # 取第一个样本的概率分布

            # 根据概率分布采样
            sampled_idx = torch.multinomial(probs, num_samples=1).item()
            predicted_id = top_indices[0, int(sampled_idx)].item()  # 获取对应的词 ID
            # 选择概率最高的词作为预测
            # predicted_id = torch.argmax(output, dim=-1).item()

            generated_ids.append(predicted_id)
            
            if predicted_id == EOS_ID:
                break
                
            # 下一步输入为当前预测的词
            decoder_input = torch.tensor([[predicted_id]], device=device)
        
        return generated_ids

# 主交互循环
def chat_loop(model):
    print("聊天机器人已启动（输入 'exit' 退出）")
    while True:
        user_input = input("[user]: ").strip()
        if user_input.lower() == 'exit':
            print("[bot]: 再见！")
            break
        if not user_input:
            print("[bot]: 请输入一些内容...")
            continue
        # 编码用户输入
        input_tensor = encode_sentence(user_input)  # (1, seq_len)
        
        # 生成回答
        output_ids = generate_response(model, input_tensor)
        
        # 解码为中文
        response = decode_ids(output_ids)
        print(f"[bot]: {response}")

if __name__ == '__main__':
    # 模型参数（必须与训练时一致）
    EMBED_SIZE = 256
    HIDDEN_SIZE = 1024
    NUM_LAYERS = 1
    DROPOUT = 0.5
    
    # 模型文件路径（根据实际情况修改）
    model_path = 'model/chat_model.pth'  # 如果放在当前目录可直接用 'chat_model.pth'
    # model_path = 'model/pretrained_model.pth'  # 如果放在当前目录可直接用 'pretrained_model.pth'
    
    # 加载模型
    model = load_model(model_path, vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    print("模型加载完成，开始聊天！")
    
    # 启动对话
    chat_loop(model)