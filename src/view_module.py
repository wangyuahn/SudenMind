"""
SudenMind 模型可视化工具

使用 Netron 可视化导出的 ONNX 模型。
可以查看完整的模型结构，包括 MoE 层。

使用方法：
1. 先运行 train.py 导出 ONNX 模型
2. 运行本脚本: python view_module.py
3. 浏览器会自动打开显示模型结构

注意：需要安装 netron: pip install netron

作者：SudenMind 团队
版本：2.0
"""

import netron

# 启动 Netron 可视化
netron.start("model/sudenmind.onnx", browse=True)

# 保持脚本运行，直到用户输入退出
while True:
    cmd = input("输入 'exit' 退出 Netron 可视化: ")
    if cmd.strip().lower() == "exit":
        print("正在关闭 Netron...")
        netron.stop()
        break
