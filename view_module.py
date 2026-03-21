import netron
netron.start('model/sudenmind.onnx', browse=True)

while True:
    cmd = input("输入 'exit' 退出 Netron 可视化: ")
    if cmd.strip().lower() == 'exit':
        print("正在关闭 Netron...")
        netron.stop()
        break
