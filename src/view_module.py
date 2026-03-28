"""
模型可视化模块
使用 Netron 可视化 ONNX 模型
"""

import netron
import sys


def view_model(model_path: str = "model/sudenmind.onnx", browse: bool = True):
    """
    启动 Netron 可视化模型

    Args:
        model_path: ONNX 模型路径
        browse: 是否自动打开浏览器
    """
    print(f"正在启动 Netron 可视化: {model_path}")
    print("提示: 按 Ctrl+C 或在浏览器中关闭窗口退出")
    
    try:
        netron.start(model_path, browse=browse)
        
        # 保持脚本运行
        while True:
            try:
                cmd = input("\n输入 'exit' 或 'quit' 退出可视化: ")
                if cmd.strip().lower() in ("exit", "quit", "q"):
                    break
            except KeyboardInterrupt:
                break
    finally:
        print("正在关闭 Netron...")
        netron.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化 SudenMind 模型")
    parser.add_argument(
        "--model-path",
        type=str,
        default="model/sudenmind.onnx",
        help="ONNX 模型路径 (默认: model/sudenmind.onnx)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开浏览器"
    )
    
    args = parser.parse_args()
    view_model(args.model_path, browse=not args.no_browser)
