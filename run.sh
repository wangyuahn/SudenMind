#!/bin/bash
# SudenMind-BERT 启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数定义
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  SudenMind-BERT 启动脚本${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 检查环境
check_environment() {
    print_info "检查Python环境..."
    
    if ! command -v python &> /dev/null; then
        print_error "未找到Python，请先安装Python"
        exit 1
    fi
    
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Python版本: $python_version"
    
    # 检查必要包
    print_info "检查必要包..."
    
    required_packages=("torch" "transformers" "jieba")
    missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -eq 0 ]; then
        print_success "所有必要包已安装"
    else
        print_warning "缺少包: ${missing_packages[*]}"
        print_info "请运行: pip install ${missing_packages[*]}"
    fi
}

# 主菜单
show_menu() {
    echo ""
    echo "请选择操作:"
    echo "1. 检查LCCC-base数据"
    echo "2. 训练模型 (train.py)"
    echo "3. 对话测试 (chat.py)"
    echo "4. 运行完整测试"
    echo "5. 检查环境"
    echo "6. 查看目录结构"
    echo "7. 退出"
    echo ""
    read -p "请输入选项 [1-7]: " choice
}

# 数据检查
run_data_check() {
    print_header
    print_info "检查LCCC-base数据..."
    
    print_info "LCCC-base数据集信息:"
    echo "  📊 数据集: thu-coai/lccc"
    echo "  ⚙️  配置: base (LCCC-base)"
    echo "  📈 对话数量: 6,820,506"
    echo "  📝 描述: 大规模清洗中文对话语料"
    echo "  🔗 来源: 清华大学CoAI组"
    
    print_info "数据将在训练时自动从Hugging Face加载"
    print_info "如需离线使用，请先下载数据集"
    
    echo ""
    print_info "安装必要依赖:"
    echo "  pip install datasets"
}

# 训练模型
run_training() {
    print_header
    print_info "开始训练模型..."
    
    # 检查处理后的数据
    if [ ! -f "data/processed/chat_data.json" ]; then
        print_warning "未找到处理后的数据，请先运行数据处理"
        return
    fi
    
    # 运行训练
    python src/train.py
    
    if [ $? -eq 0 ]; then
        print_success "训练完成"
        print_info "模型保存在: model/sudenmind.pth"
    else
        print_error "训练失败"
    fi
}

# 对话测试
run_chat() {
    print_header
    print_info "启动对话测试..."
    
    # 检查模型
    if [ ! -f "model/sudenmind.pth" ]; then
        print_warning "未找到训练好的模型，请先训练模型"
        return
    fi
    
    # 运行对话
    python src/chat.py
}

# 运行测试
run_tests() {
    print_header
    print_info "运行完整测试..."
    
    if [ -f "tests/test_integration.py" ]; then
        python tests/test_integration.py
    else
        print_error "未找到测试文件"
    fi
}

# 检查目录结构
show_structure() {
    print_header
    print_info "项目目录结构:"
    echo ""
    
    # 显示主要目录
    find . -maxdepth 2 -type d | sort | grep -v "__pycache__" | grep -v ".git" | sed 's|\./||' | while read -r dir; do
        if [ -n "$dir" ]; then
            echo "  📁 $dir"
            # 显示该目录下的文件
            find "$dir" -maxdepth 1 -type f -name "*.py" -o -name "*.json" -o -name "*.md" -o -name "*.txt" 2>/dev/null | sed 's|^|    📄 |'
        fi
    done
    
    echo ""
    print_info "数据文件:"
    echo "  📊 LCCC-base数据集"
    echo "    来源: lccc (Hugging Face)"
    echo "    配置: base"
    echo "    对话数: 6,820,506"
    echo "    描述: 大规模清洗中文对话语料"
    
    if [ -f "model/sudenmind.pth" ]; then
        echo "  📄 model/sudenmind.pth"
        file_size=$(du -h "model/sudenmind.pth" 2>/dev/null | cut -f1 || echo "未知")
        echo "    大小: $file_size"
    fi
}

# 主函数
main() {
    print_header
    
    # 检查环境
    check_environment
    
    while true; do
        show_menu
        
        case $choice in
            1)
                run_data_check
                ;;
            2)
                run_training
                ;;
            3)
                run_chat
                ;;
            4)
                run_tests
                ;;
            5)
                check_environment
                ;;
            6)
                show_structure
                ;;
            7)
                print_info "再见！"
                exit 0
                ;;
            *)
                print_error "无效选项，请重新选择"
                ;;
        esac
    done
}

# 运行主函数
main "$@"