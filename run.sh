#!/bin/bash
# CyberRAG 常用命令脚本

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo -e "${BLUE}CyberRAG 常用命令${NC}"
    echo ""
    echo "用法: ./run.sh <命令> [参数]"
    echo ""
    echo "可用命令:"
    echo ""
    echo -e "${GREEN}[环境配置]${NC}"
    echo "  setup              安装/更新依赖"
    echo "  check              检查配置状态"
    echo ""
    echo -e "${GREEN}[索引构建]${NC}"
    echo "  build-index        构建 FAISS 索引"
    echo "  build-index-custom <chunk_size> <chunk_overlap>  自定义分块参数"
    echo ""
    echo -e "${GREEN}[查询与评估]${NC}"
    echo "  query <问题>        单条检索问答"
    echo "  eval <数据集路径>   批量评估 (baseline vs RAG)"
    echo ""
    echo -e "${GREEN}[测试]${NC}"
    echo "  test               运行全部测试"
    echo "  test-chunking      运行分块测试"
    echo "  lint               代码风格检查"
    echo ""
    echo -e "${GREEN}[其他]${NC}"
    echo "  help               显示本帮助信息"
    echo ""
    echo "示例:"
    echo "  ./run.sh setup"
    echo "  ./run.sh build-index"
    echo "  ./run.sh query \"什么是XSS攻击？\""
    echo "  ./run.sh eval CyberMetric-80-v1.jsonl"
}

# 环境配置
setup() {
    echo -e "${YELLOW}>>> 安装/更新依赖...${NC}"
    conda activate cyber-rag
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo -e "${GREEN}>>> 依赖安装完成${NC}"
}

# 检查配置
check() {
    echo -e "${YELLOW}>>> 检查配置状态...${NC}"
    conda activate cyber-rag
    python -m cyber_rag.cli.check_config
}

# 构建索引
build_index() {
    echo -e "${YELLOW}>>> 构建 FAISS 索引...${NC}"
    conda activate cyber-rag
    python -m cyber_rag.cli.build_index data/raw --index-path artifacts/indexes/default
    echo -e "${GREEN}>>> 索引构建完成${NC}"
}

# 自定义参数构建索引
build_index_custom() {
    local chunk_size=${1:-1000}
    local chunk_overlap=${2:-200}
    echo -e "${YELLOW}>>> 构建 FAISS 索引 (chunk_size=$chunk_size, overlap=$chunk_overlap)...${NC}"
    conda activate cyber-rag
    python -m cyber_rag.cli.build_index data/raw \
        --index-path artifacts/indexes/default \
        --chunk-size "$chunk_size" \
        --chunk-overlap "$chunk_overlap"
    echo -e "${GREEN}>>> 索引构建完成${NC}"
}

# 单条查询
run_query() {
    if [ -z "$1" ]; then
        echo -e "${RED}错误: 请提供问题${NC}"
        echo "用法: ./run.sh query <问题>"
        exit 1
    fi
    echo -e "${YELLOW}>>> 执行检索问答: $1${NC}"
    conda activate cyber-rag
    python -m cyber_rag.cli.run_query "$1"
}

# 批量评估
run_eval() {
    if [ -z "$1" ]; then
        echo -e "${RED}错误: 请提供数据集路径${NC}"
        echo "用法: ./run.sh eval <数据集路径>"
        exit 1
    fi
    local dataset="$1"
    local output=${2:-artifacts/evals/latest.csv}
    echo -e "${YELLOW}>>> 批量评估: $dataset${NC}"
    conda activate cyber-rag
    python -m cyber_rag.cli.run_eval "$dataset" \
        --index-path artifacts/indexes/default \
        --output "$output"
    echo -e "${GREEN}>>> 评估完成，结果保存至: $output${NC}"
}

# 运行测试
run_tests() {
    echo -e "${YELLOW}>>> 运行全部测试...${NC}"
    conda activate cyber-rag
    pytest
    echo -e "${GREEN}>>> 测试完成${NC}"
}

# 运行分块测试
test_chunking() {
    echo -e "${YELLOW}>>> 运行分块测试...${NC}"
    conda activate cyber-rag
    pytest tests/test_chunking.py
    echo -e "${GREEN}>>> 分块测试完成${NC}"
}

# 代码检查
run_lint() {
    echo -e "${YELLOW}>>> 代码风格检查...${NC}"
    conda activate cyber-rag
    ruff check cyber_rag scripts tests
    echo -e "${GREEN}>>> 检查完成${NC}"
}

# 主逻辑
case "$1" in
    setup)
        setup
        ;;
    check)
        check
        ;;
    build-index)
        build_index
        ;;
    build-index-custom)
        build_index_custom "$2" "$3"
        ;;
    query)
        shift
        run_query "$*"
        ;;
    eval)
        run_eval "$2" "$3"
        ;;
    test)
        run_tests
        ;;
    test-chunking)
        test_chunking
        ;;
    lint)
        run_lint
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo -e "${RED}未知命令: $1${NC}"
        show_help
        exit 1
        ;;
esac
