#!/bin/bash
# CyberRAG 常用命令脚本

set -e  # 遇到错误时退出

# 初始化 conda (支持非交互式 shell)
if command -v conda &> /dev/null; then
    # 如果 conda 是函数，直接使用
    eval "$(command conda shell.bash hook 2>/dev/null)" 2>/dev/null || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

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
    echo "  eval <别名/路径>    批量评估 (baseline vs RAG)"
    echo "                      别名: CM-01-v1, CM-01-v2, CM-80, CM-500, SecQA, CTF-MC, CTF-SA, test"
    echo ""
    echo -e "${GREEN}[评估分析]${NC}"
    echo "  analyze [CSV路径]    分析评估结果 (默认: 最新带时间戳的CSV文件)"
    echo "  analyze -v          详细文本报告"
    echo "  analyze --report <文件>   保存文本报告"
    echo "  analyze --json <文件>     保存JSON摘要"
    echo "  analyze -e <类型>        导出错误案例 (rag_improved/rag_regressed/both_wrong)"
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
    echo "  ./run.sh eval CM-80"
    echo "  ./run.sh eval CM-500"
    echo "  ./run.sh eval SecQA"
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
        echo -e "${RED}错误: 请提供数据集路径或别名${NC}"
        echo "用法: ./run.sh eval <数据集路径或别名>"
        echo ""
        echo "支持的别名:"
        echo "  CM-01-v1, CM-01-v2     -> CyberMetric-01 数据集 (1条)"
        echo "  CM-80, CM-80-v1        -> CyberMetric-80 数据集 (80条)"
        echo "  CM-500, CM-500-v1      -> CyberMetric-500 数据集 (500条)"
        echo "  SecQA                  -> SecQA 安全问答数据集"
        echo "  CTF-MC                 -> CTFKnow 多选题数据集"
        echo "  CTF-SA                 -> CTFKnow 简答题数据集"
        echo "  test                   -> 测试数据集"
        echo ""
        echo "示例:"
        echo "  ./run.sh eval CM-80"
        echo "  ./run.sh eval eval_datasets/CyberMetric-80-v1.jsonl"
        exit 1
    fi

    local dataset="$1"
    local output=${2:-artifacts/evals/$(date +%Y%m%d_%H%M%S).csv}

    # 解析别名
    case "$dataset" in
        CM-01-v1|cm-01-v1)
            dataset="eval_datasets/CyberMetric-01-v1.jsonl"
            ;;
        CM-01-v2|cm-01-v2)
            dataset="eval_datasets/CyberMetric-01-v2.jsonl"
            ;;
        CM-80|cm-80|CM-80-v1|cm-80-v1)
            dataset="eval_datasets/CyberMetric-80-v1.jsonl"
            ;;
        CM-500|cm-500|CM-500-v1|cm-500-v1)
            dataset="eval_datasets/CyberMetric-500-v1.jsonl"
            ;;
        SecQA|secqa|SECQA)
            dataset="eval_datasets/SecQA.jsonl"
            ;;
        CTF-MC|ctf-mc|CTFMC|ctfmc)
            dataset="eval_datasets/ctfknow_multiple_choice.jsonl"
            ;;
        CTF-SA|ctf-sa|CTFSA|ctfsa)
            dataset="eval_datasets/ctfknow_short_answer.jsonl"
            ;;
        test|TEST)
            dataset="eval_datasets/test.jsonl"
            ;;
    esac

    # 检查文件是否存在
    if [ ! -f "$dataset" ]; then
        echo -e "${RED}错误: 数据集文件不存在: $dataset${NC}"
        exit 1
    fi

    echo -e "${YELLOW}>>> 批量评估: $dataset${NC}"
    conda activate cyber-rag
    python -m cyber_rag.cli.run_eval "$dataset" \
        --index-path artifacts/indexes/default \
        --output "$output"
    echo -e "${GREEN}>>> 评估完成，结果保存至: $output${NC}"
}

# 分析评估结果
analyze() {
    local csv_path
    csv_path=$(ls -t artifacts/evals/eval_*.csv 2>/dev/null | head -1 || echo "artifacts/evals/latest.csv")
    local extra_args=""

    # 检查第一个参数是否为文件路径
    if [ -n "$1" ] && [ ! "${1:0:1}" = "-" ]; then
        csv_path="$1"
        shift
    fi

    if [ ! -f "$csv_path" ]; then
        echo -e "${RED}错误: 文件不存在: $csv_path${NC}"
        exit 1
    fi

    echo -e "${YELLOW}>>> 分析评估结果: $csv_path${NC}"
    conda activate cyber-rag
    python scripts/analyze_eval.py "$csv_path" "$@"
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
    analyze)
        shift  # 移除 analyze，保留其余参数
        analyze "$@"
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
