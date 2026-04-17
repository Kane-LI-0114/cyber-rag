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
    echo "                      [--provider <提供商>] [--model <回答模型>]"
    echo "                      [--judge-provider <Judge提供商>] [--judge-model <Judge模型>]"
    echo "                      别名: CM-01-v1, CM-01-v2, CM-80, CM-500, SecQA, CTF-MC, CTF-SA, test"
    echo ""
    echo -e "${GREEN}[评估分析]${NC}"
    echo "  analyze [CSV路径]    分析评估结果 (默认: 最新带时间戳的CSV文件)"
    echo "  analyze -v          详细文本报告"
    echo "  analyze --report <文件>   保存文本报告"
    echo "  analyze --md <文件>       保存Markdown格式报告"
    echo "  analyze --json <文件>     保存JSON摘要"
    echo "  analyze -e <类型>        导出错误案例 (rag_improved/rag_regressed/both_wrong)"
    echo ""
    echo "  示例:"
    echo "    ./run.sh analyze                              # 分析最新结果"
    echo "    ./run.sh analyze artifacts/evals/eval_xxx.csv # 分析指定CSV"
    echo "    ./run.sh analyze --md report.md               # 保存Markdown报告"
    echo "    ./run.sh analyze eval.csv --md report.md      # 指定CSV+保存MD"
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
    echo "  ./run.sh eval CM-80 --provider azure --model gpt-4o"
    echo "  ./run.sh eval CM-80 --provider oneapi --model DeepSeek-V3.2 --judge-model gpt-4o-mini"
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
    local dataset=""
    local provider=""
    local model=""
    local judge_provider=""
    local judge_model=""
    local extra_args=()

    # 解析参数
    while [ $# -gt 0 ]; do
        case "$1" in
            --provider)
                provider="$2"
                shift 2
                ;;
            --model)
                model="$2"
                shift 2
                ;;
            --judge-provider)
                judge_provider="$2"
                shift 2
                ;;
            --judge-model)
                judge_model="$2"
                shift 2
                ;;
            --*)
                extra_args+=("$1" "$2")
                shift 2
                ;;
            *)
                if [ -z "$dataset" ]; then
                    dataset="$1"
                fi
                shift
                ;;
        esac
    done

    if [ -z "$dataset" ]; then
        echo -e "${RED}错误: 请提供数据集路径或别名${NC}"
        echo "用法: ./run.sh eval <数据集> [--provider <提供商>] [--model <回答模型>] [--judge-provider <Judge提供商>] [--judge-model <Judge模型>]"
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
        echo "可选参数:"
        echo "  --provider <提供商>           回答模型提供商 (azure, oneapi, huggingface)"
        echo "  --model <模型名>             回答模型 (如 gpt-4o)"
        echo "  --judge-provider <提供商>     Judge模型提供商 (默认: --provider 或 .env)"
        echo "  --judge-model <模型名>       Judge模型 (如 gpt-4o-mini)"
        echo ""
        echo "示例:"
        echo "  ./run.sh eval CM-80"
        echo "  ./run.sh eval CM-80 --model gpt-4o"
        echo "  ./run.sh eval CM-80 --provider azure --model gpt-4o"
        echo "  ./run.sh eval CM-80 --provider oneapi --model DeepSeek-V3.2 --judge-model gpt-4o-mini"
        exit 1
    fi

    # Build output filename with model name and optional judge model name
    local model_tag=""
    local judge_tag=""
    if [ -n "$model" ]; then
        model_tag="_${model//\//-}"
        model_tag="${model_tag//:/-}"
    fi
    if [ -n "$judge_model" ]; then
        judge_tag="_judge-${judge_model//\//-}"
        judge_tag="${judge_tag//:/-}"
    fi
    local output="artifacts/evals/$(date +%Y%m%d_%H%M%S)${model_tag}${judge_tag}.csv"

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
    [ -n "$provider" ] && echo -e "${BLUE}    回答提供商: $provider${NC}"
    [ -n "$model" ] && echo -e "${BLUE}    回答模型: $model${NC}"
    [ -n "$judge_provider" ] && echo -e "${BLUE}    Judge提供商: $judge_provider${NC}"
    [ -n "$judge_model" ] && echo -e "${BLUE}    Judge模型: $judge_model${NC}"

    conda activate cyber-rag

    local cmd_args=("$dataset" "--index-path" "artifacts/indexes/default" "--output" "$output")
    [ -n "$provider" ] && cmd_args+=("--provider" "$provider")
    [ -n "$model" ] && cmd_args+=("--model" "$model")
    [ -n "$judge_provider" ] && cmd_args+=("--judge-provider" "$judge_provider")
    [ -n "$judge_model" ] && cmd_args+=("--judge-model" "$judge_model")
    [ ${#extra_args[@]} -gt 0 ] && cmd_args+=("${extra_args[@]}")

    python -m cyber_rag.cli.run_eval "${cmd_args[@]}"
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
        shift  # 移除 eval，保留其余参数
        run_eval "$@"
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
