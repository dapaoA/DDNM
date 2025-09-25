#!/bin/bash

# DDNM任务评估脚本
# 使用方法: ./run_evaluation.sh [选项]

set -e

# 默认参数
EXP_DIR="/workspace/DDNM/exp"
TASK_FILTER=""
SKIP_FID=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --tasks)
            TASK_FILTER="$2"
            shift 2
            ;;
        --skip-fid)
            SKIP_FID="--skip_fid"
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --exp-dir DIR     指定exp目录路径 (默认: /workspace/DDNM/exp)"
            echo "  --tasks PATTERNS  过滤任务名称 (用空格分隔多个模式)"
            echo "  --skip-fid        跳过FID计算"
            echo "  --help            显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                    # 评估所有任务"
            echo "  $0 --tasks ffhq                      # 只评估FFHQ任务"
            echo "  $0 --tasks ffhq imagenet             # 评估FFHQ和ImageNet任务"
            echo "  $0 --skip-fid                        # 跳过FID计算"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "DDNM 任务评估脚本"
echo "=========================================="
echo "Exp目录: $EXP_DIR"
if [ -n "$TASK_FILTER" ]; then
    echo "任务过滤: $TASK_FILTER"
fi
if [ -n "$SKIP_FID" ]; then
    echo "跳过FID计算"
fi
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 检查评估脚本
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_SCRIPT="$SCRIPT_DIR/evaluate_all_tasks.py"

if [ ! -f "$EVALUATION_SCRIPT" ]; then
    echo "错误: 未找到评估脚本: $EVALUATION_SCRIPT"
    exit 1
fi

# 检查exp目录
if [ ! -d "$EXP_DIR" ]; then
    echo "错误: exp目录不存在: $EXP_DIR"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import torch, torchvision, torchmetrics, numpy, PIL" 2>/dev/null || {
    echo "警告: 某些依赖可能缺失。请运行:"
    echo "pip install -r requirements_evaluation.txt"
    echo ""
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# 运行评估
echo "开始评估..."
python3 "$EVALUATION_SCRIPT" \
    --exp_dir "$EXP_DIR" \
    $([ -n "$TASK_FILTER" ] && echo "--tasks $TASK_FILTER") \
    $SKIP_FID

echo ""
echo "=========================================="
echo "评估完成!"
echo "结果保存在: $EXP_DIR/evaluation_results/"
echo "=========================================="
