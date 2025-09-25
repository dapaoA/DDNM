#!/bin/bash

# 安装评估脚本的依赖

echo "=========================================="
echo "Installing DDNM Evaluation Dependencies"
echo "=========================================="

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install pip3 first."
    exit 1
fi

# 升级pip
echo "Upgrading pip..."
pip3 install --upgrade pip

# 安装PyTorch (根据系统选择)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装其他依赖
echo "Installing other dependencies..."
pip3 install -r requirements_evaluation.txt

# 验证安装
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

echo ""
echo "=========================================="
echo "Installation completed!"
echo "You can now run the evaluation script."
echo "=========================================="
