#!/bin/bash
# RL Learning Project - 安装脚本
# 环境: CUDA 12.2 / RTX 4090
# 推荐 Python: 3.10 或 3.11

set -e

echo "=== RL Learning Project 依赖安装 ==="
echo "检测到 CUDA 12.2，使用 PyTorch cu121 构建 (兼容 CUDA 12.1/12.2)"
echo ""

# 1. 先安装 PyTorch (CUDA 12.1，兼容你的 CUDA 12.2)
echo "[1/2] 安装 PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. 安装其余依赖
echo "[2/2] 安装其余依赖..."
pip install -r requirements.txt

echo ""
echo "=== 安装完成 ==="
echo "验证 GPU: python -c \"import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)\""
