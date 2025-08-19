#!/bin/bash

# LayoutMaster 8GPU训练一键启动脚本

echo "🚀 LayoutMaster 8GPU训练启动"
echo "=================================="

# 检查当前目录
if [ ! -f "train.py" ]; then
    echo "❌ 请在LayoutMaster目录下运行此脚本"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export TORCH_CUDNN_V8_API_ENABLED=1

# 显示GPU状态
echo "🔍 当前GPU状态:"-format=csv,noheader

nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total -
echo ""
echo "⚙️ 训练配置:"
echo "- 使用GPU: 8x NVIDIA H20"
echo "- 批次大小: 32 (每GPU 4个样本)"
echo "- 工作进程: 32"
echo "- 学习率: 4e-4"
echo "- 数据目录: /home/data-action_head"
echo ""

# 确认开始训练
read -p "是否开始训练? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "🛑 训练已取消"
    exit 0
fi

# 创建日志目录
mkdir -p logs

# 启动训练（带日志记录）
echo "🎯 开始训练，日志保存到 logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python train.py --config config_data_action_head.json --mode predict 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ 训练完成！"
echo "📁 检查 ./checkpoints/ 目录查看保存的模型"
echo "📋 检查 ./logs/ 目录查看训练日志"
