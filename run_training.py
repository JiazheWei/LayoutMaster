#!/usr/bin/env python3
"""
LayoutMaster训练和预测使用示例
"""

import subprocess
import os
import sys
from pathlib import Path

def run_training():
    """运行训练"""
    print("🚀 开始训练 LayoutMaster 模型")
    
    config_file = "config_data_action_head.json"
    
    cmd = [
        sys.executable, "train.py",
        "--config", config_file,
        "--mode", "train"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ 训练完成！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return False
    
    return True

def run_prediction(checkpoint_path, num_samples=10):
    """运行批量预测"""
    print(f"🔮 开始批量预测，样本数量: {num_samples}")
    
    config_file = "config_data_action_head.json"
    
    cmd = [
        sys.executable, "train.py",
        "--config", config_file,
        "--mode", "predict",
        "--checkpoint", checkpoint_path,
        "--num_samples", str(num_samples)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ 预测完成！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 预测失败: {e}")
        return False
    
    return True

def run_single_prediction(checkpoint_path, poster_dir):
    """运行单个海报预测"""
    print(f"🎯 对单个海报进行预测: {poster_dir}")
    
    config_file = "config_data_action_head.json"
    
    cmd = [
        sys.executable, "train.py",
        "--config", config_file,
        "--mode", "predict",
        "--checkpoint", checkpoint_path,
        "--poster_dir", poster_dir
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ 单个预测完成！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 预测失败: {e}")
        return False
    
    return True

def main():
    print("=== LayoutMaster 训练和预测工具 ===")
    print()
    print("选择运行模式:")
    print("1. 训练模型")
    print("2. 批量预测 (需要已训练的模型)")
    print("3. 单个海报预测 (需要已训练的模型)")
    print("4. 显示使用说明")
    print()
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == "1":
        # 训练模式
        run_training()
        
    elif choice == "2":
        # 批量预测模式
        checkpoint_path = input("请输入模型检查点路径 (如: ./checkpoints/best_model.pth): ").strip()
        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return
        
        try:
            num_samples = int(input("请输入要预测的样本数量 (默认10): ").strip() or "10")
        except ValueError:
            num_samples = 10
        
        run_prediction(checkpoint_path, num_samples)
        
    elif choice == "3":
        # 单个预测模式
        checkpoint_path = input("请输入模型检查点路径 (如: ./checkpoints/best_model.pth): ").strip()
        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return
        
        poster_dir = input("请输入海报目录路径: ").strip()
        if not os.path.exists(poster_dir):
            print(f"❌ 海报目录不存在: {poster_dir}")
            return
        
        run_single_prediction(checkpoint_path, poster_dir)
        
    elif choice == "4":
        # 显示使用说明
        show_usage()
        
    else:
        print("❌ 无效的选择")

def show_usage():
    """显示使用说明"""
    print()
    print("=== 使用说明 ===")
    print()
    print("📁 数据格式要求:")
    print("- 数据应存放在 /home/data-action_head/ 目录下")
    print("- 每个海报文件夹包含:")
    print("  - json/ 目录: ground truth JSON文件")
    print("  - parse/ 目录: PNG元素文件")
    print("  - trajectories.pkl: 轨迹数据文件")
    print()
    print("🚀 训练:")
    print("python run_training.py")
    print("或直接运行:")
    print("python train.py --config config_data_action_head.json --mode train")
    print()
    print("🔮 批量预测:")
    print("python train.py --config config_data_action_head.json --mode predict --checkpoint ./checkpoints/best_model.pth --num_samples 20")
    print()
    print("🎯 单个预测:")
    print("python train.py --config config_data_action_head.json --mode predict --checkpoint ./checkpoints/best_model.pth --poster_dir '/home/data-action_head/Ai Poster-xxx'")
    print()
    print("📋 输出文件:")
    print("- 训练: 模型检查点保存在 ./checkpoints/ 目录")
    print("- 预测: prediction.json 文件保存在对应的海报目录下")
    print()

if __name__ == "__main__":
    # 检查当前目录
    if not os.path.exists("train.py"):
        print("❌ 请在LayoutMaster目录下运行此脚本")
        sys.exit(1)
    
    main()
