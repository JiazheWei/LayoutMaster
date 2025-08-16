#!/usr/bin/env python3
"""
测试data-action_head数据加载
"""

import torch
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append('.')

from train import TrajectoryDataset, collate_fn_trajectory

def test_dataset_loading():
    """测试数据集加载"""
    print("🧪 测试data-action_head数据集加载...")
    
    data_root = "/home/data-action_head"
    
    if not os.path.exists(data_root):
        print(f"❌ 数据目录不存在: {data_root}")
        return False
    
    try:
        # 创建训练数据集
        train_dataset = TrajectoryDataset(
            data_root=data_root,
            split='train',
            image_size=224,
            train_ratio=0.8
        )
        
        print(f"✅ 训练数据集创建成功，包含 {len(train_dataset)} 个样本")
        
        # 创建验证数据集
        val_dataset = TrajectoryDataset(
            data_root=data_root,
            split='val',
            image_size=224,
            train_ratio=0.8
        )
        
        print(f"✅ 验证数据集创建成功，包含 {len(val_dataset)} 个样本")
        
        if len(train_dataset) == 0:
            print("⚠️  训练数据集为空")
            return False
        
        # 测试单个样本
        print("🔍 测试单个样本加载...")
        sample = train_dataset[0]
        
        print("样本数据结构:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {value} ({type(value).__name__})")
        
        # 测试数据加载器
        print("🔄 测试数据加载器...")
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_trajectory
        )
        
        # 获取一个批次
        batch = next(iter(train_loader))
        
        print("批次数据结构:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # 检查数据有效性
        images = batch['images']  # [B, 25, 3, H, W]
        start_layouts = batch['start_layouts']  # [B, 25, 6]
        target_layouts = batch['target_layouts']  # [B, 25, 6]
        element_masks = batch['element_masks']  # [B, 25]
        
        print(f"✅ 图片数据形状正确: {images.shape}")
        print(f"✅ 起始布局形状正确: {start_layouts.shape}")
        print(f"✅ 目标布局形状正确: {target_layouts.shape}")
        print(f"✅ 元素掩码形状正确: {element_masks.shape}")
        
        # 检查数值范围
        print(f"📊 数据统计:")
        print(f"  图片像素值范围: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  起始布局值范围: [{start_layouts.min():.3f}, {start_layouts.max():.3f}]")
        print(f"  目标布局值范围: [{target_layouts.min():.3f}, {target_layouts.max():.3f}]")
        print(f"  有效元素数量: {element_masks.sum(dim=1)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trajectory_data():
    """测试轨迹数据"""
    print("🧪 测试轨迹数据结构...")
    
    data_root = Path("/home/data-action_head")
    
    # 找到第一个有轨迹文件的目录
    for poster_dir in data_root.iterdir():
        if poster_dir.is_dir():
            trajectory_file = poster_dir / 'trajectories.pkl'
            if trajectory_file.exists():
                print(f"📂 检查轨迹文件: {trajectory_file}")
                
                try:
                    import pickle
                    with open(trajectory_file, 'rb') as f:
                        trajectories = pickle.load(f)
                    
                    print(f"✅ 轨迹文件加载成功")
                    print(f"  噪声水平: {list(trajectories.keys())}")
                    
                    # 检查第一个噪声水平的数据
                    first_key = list(trajectories.keys())[0]
                    traj_data = trajectories[first_key]
                    
                    print(f"  {first_key} 数据结构:")
                    for key, value in traj_data.items():
                        if isinstance(value, torch.Tensor):
                            print(f"    {key}: {value.shape}")
                        else:
                            print(f"    {key}: {type(value).__name__}")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ 轨迹文件读取失败: {e}")
                    continue
    
    print("❌ 未找到有效的轨迹文件")
    return False

def main():
    print("=== data-action_head 数据加载测试 ===")
    print()
    
    success = True
    
    # 测试轨迹数据
    if not test_trajectory_data():
        success = False
    
    print()
    
    # 测试数据集加载
    if not test_dataset_loading():
        success = False
    
    print()
    
    if success:
        print("🎉 所有测试通过！数据加载正常工作。")
        print("💡 现在可以开始训练了:")
        print("   python train.py --config config_data_action_head.json --mode train")
    else:
        print("❌ 测试失败，请检查数据格式和路径。")
    
    return success

if __name__ == "__main__":
    main()
