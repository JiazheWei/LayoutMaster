#!/usr/bin/env python3
"""
测试模型初始化和backbone设置
"""

import torch
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append('.')

from train import LayoutTrainer, load_config

def test_model_initialization():
    """测试模型初始化和backbone设置"""
    print("🧪 测试模型初始化...")
    
    config_file = "config_data_action_head.json"
    
    if not Path(config_file).exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    try:
        # 加载配置
        config = load_config(config_file)
        
        # 临时设置小批次用于测试
        config['data']['batch_size'] = 2
        config['use_multi_gpu'] = False  # 测试时使用单GPU
        
        print("✅ 配置文件加载成功")
        
        # 创建训练器
        print("🔧 创建训练器...")
        trainer = LayoutTrainer(config)
        
        print("✅ 训练器创建成功")
        print(f"✅ 设备: {trainer.device}")
        print(f"✅ 多GPU: {trainer.is_multi_gpu}")
        
        # 测试数据加载器
        print("📊 测试数据加载...")
        try:
            batch = next(iter(trainer.train_loader))
            print("✅ 数据加载成功")
            
            # 测试前向传播
            print("🚀 测试模型前向传播...")
            
            # 移动数据到设备
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(trainer.device)
            
            # 提取数据
            images = batch['images']  # [B, N, 3, H, W]
            start_layouts = batch['start_layouts']  # [B, N, 6]
            element_masks = batch['element_masks']  # [B, N]
            
            print(f"  图片形状: {images.shape}")
            print(f"  起始布局形状: {start_layouts.shape}")
            print(f"  元素掩码形状: {element_masks.shape}")
            
            # 测试视觉特征提取
            print("👁️ 测试视觉特征提取...")
            trainer.visual_extractor.eval()
            with torch.no_grad():
                visual_features = trainer.visual_extractor(images)
                print(f"✅ 视觉特征形状: {visual_features.shape}")
            
            # 测试完整模型前向传播
            print("🧠 测试布局模型...")
            trainer.layout_model.eval()
            
            # 创建时间步
            B = images.shape[0]
            timesteps = torch.randint(0, 1000, (B,), device=trainer.device)
            
            # 标准化布局
            normalized_layouts = trainer.normalizer.normalize_layout(start_layouts)
            
            with torch.no_grad():
                predicted_noise = trainer.layout_model(
                    visual_features=visual_features,
                    layout_state=normalized_layouts,
                    timestep=timesteps,
                    element_mask=element_masks
                )
                print(f"✅ 模型输出形状: {predicted_noise.shape}")
            
            print("\n🎉 所有测试通过！模型初始化和前向传播正常。")
            return True
            
        except Exception as e:
            print(f"❌ 模型测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ 训练器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== LayoutMaster 模型初始化测试 ===")
    print()
    
    success = test_model_initialization()
    
    if success:
        print("\n✅ 模型初始化测试成功！")
        print("🚀 现在可以开始正式训练了:")
        print("   ./start_training.sh")
    else:
        print("\n❌ 模型初始化测试失败，请检查错误信息。")
    
    return success

if __name__ == "__main__":
    main()
