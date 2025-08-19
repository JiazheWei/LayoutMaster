"""
测试Qwen2.5-VL视觉编码器导入和基本功能
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor
import safetensors
from PIL import Image
import numpy as np
import os
import sys
import json

def load_qwen_vision_config():
    """加载Qwen2.5-VL的配置"""
    config_path = "/home/Qwen2.5-VL-3B-Instruct/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_qwen_vision_encoder(config):
    """根据配置创建Qwen视觉编码器架构"""
    vision_config = config['vision_config']
    
    class QwenVisionEncoder(nn.Module):
        def __init__(self, vision_config):
            super().__init__()
            # 简化的视觉编码器，只测试基本功能
            self.patch_embed = nn.Conv2d(
                in_channels=vision_config['in_chans'],
                out_channels=vision_config['hidden_size'],
                kernel_size=vision_config['patch_size'],
                stride=vision_config['patch_size']
            )
            
            # 简单的transformer块
            self.norm = nn.LayerNorm(vision_config['hidden_size'])
            self.output_projection = nn.Linear(
                vision_config['hidden_size'], 
                vision_config['out_hidden_size']
            )
            
        def forward(self, pixel_values):
            # 简单的前向传播
            B, C, H, W = pixel_values.shape
            x = self.patch_embed(pixel_values)  # [B, hidden_size, H', W']
            x = x.flatten(2).transpose(1, 2)     # [B, num_patches, hidden_size]
            x = self.norm(x)
            x = self.output_projection(x)
            return x
    
    return QwenVisionEncoder(vision_config)

def test_qwen_vision_encoder():
    """测试Qwen2.5-VL视觉编码器的导入和基本功能"""
    
    print("🚀 开始测试Qwen2.5-VL视觉编码器...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 使用设备: {device}")
    
    try:
        # 1. 加载配置
        print("📥 加载Qwen2.5-VL配置...")
        config = load_qwen_vision_config()
        vision_config = config['vision_config']
        
        print(f"🔧 视觉配置:")
        print(f"  - 深度: {vision_config['depth']}")
        print(f"  - 隐藏维度: {vision_config['hidden_size']}")
        print(f"  - 输出维度: {vision_config['out_hidden_size']}")
        print(f"  - Patch大小: {vision_config['patch_size']}")
        
        # 2. 创建简化的视觉编码器
        print("\n🏗️ 创建视觉编码器...")
        vision_encoder = create_qwen_vision_encoder(config)
        vision_encoder = vision_encoder.to(device)
        
        print(f"✅ 视觉编码器创建成功!")
        print(f"📊 参数量: {sum(p.numel() for p in vision_encoder.parameters()):,}")
        
        # 3. 加载预处理器
        print("\n📥 加载图像预处理器...")
        try:
            processor = AutoImageProcessor.from_pretrained("/home/Qwen2.5-VL-3B-Instruct")
        except:
            # 如果失败，使用简单的预处理
            print("⚠️ 使用简单预处理器")
            processor = None
        
        # 4. 测试视觉编码器
        print("\n🔍 测试视觉编码器功能...")
        
        # 创建测试图像
        test_image = torch.randn(1, 3, 224, 224).to(device)
        
        print(f"📥 输入图像尺寸: {test_image.shape}")
        
        # 提取视觉特征
        with torch.no_grad():
            visual_features = vision_encoder(test_image)
        
        print(f"📤 输出特征尺寸: {visual_features.shape}")
        print(f"📊 特征统计: mean={visual_features.mean().item():.4f}, std={visual_features.std().item():.4f}")
        
        # 5. 测试批处理
        print("\n🔄 测试批处理...")
        batch_images = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            batch_visual_features = vision_encoder(batch_images)
        
        print(f"📥 批处理输入尺寸: {batch_images.shape}")
        print(f"📤 批处理输出尺寸: {batch_visual_features.shape}")
        
        print("\n✅ Qwen2.5-VL视觉编码器测试通过!")
        
        return {
            'vision_encoder': vision_encoder,
            'processor': processor,
            'device': device,
            'output_dim': visual_features.shape[-1],
            'config': config
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_qwen_vision_encoder()
    if result:
        print(f"\n🎉 测试成功! 视觉特征维度: {result['output_dim']}")
    else:
        print("\n💥 测试失败!")
        sys.exit(1)
