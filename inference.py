"""
布局推理脚本
使用训练好的模型生成布局并输出JSON格式
"""

import torch
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Optional
import os
import glob

from models import create_layout_model
from diffusion_utils import DiffusionScheduler, LayoutNormalizer
from data_utils import layout_to_json


class LayoutInference:
    """
    布局推理器
    """
    
    def __init__(self, model_path: str, config_path: str, backbone_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 保存backbone路径
        self.backbone_path = backbone_path
        
        # 初始化模型
        self._init_models()
        
        # 初始化扩散组件
        self._init_diffusion()
        
        # 加载模型权重
        self._load_model(model_path)
        
        # 图片变换
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"推理器初始化完成，使用设备: {self.device}")
    
    def _init_models(self):
        """初始化模型"""
        model_config = self.config['model']
        
        self.layout_model, self.visual_extractor = create_layout_model(
            visual_backbone=model_config.get('visual_backbone', 'Qwen-VL-2.5-3B-Instruct'),
            visual_dim=model_config.get('visual_dim', 1280),
            hidden_dim=model_config.get('hidden_dim', 2048),
            num_layers=model_config.get('num_layers', 32),
            max_elements=model_config.get('max_elements', 25)
        )
        
        self.layout_model = self.layout_model.to(self.device)
        self.visual_extractor = self.visual_extractor.to(self.device)
        
        # 设置为评估模式
        self.layout_model.eval()
        self.visual_extractor.eval()
        
        # 加载视觉backbone
        self._load_visual_backbone()
        
        print("模型初始化完成")
    
    def _load_visual_backbone(self):
        """加载视觉backbone"""
        backbone_path = self.backbone_path or self.config.get('backbone_path')
        
        if backbone_path and os.path.exists(backbone_path):
            try:
                print(f"正在加载视觉backbone: {backbone_path}")
                
                # 根据backbone类型加载
                if 'clip' in backbone_path.lower():
                    # 加载CLIP模型
                    import clip
                    model, preprocess = clip.load("ViT-B/32", device=self.device)
                    self.visual_extractor.set_backbone(model)
                    print("CLIP视觉backbone加载成功")
                    
                elif 'qwen' in backbone_path.lower():
                    # 加载Qwen-VL模型
                    from transformers import AutoProcessor, AutoModelForVision2Seq
                    
                    processor = AutoProcessor.from_pretrained(backbone_path)
                    model = AutoModelForVision2Seq.from_pretrained(backbone_path)
                    model = model.to(self.device)
                    
                    # 设置到视觉特征提取器
                    self.visual_extractor.set_backbone(model)
                    self.visual_extractor.set_processor(processor)
                    print("Qwen-VL视觉backbone加载成功")
                    
                    
            except Exception as e:
                print(f"加载视觉backbone失败: {e}")
                print("将使用简单的CNN backbone作为替代")
                self._create_simple_backbone()
        else:
            print("未提供backbone路径，创建简单的视觉特征提取器")
            self._create_simple_backbone()
    
    def _create_simple_backbone(self):
        """创建简单的视觉backbone作为替代"""
        import torch.nn as nn
        
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    # 224x224 -> 112x112
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    # 56x56
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    # 28x28
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    # 14x14
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
            def encode_image(self, x):
                """模仿CLIP的encode_image接口"""
                features = self.features(x)
                return features.flatten(1)  # [B, 512]
        
        simple_backbone = SimpleCNN().to(self.device)
        
        # 设置backbone到视觉特征提取器
        if hasattr(self.visual_extractor, 'set_backbone'):
            self.visual_extractor.set_backbone(simple_backbone)
        else:
            # 如果视觉特征提取器没有set_backbone方法，直接替换
            self.visual_extractor.backbone = simple_backbone
        
        print("已创建并设置简单的视觉backbone")
    
    def _init_diffusion(self):
        """初始化扩散组件"""
        diffusion_config = self.config.get('diffusion', {})
        
        self.scheduler = DiffusionScheduler(
            num_timesteps=diffusion_config.get('num_timesteps', 1000),
            beta_start=diffusion_config.get('beta_start', 0.0001),
            beta_end=diffusion_config.get('beta_end', 0.02),
            schedule_type=diffusion_config.get('schedule_type', 'linear')
        )
        
        self.normalizer = LayoutNormalizer(
            canvas_size=tuple(self.config.get('canvas_size', [1024, 1024])),
            max_layer=self.config.get('max_elements', 25)
        )
        
        # 移动调度器常数到设备
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                         'sqrt_recip_alphas', 'posterior_variance']:
            if hasattr(self.scheduler, attr_name):
                setattr(self.scheduler, attr_name, 
                       getattr(self.scheduler, attr_name).to(self.device))
    
    def _load_model(self, model_path: str):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.layout_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.layout_model.load_state_dict(checkpoint)
        
        print(f"DiT模型已加载: {model_path}")
    
    def load_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        加载图片
        
        Args:
            image_paths: 图片路径列表
        
        Returns:
            images: [N, 3, H, W] 图片张量
        """
        images = []
        max_elements = self.config.get('max_elements', 25)
        
        for img_path in image_paths[:max_elements]:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.image_transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"加载图片失败 {img_path}: {e}")
                # 创建空白图片
                blank_img = torch.zeros(3, 224, 224)
                images.append(blank_img)
        
        # 补充到最大元素数量
        while len(images) < max_elements:
            blank_img = torch.zeros(3, 224, 224)
            images.append(blank_img)
        
        return torch.stack(images)  # [N, 3, H, W]
    
    def create_initial_layout(self, num_elements: int) -> torch.Tensor:
        """
        创建初始布局（随机分布）
        
        Args:
            num_elements: 元素数量
        
        Returns:
            initial_layout: [N, 6] 初始布局
        """
        max_elements = self.config.get('max_elements', 25)
        canvas_w, canvas_h = self.config.get('canvas_size', [1024, 1024])
        
        layout = torch.zeros(max_elements, 6)
        
        for i in range(num_elements):
            # 随机位置
            x = np.random.uniform(0, canvas_w * 0.8)
            y = np.random.uniform(0, canvas_h * 0.8)
            
            # 随机尺寸
            w = np.random.uniform(50, 200)
            h = np.random.uniform(30, 150)
            
            # 确保不超出画布
            x = min(x, canvas_w - w)
            y = min(y, canvas_h - h)
            
            # 随机层序
            layer = i
            
            # 置信度
            confidence = 1.0
            
            layout[i] = torch.tensor([x, y, w, h, layer, confidence])
        
        return layout
    
    @torch.no_grad()
    def generate_layout(
        self, 
        image_paths: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        initial_layout: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        生成布局
        
        Args:
            image_paths: 图片路径列表
            num_inference_steps: 推理步数
            guidance_scale: 引导比例
            initial_layout: 可选的初始布局
        
        Returns:
            layout_json: JSON格式的布局描述
        """
        num_elements = len(image_paths)
        max_elements = self.config.get('max_elements', 25)
        
        # 加载图片
        images = self.load_images(image_paths)  # [N, 3, H, W]
        images = images.unsqueeze(0).to(self.device)  # [1, N, 3, H, W]
        
        # 提取视觉特征
        visual_features = self.visual_extractor(images)  # [1, N, visual_dim]
        
        # 创建元素掩码
        element_mask = torch.zeros(1, max_elements, device=self.device)
        element_mask[0, :num_elements] = 1.0
        
        # 初始布局
        if initial_layout is None:
            initial_layout = self.create_initial_layout(num_elements)
        
        current_layout = initial_layout.unsqueeze(0).to(self.device)  # [1, N, 6]
        
        # 标准化
        current_layout = self.normalizer.normalize_layout(current_layout)
        
        # DDPM采样过程
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps
        ).long().to(self.device)
        
        print("正在生成布局...")
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(1)
            
            # 模型预测
            predicted_noise = self.layout_model(
                visual_features=visual_features,
                layout_state=current_layout,
                timestep=t_batch,
                element_mask=element_mask
            )
            
            # 更新布局状态
            if i < len(timesteps) - 1:
                # 计算前一个时间步的布局
                alpha_t = self.scheduler.alphas[t]
                alpha_t_prev = self.scheduler.alphas[t-1] if t > 0 else 1.0
                
                # 预测x_0
                pred_x0 = self.scheduler.predict_start_from_noise(current_layout, t_batch, predicted_noise)
                
                # 计算前一步的均值
                pred_mean = (
                    (alpha_t_prev.sqrt() * pred_x0 + 
                     (1 - alpha_t_prev - (1 - alpha_t) * alpha_t_prev / (1 - alpha_t)).sqrt() * current_layout)
                )
                
                # 添加噪声（除了最后一步）
                if i < len(timesteps) - 1:
                    noise = torch.randn_like(current_layout)
                    variance = self.scheduler.posterior_variance[t]
                    current_layout = pred_mean + variance.sqrt() * noise
                else:
                    current_layout = pred_mean
            else:
                # 最后一步，直接预测
                current_layout = self.scheduler.predict_start_from_noise(
                    current_layout, t_batch, predicted_noise
                )
        
        # 反标准化
        final_layout = self.normalizer.denormalize_layout(current_layout)
        final_layout = final_layout.squeeze(0)  # [N, 6]
        
        # 创建有效元素掩码
        valid_mask = element_mask.squeeze(0)  # [N]
        
        # 转换为JSON格式
        element_ids = [f"element_{i}" for i in range(num_elements)]
        layout_json = layout_to_json(final_layout, valid_mask, element_ids)
        
        print("布局生成完成！")
        return layout_json
    
    def optimize_layout_iteratively(
        self,
        image_paths: List[str],
        num_iterations: int = 3,
        steps_per_iteration: int = 20
    ) -> Dict:
        """
        迭代优化布局
        
        Args:
            image_paths: 图片路径列表
            num_iterations: 迭代次数
            steps_per_iteration: 每次迭代的步数
        
        Returns:
            layout_json: 最终的布局JSON
        """
        print(f"开始迭代优化，共 {num_iterations} 次迭代")
        
        # 初始布局
        current_layout = None
        
        for iteration in range(num_iterations):
            print(f"迭代 {iteration + 1}/{num_iterations}")
            
            # 生成布局
            layout_json = self.generate_layout(
                image_paths=image_paths,
                num_inference_steps=steps_per_iteration,
                initial_layout=current_layout
            )
            
            # 提取布局用于下次迭代
            elements = layout_json['elements']
            max_elements = self.config.get('max_elements', 25)
            next_layout = torch.zeros(max_elements, 6)
            
            for i, element in enumerate(elements):
                if i >= max_elements:
                    break
                bbox = element['bounding_box']
                next_layout[i] = torch.tensor([
                    bbox['x'], bbox['y'], bbox['width'], bbox['height'],
                    element['layer_order'], element['confidence']
                ])
            
            current_layout = next_layout
        
        print("迭代优化完成！")
        return layout_json

    def process_data_action_head_folder(self, data_folder: str, output_suffix: str = "_predicted"):
        """
        批量处理data-action_head文件夹下的所有海报设计
        
        Args:
            data_folder: data-action_head文件夹路径
            output_suffix: 输出文件名后缀
        """
        data_path = Path(data_folder)
        if not data_path.exists():
            print(f"数据文件夹不存在: {data_folder}")
            return
        
        # 获取所有海报文件夹
        poster_folders = [f for f in data_path.iterdir() if f.is_dir()]
        print(f"找到 {len(poster_folders)} 个海报文件夹")
        
        for i, poster_folder in enumerate(poster_folders):
            print(f"\n处理第 {i+1}/{len(poster_folders)} 个文件夹: {poster_folder.name}")
            
            try:
                # 查找parse文件夹中的PNG图片
                parse_folder = poster_folder / "parse"
                if not parse_folder.exists():
                    print(f"  跳过: 未找到parse文件夹")
                    continue
                
                # 获取所有PNG图片
                png_files = list(parse_folder.glob("*.png"))
                if not png_files:
                    print(f"  跳过: 未找到PNG图片")
                    continue
                
                # 按文件名排序（按层序）
                png_files.sort(key=lambda x: x.name)
                image_paths = [str(f) for f in png_files]
                
                print(f"  找到 {len(image_paths)} 个视觉元素")
                
                # 生成布局
                layout_json = self.generate_layout(
                    image_paths=image_paths,
                    num_inference_steps=50
                )
                
                # 保存预测结果
                output_file = poster_folder / f"predicted_layout{output_suffix}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(layout_json, f, indent=2, ensure_ascii=False)
                
                print(f"  预测结果已保存到: {output_file}")
                
            except Exception as e:
                print(f"  处理失败: {e}")
                continue
        
        print(f"\n批量处理完成！")


def main():
    parser = argparse.ArgumentParser(description='Layout Generation Inference')
    parser.add_argument('--model', type=str, required=True, default='/home/LayoutMaster/checkpoints/best_model.pth', help='训练好的DiT模型路径')
    parser.add_argument('--config', type=str, required=True, default='/home/LayoutMaster/config_data_action_head.json', help='配置文件路径')
    parser.add_argument('--backbone', type=str, help='视觉编码器权重路径（可选，如果不提供将使用简单CNN）')
    parser.add_argument('--images', type=str, nargs='+', help='输入图片路径列表（单次推理时使用）')
    parser.add_argument('--output', type=str, default='generated_layout.json', help='输出JSON文件路径（单次推理时使用）')
    parser.add_argument('--steps', type=int, default=50, help='推理步数')
    parser.add_argument('--iterative', action='store_true', help='是否使用迭代优化')
    parser.add_argument('--iterations', type=int, default=3, help='迭代次数')
    
    # 新增批量处理参数
    parser.add_argument('--batch_process', action='store_true', help='批量处理data-action_head文件夹')
    parser.add_argument('--data_folder', type=str, default='data-action_head', help='数据文件夹路径')
    parser.add_argument('--output_suffix', type=str, default='_predicted', help='输出文件名后缀')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = LayoutInference(args.model, args.config, args.backbone)
    
    # 批量处理模式
    if args.batch_process:
        inference.process_data_action_head_folder(
            data_folder=args.data_folder,
            output_suffix=args.output_suffix
        )
    else:
        # 单次推理模式
        if not args.images:
            print("错误: 单次推理模式需要提供 --images 参数")
            return
        
        # 生成布局
        if args.iterative:
            layout_json = inference.optimize_layout_iteratively(
                image_paths=args.images,
                num_iterations=args.iterations
            )
        else:
            layout_json = inference.generate_layout(
                image_paths=args.images,
                num_inference_steps=args.steps
            )
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(layout_json, f, indent=2, ensure_ascii=False)
        
        print(f"布局已保存到: {args.output}")
        print("\n生成的布局:")
        print(json.dumps(layout_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
