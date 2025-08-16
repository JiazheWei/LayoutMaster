"""
基于轨迹优化的布局训练脚本
使用扩散模型训练多层海报布局生成
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import swanlab
from typing import Dict, Optional
import numpy as np
import random
import datetime

# 导入自定义模块
from models import create_layout_model, LayoutDiffusionTransformer, VisualFeatureExtractor
from diffusion_utils import DiffusionScheduler, LayoutNormalizer, calculate_layout_metrics
from loss_functions import create_loss_function
from data_utils import create_dataloader, layout_to_json, TrajectoryGenerator
import pickle
from PIL import Image
import torchvision.transforms as transforms

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    轨迹数据集类 - 读取data-action_head格式的数据
    每个文件夹包含：json/, parse/, trajectories.pkl
    """
    
    def __init__(
        self, 
        data_root: str, 
        split: str = 'train',
        image_size: int = 224,
        train_ratio: float = 0.8
    ):
        """
        Args:
            data_root: data-action_head目录路径
            split: 'train' or 'val'
            image_size: 图片尺寸
            train_ratio: 训练集比例
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.split = split
        
        # 获取所有海报目录
        all_poster_dirs = [d for d in self.data_root.iterdir() 
                          if d.is_dir() and (d / 'trajectories.pkl').exists()]
        
        # 划分训练集和验证集
        num_train = int(len(all_poster_dirs) * train_ratio)
        if split == 'train':
            self.poster_dirs = all_poster_dirs[:num_train]
        else:
            self.poster_dirs = all_poster_dirs[num_train:]
        
        # 图片变换
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"{split} 数据集包含 {len(self.poster_dirs)} 个样本")
    
    def __len__(self):
        return len(self.poster_dirs)
    
    def __getitem__(self, idx):
        poster_dir = self.poster_dirs[idx]
        
        try:
            # 加载轨迹数据
            trajectory_file = poster_dir / 'trajectories.pkl'
            with open(trajectory_file, 'rb') as f:
                trajectories = pickle.load(f)
            
            # 随机选择一个噪声水平
            noise_levels = list(trajectories.keys())
            noise_key = random.choice(noise_levels)
            traj_data = trajectories[noise_key]
            
            # 提取轨迹信息
            start_layout = traj_data['start_layout']  # [25, 6]
            target_layout = traj_data['target_layout']  # [25, 6]
            element_mask = traj_data['element_mask']  # [25]
            
            # 加载ground truth JSON获取图片路径
            json_dir = poster_dir / 'json'
            json_files = list(json_dir.glob('*.json'))
            
            if json_files:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 加载图片
                images = self._load_images(poster_dir, json_data, element_mask)
            else:
                # 如果没有JSON文件，创建空白图片
                images = self._create_blank_images(element_mask)
            
            return {
                'images': images,              # [25, 3, H, W]
                'start_layout': start_layout,  # [25, 6]
                'target_layout': target_layout, # [25, 6]
                'element_mask': element_mask,  # [25]
                'num_elements': element_mask.sum().long(),
                'poster_name': poster_dir.name
            }
            
        except Exception as e:
            logger.warning(f"加载数据失败 {poster_dir}: {e}")
            # 返回空数据
            return self._create_empty_sample()
    
    def _load_images(self, poster_dir, json_data, element_mask):
        """加载图片"""
        images = []
        parse_dir = poster_dir / 'parse'
        
        # 获取有效元素数量
        num_valid = int(element_mask.sum())
        layers = json_data.get('layers', [])[:num_valid]
        
        for i, layer in enumerate(layers):
            try:
                # 尝试从parse目录找到对应的PNG文件
                src_name = layer.get('src', '')
                
                # 尝试不同的文件名模式
                possible_names = [
                    src_name,
                    src_name.replace(' ', '_'),
                    f"{i}.png",
                    f"layer_{i}.png"
                ]
                
                img_loaded = False
                for name in possible_names:
                    img_path = parse_dir / name
                    if img_path.exists():
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = self.image_transform(img)
                        images.append(img_tensor)
                        img_loaded = True
                        break
                
                if not img_loaded:
                    # 如果找不到图片，创建空白图片
                    blank_img = torch.zeros(3, self.image_size, self.image_size)
                    images.append(blank_img)
                    
            except Exception as e:
                # 创建空白图片作为占位符
                blank_img = torch.zeros(3, self.image_size, self.image_size)
                images.append(blank_img)
        
        # 补充到25个元素
        while len(images) < 25:
            blank_img = torch.zeros(3, self.image_size, self.image_size)
            images.append(blank_img)
        
        # 截断到25个
        images = images[:25]
        
        return torch.stack(images)
    
    def _create_blank_images(self, element_mask):
        """创建空白图片"""
        images = []
        for i in range(25):
            blank_img = torch.zeros(3, self.image_size, self.image_size)
            images.append(blank_img)
        return torch.stack(images)
    
    def _create_empty_sample(self):
        """创建空样本"""
        return {
            'images': torch.zeros(25, 3, self.image_size, self.image_size),
            'start_layout': torch.zeros(25, 6),
            'target_layout': torch.zeros(25, 6),
            'element_mask': torch.zeros(25),
            'num_elements': torch.tensor(0),
            'poster_name': 'empty'
        }


def collate_fn_trajectory(batch):
    """轨迹数据集的collate函数"""
    # 过滤掉空样本
    valid_batch = [sample for sample in batch if sample['num_elements'] > 0]
    
    if not valid_batch:
        # 如果没有有效样本，返回空批次
        return {
            'images': torch.zeros(1, 25, 3, 224, 224),
            'start_layouts': torch.zeros(1, 25, 6),
            'target_layouts': torch.zeros(1, 25, 6),
            'element_masks': torch.zeros(1, 25),
            'num_elements': torch.zeros(1),
            'poster_names': ['empty']
        }
    
    batch = valid_batch
    
    # 提取各个字段
    images = torch.stack([sample['images'] for sample in batch])
    start_layouts = torch.stack([sample['start_layout'] for sample in batch])
    target_layouts = torch.stack([sample['target_layout'] for sample in batch])
    element_masks = torch.stack([sample['element_mask'] for sample in batch])
    num_elements = torch.stack([sample['num_elements'] for sample in batch])
    poster_names = [sample['poster_name'] for sample in batch]
    
    return {
        'images': images,               # [B, 25, 3, H, W]
        'start_layouts': start_layouts, # [B, 25, 6]
        'target_layouts': target_layouts, # [B, 25, 6]
        'element_masks': element_masks, # [B, 25]
        'num_elements': num_elements,   # [B]
        'poster_names': poster_names    # [B]
    }


class LayoutTrainer:
    """
    布局轨迹优化训练器
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 设置随机种子
        self._set_seed(config.get('seed', 42))
        
        # 初始化组件
        self._init_models()
        self._init_diffusion()
        self._init_optimizer()
        self._init_data()
        self._init_loss()
        self._init_logging()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.is_multi_gpu = False  # 初始化多GPU标志
        
        logger.info(f"训练器初始化完成，使用设备: {self.device}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _init_models(self):
        """初始化模型"""
        config = self.config['model']
        
        # 创建布局模型和视觉特征提取器
        self.layout_model, self.visual_extractor = create_layout_model(
            visual_backbone=config.get('visual_backbone', 'clip'),
            visual_dim=config.get('visual_dim', 768),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 12),
            max_elements=config.get('max_elements', 25)
        )
        
        # 移动到设备
        self.layout_model = self.layout_model.to(self.device)
        self.visual_extractor = self.visual_extractor.to(self.device)
        
        # 多GPU支持
        if torch.cuda.device_count() > 1 and self.config.get('use_multi_gpu', True):
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.layout_model = torch.nn.DataParallel(self.layout_model)
            self.visual_extractor = torch.nn.DataParallel(self.visual_extractor)
            self.is_multi_gpu = True
        else:
            self.is_multi_gpu = False
            logger.info(f"使用单GPU训练: {self.device}")
        
        # 加载预训练的CLIP模型（需要用户提供路径）
        self._load_visual_backbone()
        
        # 计算参数数量
        layout_params = sum(p.numel() for p in self.layout_model.parameters())
        visual_params = sum(p.numel() for p in self.visual_extractor.parameters())
        logger.info(f"布局模型参数数量: {layout_params:,}")
        logger.info(f"视觉模型参数数量: {visual_params:,}")
        logger.info(f"总参数数量: {layout_params + visual_params:,}")
    
    def _load_visual_backbone(self):
        """加载视觉backbone"""
        backbone_path = self.config.get('backbone_path')
        
        if backbone_path and os.path.exists(backbone_path):
            try:
                # 这里需要根据实际的backbone类型来加载
                # 暂时使用占位符，用户需要提供具体的加载方法
                logger.info(f"需要加载视觉backbone: {backbone_path}")
                logger.info("请确保提供正确的CLIP或其他视觉模型路径")
            except Exception as e:
                logger.warning(f"加载backbone失败: {e}")
        else:
            logger.warning("未提供backbone路径，创建简单的视觉特征提取器")
            
            # 创建一个简单的CNN backbone作为替代
            simple_backbone = self._create_simple_backbone()
            
            # 设置backbone到视觉特征提取器
            if hasattr(self.visual_extractor, 'module'):
                # 多GPU情况
                self.visual_extractor.module.set_backbone(simple_backbone)
            else:
                # 单GPU情况
                self.visual_extractor.set_backbone(simple_backbone)
            
            logger.info("已创建并设置简单的视觉backbone")
    
    def _create_simple_backbone(self):
        """创建简单的视觉backbone作为CLIP的替代"""
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
        
        return SimpleCNN().to(self.device)
    
    def _init_diffusion(self):
        """初始化扩散调度器"""
        diffusion_config = self.config.get('diffusion', {})
        
        self.scheduler = DiffusionScheduler(
            num_timesteps=diffusion_config.get('num_timesteps', 1000),
            beta_start=diffusion_config.get('beta_start', 0.0001),
            beta_end=diffusion_config.get('beta_end', 0.02),
            schedule_type=diffusion_config.get('schedule_type', 'linear')
        )
        
        self.normalizer = LayoutNormalizer(
            canvas_size=self.config.get('canvas_size', (1024, 1024)),
            max_layer=self.config.get('max_elements', 25)
        )
        
        # 将调度器常数移动到设备
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                         'sqrt_recip_alphas', 'posterior_variance']:
            if hasattr(self.scheduler, attr_name):
                setattr(self.scheduler, attr_name, 
                       getattr(self.scheduler, attr_name).to(self.device))
    
    def _init_optimizer(self):
        """初始化优化器"""
        optim_config = self.config.get('optimizer', {})
        
        # 只优化layout_model的参数
        params = list(self.layout_model.parameters())
        
        # 如果不冻结视觉特征提取器
        if not optim_config.get('freeze_visual', True):
            params.extend(list(self.visual_extractor.parameters()))
        
        self.optimizer = optim.AdamW(
            params,
            lr=optim_config.get('lr', 1e-4),
            weight_decay=optim_config.get('weight_decay', 0.01),
            betas=optim_config.get('betas', (0.9, 0.999))
        )
        
        # 学习率调度器
        self.scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('training', {}).get('num_epochs', 100),
            eta_min=optim_config.get('min_lr', 1e-6)
        )
    
    def _init_data(self):
        """初始化数据加载器"""
        data_config = self.config.get('data', {})
        
        # 检查是否使用轨迹数据集
        if data_config.get('use_trajectory_dataset', False):
            # 使用轨迹数据集 (data-action_head格式)
            train_dataset = TrajectoryDataset(
                data_root=data_config['data_root'],
                split='train',
                image_size=data_config.get('image_size', 224),
                train_ratio=data_config.get('train_ratio', 0.8)
            )
            
            val_dataset = TrajectoryDataset(
                data_root=data_config['data_root'],
                split='val',
                image_size=data_config.get('image_size', 224),
                train_ratio=data_config.get('train_ratio', 0.8)
            )
            
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=data_config.get('batch_size', 4),
                shuffle=True,
                num_workers=data_config.get('num_workers', 4),
                collate_fn=collate_fn_trajectory,
                pin_memory=True,
                drop_last=True
            )
            
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=data_config.get('batch_size', 4),
                shuffle=False,
                num_workers=data_config.get('num_workers', 4),
                collate_fn=collate_fn_trajectory,
                pin_memory=True,
                drop_last=False
            ) if len(val_dataset) > 0 else None
            
        else:
            # 使用原始数据加载器
            self.train_loader = create_dataloader(
                data_file=data_config['train_file'],
                image_root=data_config['image_root'],
                batch_size=data_config.get('batch_size', 4),
                num_workers=data_config.get('num_workers', 4),
                shuffle=True,
                canvas_size=self.config.get('canvas_size', (1024, 1024)),
                max_elements=self.config.get('max_elements', 25),
                augment=data_config.get('augment', True)
            )
            
            # 验证数据
            if data_config.get('val_file'):
                self.val_loader = create_dataloader(
                    data_file=data_config['val_file'],
                    image_root=data_config['image_root'],
                    batch_size=data_config.get('batch_size', 4),
                    num_workers=data_config.get('num_workers', 4),
                    shuffle=False,
                    canvas_size=self.config.get('canvas_size', (1024, 1024)),
                    max_elements=self.config.get('max_elements', 25),
                    augment=False
                )
            else:
                self.val_loader = None
        
        # 轨迹生成器
        self.trajectory_gen = TrajectoryGenerator(
            num_steps=self.config.get('diffusion', {}).get('trajectory_steps', 50)
        )
    
    def _init_loss(self):
        """初始化损失函数"""
        loss_config = self.config.get('loss', {})
        self.loss_fn = create_loss_function(loss_config)
        self.loss_fn = self.loss_fn.to(self.device)
    
    def _init_logging(self):
        """初始化日志记录"""
        if self.config.get('use_swanlab', False):
            swanlab.init(
                project=self.config.get('project_name', 'layout-diffusion'),
                config=self.config,
                experiment_name=self.config.get('run_name')
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行一个训练步骤"""
        self.layout_model.train()
        self.visual_extractor.eval()  # 保持视觉特征提取器为评估模式
        
        # 移动数据到设备
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        # 提取数据
        images = batch['images']                # [B, N, 3, H, W]
        start_layouts = batch['start_layouts'] # [B, N, 6]
        target_layouts = batch['target_layouts'] # [B, N, 6]
        element_masks = batch['element_masks'] # [B, N]
        
        B, N = start_layouts.shape[:2]
        
        # 提取视觉特征
        visual_features = self.visual_extractor(images)  # [B, N, visual_dim]
        
        # 标准化布局
        normalized_start = self.normalizer.normalize_layout(start_layouts)
        normalized_target = self.normalizer.normalize_layout(target_layouts)
        
        # 随机采样时间步
        timesteps = self.scheduler.sample_timesteps(B, self.device)
        
        # 生成噪声
        noise = torch.randn_like(normalized_target)
        
        # 前向扩散过程
        noisy_layout = self.scheduler.add_noise(normalized_target, noise, timesteps)
        
        # 模型预测
        predicted_noise = self.layout_model(
            visual_features=visual_features,
            layout_state=noisy_layout,
            timestep=timesteps,
            element_mask=element_masks
        )
        
        # 预测原始布局
        predicted_layout = self.scheduler.predict_start_from_noise(
            noisy_layout, timesteps, predicted_noise
        )
        
        # 反标准化用于损失计算
        pred_layout_denorm = self.normalizer.denormalize_layout(predicted_layout)
        
        # 计算损失
        total_loss, loss_dict = self.loss_fn(
            predicted_noise=predicted_noise,
            target_noise=noise,
            predicted_layout=pred_layout_denorm,
            target_layout=target_layouts,
            element_mask=element_masks
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.layout_model.parameters(), 
            self.config.get('training', {}).get('max_grad_norm', 1.0)
        )
        
        self.optimizer.step()
        
        # 更新全局步数
        self.global_step += 1
        
        return loss_dict
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_loader is None:
            return {}
        
        self.layout_model.eval()
        total_losses = {}
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="验证中"):
            # 移动数据到设备
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            
            # 提取数据
            images = batch['images']
            start_layouts = batch['start_layouts']
            target_layouts = batch['target_layouts']
            element_masks = batch['element_masks']
            
            B, N = start_layouts.shape[:2]
            
            # 提取视觉特征
            visual_features = self.visual_extractor(images)
            
            # 标准化布局
            normalized_target = self.normalizer.normalize_layout(target_layouts)
            
            # 随机采样时间步
            timesteps = self.scheduler.sample_timesteps(B, self.device)
            
            # 生成噪声
            noise = torch.randn_like(normalized_target)
            
            # 前向扩散过程
            noisy_layout = self.scheduler.add_noise(normalized_target, noise, timesteps)
            
            # 模型预测
            predicted_noise = self.layout_model(
                visual_features=visual_features,
                layout_state=noisy_layout,
                timestep=timesteps,
                element_mask=element_masks
            )
            
            # 预测原始布局
            predicted_layout = self.scheduler.predict_start_from_noise(
                noisy_layout, timesteps, predicted_noise
            )
            
            # 反标准化
            pred_layout_denorm = self.normalizer.denormalize_layout(predicted_layout)
            
            # 计算损失
            _, loss_dict = self.loss_fn(
                predicted_noise=predicted_noise,
                target_noise=noise,
                predicted_layout=pred_layout_denorm,
                target_layout=target_layouts,
                element_mask=element_masks
            )
            
            # 累积损失
            for key, value in loss_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value
            
            num_batches += 1
        
        # 计算平均损失
        avg_losses = {f"val_{key}": value / num_batches for key, value in total_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def sample_layout(
        self, 
        visual_features: torch.Tensor, 
        initial_layout: torch.Tensor,
        element_mask: torch.Tensor,
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """采样生成布局"""
        self.layout_model.eval()
        
        B, N, _ = initial_layout.shape
        
        # 标准化初始布局
        current_layout = self.normalizer.normalize_layout(initial_layout)
        
        # DDPM采样过程
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps
        ).long().to(self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            
            # 模型预测
            predicted_noise = self.layout_model(
                visual_features=visual_features,
                layout_state=current_layout,
                timestep=t_batch,
                element_mask=element_mask
            )
            
            # 更新布局
            if i < len(timesteps) - 1:
                # 不是最后一步，添加噪声
                noise = torch.randn_like(current_layout)
                current_layout = self.scheduler.add_noise(current_layout, noise, t_batch)
            else:
                # 最后一步，直接预测
                current_layout = self.scheduler.predict_start_from_noise(
                    current_layout, t_batch, predicted_noise
                )
        
        # 反标准化
        final_layout = self.normalizer.denormalize_layout(current_layout)
        
        return final_layout
    
    def save_checkpoint(self, save_path: str):
        """保存检查点"""
        # 处理多GPU模型保存
        if self.is_multi_gpu:
            model_state_dict = self.layout_model.module.state_dict()
        else:
            model_state_dict = self.layout_model.state_dict()
            
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler_lr.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'is_multi_gpu': self.is_multi_gpu
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"检查点已保存到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 处理多GPU模型加载
        if self.is_multi_gpu:
            self.layout_model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容性处理：如果检查点来自多GPU模型但当前是单GPU
            try:
                self.layout_model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                # 尝试移除module前缀
                if 'module.' in str(e):
                    state_dict = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        if key.startswith('module.'):
                            state_dict[key[7:]] = value  # 移除'module.'前缀
                        else:
                            state_dict[key] = value
                    self.layout_model.load_state_dict(state_dict)
                else:
                    raise e
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler_lr.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"检查点已加载: {checkpoint_path}")
    
    def train(self):
        """主训练循环"""
        num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        save_dir = Path(self.config.get('training', {}).get('save_dir', './checkpoints'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始训练，共 {num_epochs} 个epoch")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练一个epoch
            epoch_losses = {}
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 训练步骤
                loss_dict = self.train_step(batch)
                
                # 累积损失
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value
                
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'step': self.global_step
                })
                
                # 记录到swanlab
                if self.config.get('use_swanlab', False) and batch_idx % 10 == 0:
                    swanlab.log(loss_dict, step=self.global_step)
            
            # 计算epoch平均损失
            avg_epoch_losses = {key: value / num_batches for key, value in epoch_losses.items()}
            
            # 验证
            val_losses = self.validate()
            
            # 学习率调度
            self.scheduler_lr.step()
            
            # 日志记录
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_epoch_losses['total_loss']:.4f}")
            if val_losses:
                logger.info(f"Epoch {epoch+1} - Val Loss: {val_losses.get('val_total_loss', 0):.4f}")
            
            # 保存最佳模型
            current_loss = val_losses.get('val_total_loss', avg_epoch_losses['total_loss'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(save_dir / 'best_model.pth')
            
            # 定期保存检查点
            if (epoch + 1) % self.config.get('training', {}).get('save_every', 10) == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch+1}.pth')
            
            # swanlab记录
            if self.config.get('use_swanlab', False):
                log_dict = {**avg_epoch_losses, **val_losses, 'epoch': epoch}
                swanlab.log(log_dict, step=self.global_step)
        
        logger.info("训练完成！")
    
    @torch.no_grad()
    def predict_and_save(self, output_dir: Optional[str] = None, num_samples: int = 10):
        """
        预测并保存JSON结果到data-action_head目录
        
        Args:
            output_dir: 输出目录，如果为None则保存到原始data-action_head目录
            num_samples: 要预测的样本数量
        """
        logger.info(f"开始预测并保存结果，样本数量: {num_samples}")
        
        self.layout_model.eval()
        
        # 获取数据配置
        data_config = self.config.get('data', {})
        
        if data_config.get('use_trajectory_dataset', False):
            # 创建预测用的数据集
            predict_dataset = TrajectoryDataset(
                data_root=data_config['data_root'],
                split='val',
                image_size=data_config.get('image_size', 224),
                train_ratio=data_config.get('train_ratio', 0.8)
            )
            
            predict_loader = torch.utils.data.DataLoader(
                predict_dataset,
                batch_size=1,  # 一次处理一个样本
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn_trajectory
            )
            
            predictions_saved = 0
            
            for batch_idx, batch in enumerate(predict_loader):
                if predictions_saved >= num_samples:
                    break
                
                # 移动数据到设备
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # 提取数据
                images = batch['images']                 # [1, 25, 3, H, W]  
                start_layouts = batch['start_layouts']   # [1, 25, 6]
                element_masks = batch['element_masks']   # [1, 25]
                poster_names = batch['poster_names']     # [1]
                
                # 提取视觉特征
                visual_features = self.visual_extractor(images)  # [1, 25, visual_dim]
                
                # 采样生成预测布局
                predicted_layout = self.sample_layout(
                    visual_features=visual_features,
                    initial_layout=start_layouts,
                    element_mask=element_masks,
                    num_inference_steps=self.config.get('inference', {}).get('num_steps', 50)
                )
                
                # 保存预测结果
                self._save_prediction_json(
                    predicted_layout=predicted_layout[0],  # 取第一个批次 [25, 6]
                    element_mask=element_masks[0],         # [25]
                    poster_name=poster_names[0],
                    output_dir=output_dir or data_config['data_root']
                )
                
                predictions_saved += 1
                
                if predictions_saved % 5 == 0:
                    logger.info(f"已预测并保存 {predictions_saved} 个样本")
        
        logger.info(f"预测完成，共保存了 {predictions_saved} 个结果")
    
    def _save_prediction_json(self, predicted_layout, element_mask, poster_name, output_dir):
        """保存单个预测结果为JSON"""
        try:
            # 转换为JSON格式
            result_json = layout_to_json(
                layout=predicted_layout,
                element_mask=element_mask,
                element_ids=[f"element_{i}" for i in range(element_mask.sum().int())]
            )
            
            # 添加预测信息
            result_json['prediction_info'] = {
                'model': 'LayoutMaster',
                'poster_name': poster_name,
                'prediction_time': datetime.datetime.now().isoformat(),
                'total_elements': int(element_mask.sum())
            }
            
            # 保存到对应的海报目录
            poster_dir = Path(output_dir) / poster_name
            if not poster_dir.exists():
                logger.warning(f"海报目录不存在: {poster_dir}")
                return
            
            prediction_file = poster_dir / 'prediction.json'
            
            with open(prediction_file, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"预测结果已保存到: {prediction_file}")
            
        except Exception as e:
            logger.error(f"保存预测结果失败 {poster_name}: {e}")
    
    def inference_from_images(self, poster_dir: str, num_inference_steps: int = 50):
        """
        从指定海报目录推理生成布局
        
        Args:
            poster_dir: 海报目录路径
            num_inference_steps: 推理步数
            
        Returns:
            predicted_layout: 预测的布局
        """
        self.layout_model.eval()
        
        poster_path = Path(poster_dir)
        
        # 加载轨迹数据获取初始状态
        trajectory_file = poster_path / 'trajectories.pkl'
        if not trajectory_file.exists():
            logger.error(f"轨迹文件不存在: {trajectory_file}")
            return None
        
        with open(trajectory_file, 'rb') as f:
            trajectories = pickle.load(f)
        
        # 使用第一个噪声水平的数据
        first_key = list(trajectories.keys())[0]
        traj_data = trajectories[first_key]
        
        start_layout = traj_data['start_layout'].unsqueeze(0)    # [1, 25, 6]
        element_mask = traj_data['element_mask'].unsqueeze(0)    # [1, 25]
        
        # 加载图片
        json_dir = poster_path / 'json'
        json_files = list(json_dir.glob('*.json'))
        
        if json_files:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 使用TrajectoryDataset的图片加载方法
            temp_dataset = TrajectoryDataset(str(poster_path.parent), split='train')
            images = temp_dataset._load_images(poster_path, json_data, element_mask[0]).unsqueeze(0)
        else:
            images = torch.zeros(1, 25, 3, 224, 224)
        
        # 移动到设备
        images = images.to(self.device)
        start_layout = start_layout.to(self.device)
        element_mask = element_mask.to(self.device)
        
        # 提取视觉特征
        visual_features = self.visual_extractor(images)
        
        # 生成预测
        with torch.no_grad():
            predicted_layout = self.sample_layout(
                visual_features=visual_features,
                initial_layout=start_layout,
                element_mask=element_mask,
                num_inference_steps=num_inference_steps
            )
        
        return predicted_layout[0].cpu()  # 返回CPU上的预测结果


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Layout Diffusion Training and Inference')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', 
                       help='运行模式：train（训练）或predict（预测）')
    parser.add_argument('--checkpoint', type=str, help='预测时使用的检查点路径')
    parser.add_argument('--num_samples', type=int, default=10, help='预测时要处理的样本数量')
    parser.add_argument('--output_dir', type=str, help='预测结果输出目录')
    parser.add_argument('--poster_dir', type=str, help='单个海报目录路径（用于单个预测）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器
    trainer = LayoutTrainer(config)
    
    if args.mode == 'train':
        # 训练模式
        logger.info("开始训练模式")
        
        # 恢复训练
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        trainer.train()
        
    elif args.mode == 'predict':
        # 预测模式
        logger.info("开始预测模式")
        
        # 加载检查点
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        else:
            logger.error("预测模式需要指定--checkpoint参数")
            return
        
        if args.poster_dir:
            # 单个海报预测
            logger.info(f"对单个海报进行预测: {args.poster_dir}")
            predicted_layout = trainer.inference_from_images(args.poster_dir)
            
            if predicted_layout is not None:
                # 保存单个预测结果
                poster_path = Path(args.poster_dir)
                trajectory_file = poster_path / 'trajectories.pkl'
                with open(trajectory_file, 'rb') as f:
                    trajectories = pickle.load(f)
                first_key = list(trajectories.keys())[0]
                element_mask = trajectories[first_key]['element_mask']
                
                trainer._save_prediction_json(
                    predicted_layout=predicted_layout,
                    element_mask=element_mask,
                    poster_name=poster_path.name,
                    output_dir=str(poster_path.parent)
                )
                logger.info(f"预测结果已保存到: {poster_path / 'prediction.json'}")
            
        else:
            # 批量预测
            logger.info(f"批量预测模式，样本数量: {args.num_samples}")
            trainer.predict_and_save(
                output_dir=args.output_dir,
                num_samples=args.num_samples
            )


if __name__ == "__main__":
    main()
