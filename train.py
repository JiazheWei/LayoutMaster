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
import wandb
from typing import Dict, Optional
import numpy as np
import random

# 导入自定义模块
from models import create_layout_model, LayoutDiffusionTransformer, VisualFeatureExtractor
from diffusion_utils import DiffusionScheduler, LayoutNormalizer, calculate_layout_metrics
from loss_functions import create_loss_function
from data_utils import create_dataloader, layout_to_json, TrajectoryGenerator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        # 加载预训练的CLIP模型（需要用户提供路径）
        self._load_visual_backbone()
        
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.layout_model.parameters()):,}")
    
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
            logger.warning("未提供backbone路径，请确保设置正确的视觉模型")
    
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
        
        # 训练数据
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
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('project_name', 'layout-diffusion'),
                config=self.config,
                name=self.config.get('run_name')
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
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.layout_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler_lr.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"检查点已保存到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.layout_model.load_state_dict(checkpoint['model_state_dict'])
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
                
                # 记录到wandb
                if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                    wandb.log(loss_dict, step=self.global_step)
            
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
            
            # wandb记录
            if self.config.get('use_wandb', False):
                log_dict = {**avg_epoch_losses, **val_losses, 'epoch': epoch}
                wandb.log(log_dict, step=self.global_step)
        
        logger.info("训练完成！")


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Layout Diffusion Training')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器
    trainer = LayoutTrainer(config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
