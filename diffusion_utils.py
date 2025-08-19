"""
扩散过程工具函数
实现前向扩散（加噪）和反向扩散（去噪）过程
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class DiffusionScheduler:
    """
    扩散调度器，管理噪声调度和采样过程
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # 生成噪声调度
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散过程所需的常数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 反向过程
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """余弦噪声调度"""
        x = torch.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(
        self, 
        x_start: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        前向扩散过程：向原始数据添加噪声
        
        Args:
            x_start: 原始布局状态 [B, N, 6]
            noise: 噪声 [B, N, 6]
            timesteps: 时间步 [B]
        
        Returns:
            noisy_x: 加噪后的布局状态 [B, N, 6]
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """从时间序列中提取对应时间步的值"""
        batch_size = t.shape[0]
        # 保证 t 和 a 在同一设备
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    
    def predict_start_from_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """从噪声预测原始数据"""
        sqrt_recip_alphas_cumprod = self._extract(
            1.0 / self.sqrt_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, t, x_t.shape
        )
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise
    
    def q_posterior_mean_variance(
        self, 
        x_start: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算后验分布的均值和方差"""
        posterior_mean_coef1 = self._extract(
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
            t,
            x_t.shape,
        )
        posterior_mean_coef2 = self._extract(
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod),
            t,
            x_t.shape,
        )
        posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x_t
        
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        
        return posterior_mean, posterior_variance


class LayoutNormalizer:
    """
    布局数据标准化器
    将布局坐标标准化到合适的范围用于扩散过程
    """
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (1024, 1024),
        max_layer: int = 25
    ):
        self.canvas_width, self.canvas_height = canvas_size
        self.max_layer = max_layer
    
    def normalize_layout(self, layout: torch.Tensor) -> torch.Tensor:
        """
        标准化布局数据
        
        Args:
            layout: [B, N, 6] (x, y, w, h, layer, confidence)
        
        Returns:
            normalized_layout: [B, N, 6] 标准化后的布局
        """
        normalized = layout.clone()
        
        # 位置和尺寸标准化到[-1, 1]
        normalized[..., 0] = (layout[..., 0] / self.canvas_width) * 2 - 1    # x
        normalized[..., 1] = (layout[..., 1] / self.canvas_height) * 2 - 1   # y
        normalized[..., 2] = (layout[..., 2] / self.canvas_width) * 2 - 1    # w
        normalized[..., 3] = (layout[..., 3] / self.canvas_height) * 2 - 1   # h
        
        # 层序标准化到[-1, 1]
        normalized[..., 4] = (layout[..., 4] / self.max_layer) * 2 - 1       # layer
        
        # 置信度已经在[0, 1]范围内，映射到[-1, 1]
        normalized[..., 5] = layout[..., 5] * 2 - 1                         # confidence
        
        return normalized
    
    def denormalize_layout(self, normalized_layout: torch.Tensor) -> torch.Tensor:
        """
        反标准化布局数据
        
        Args:
            normalized_layout: [B, N, 6] 标准化的布局
        
        Returns:
            layout: [B, N, 6] 原始尺度的布局
        """
        layout = normalized_layout.clone()
        
        # 位置和尺寸反标准化
        layout[..., 0] = ((normalized_layout[..., 0] + 1) / 2) * self.canvas_width    # x
        layout[..., 1] = ((normalized_layout[..., 1] + 1) / 2) * self.canvas_height   # y
        layout[..., 2] = ((normalized_layout[..., 2] + 1) / 2) * self.canvas_width    # w
        layout[..., 3] = ((normalized_layout[..., 3] + 1) / 2) * self.canvas_height   # h
        
        # 层序反标准化并取整
        layout[..., 4] = torch.round(((normalized_layout[..., 4] + 1) / 2) * self.max_layer)
        
        # 置信度反标准化并限制在[0, 1]
        layout[..., 5] = torch.clamp((normalized_layout[..., 5] + 1) / 2, 0, 1)
        
        return layout
    
    def add_layout_noise(
        self, 
        layout: torch.Tensor, 
        noise_scale: float = 0.1,
        position_noise_scale: float = 0.05,
        layer_noise_scale: float = 0.02
    ) -> torch.Tensor:
        """
        为布局添加结构化噪声
        
        Args:
            layout: [B, N, 6] 原始布局
            noise_scale: 基础噪声尺度
            position_noise_scale: 位置噪声尺度
            layer_noise_scale: 层序噪声尺度
        
        Returns:
            noisy_layout: [B, N, 6] 加噪后的布局
        """
        device = layout.device
        B, N, _ = layout.shape
        
        # 不同维度使用不同的噪声尺度
        noise_scales = torch.tensor([
            position_noise_scale,  # x
            position_noise_scale,  # y
            position_noise_scale,  # w
            position_noise_scale,  # h
            layer_noise_scale,     # layer
            noise_scale           # confidence
        ], device=device)
        
        # 生成噪声
        noise = torch.randn_like(layout) * noise_scales.view(1, 1, -1)
        
        return layout + noise


def create_trajectory_target(
    start_layout: torch.Tensor,
    target_layout: torch.Tensor,
    num_steps: int = 50
) -> torch.Tensor:
    """
    创建从起始布局到目标布局的轨迹
    
    Args:
        start_layout: [B, N, 6] 起始布局
        target_layout: [B, N, 6] 目标布局
        num_steps: 轨迹步数
    
    Returns:
        trajectory: [B, num_steps, N, 6] 完整轨迹
    """
    B, N, layout_dim = start_layout.shape
    
    # 创建时间插值
    t = torch.linspace(0, 1, num_steps, device=start_layout.device)
    t = t.view(1, num_steps, 1, 1).expand(B, num_steps, N, layout_dim)
    
    # 线性插值创建轨迹
    start_expanded = start_layout.unsqueeze(1).expand(-1, num_steps, -1, -1)
    target_expanded = target_layout.unsqueeze(1).expand(-1, num_steps, -1, -1)
    
    trajectory = start_expanded + t * (target_expanded - start_expanded)
    
    return trajectory


def calculate_layout_metrics(pred_layout: torch.Tensor, target_layout: torch.Tensor) -> dict:
    """
    计算布局预测指标
    
    Args:
        pred_layout: [B, N, 6] 预测布局
        target_layout: [B, N, 6] 目标布局
    
    Returns:
        metrics: 包含各种指标的字典
    """
    # 位置误差 (x, y, w, h)
    position_error = F.mse_loss(pred_layout[..., :4], target_layout[..., :4])
    
    # 层序误差
    layer_error = F.mse_loss(pred_layout[..., 4], target_layout[..., 4])
    
    # 置信度误差
    confidence_error = F.mse_loss(pred_layout[..., 5], target_layout[..., 5])
    
    # IoU计算（针对bounding box）
    def calculate_iou(boxes1, boxes2):
        # boxes: [B, N, 4] (x, y, w, h)
        # 转换为 (x1, y1, x2, y2)
        boxes1_xyxy = torch.stack([
            boxes1[..., 0], boxes1[..., 1],  # x1, y1
            boxes1[..., 0] + boxes1[..., 2], boxes1[..., 1] + boxes1[..., 3]  # x2, y2
        ], dim=-1)
        
        boxes2_xyxy = torch.stack([
            boxes2[..., 0], boxes2[..., 1],  # x1, y1
            boxes2[..., 0] + boxes2[..., 2], boxes2[..., 1] + boxes2[..., 3]  # x2, y2
        ], dim=-1)
        
        # 计算交集
        inter_x1 = torch.max(boxes1_xyxy[..., 0], boxes2_xyxy[..., 0])
        inter_y1 = torch.max(boxes1_xyxy[..., 1], boxes2_xyxy[..., 1])
        inter_x2 = torch.min(boxes1_xyxy[..., 2], boxes2_xyxy[..., 2])
        inter_y2 = torch.min(boxes1_xyxy[..., 3], boxes2_xyxy[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        area1 = boxes1[..., 2] * boxes1[..., 3]
        area2 = boxes2[..., 2] * boxes2[..., 3]
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-8)
        return iou.mean()
    
    iou = calculate_iou(pred_layout[..., :4], target_layout[..., :4])
    
    return {
        'position_error': position_error.item(),
        'layer_error': layer_error.item(),
        'confidence_error': confidence_error.item(),
        'iou': iou.item(),
        'total_error': (position_error + layer_error + confidence_error).item()
    }


if __name__ == "__main__":
    # 测试扩散调度器
    scheduler = DiffusionScheduler(num_timesteps=100)
    normalizer = LayoutNormalizer()
    
    # 测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 2, 5
    
    # 创建测试布局
    target_layout = torch.tensor([
        [[100, 200, 50, 30, 3, 0.9],
         [200, 300, 80, 40, 5, 0.8],
         [50, 100, 60, 35, 1, 0.95],
         [300, 150, 70, 45, 7, 0.85],
         [150, 250, 55, 25, 2, 0.92]],
        [[120, 180, 45, 35, 4, 0.88],
         [250, 320, 75, 50, 6, 0.83],
         [80, 120, 65, 30, 2, 0.97],
         [320, 180, 60, 40, 8, 0.86],
         [180, 280, 50, 20, 3, 0.91]]
    ], dtype=torch.float32, device=device)
    
    # 标准化
    normalized_layout = normalizer.normalize_layout(target_layout)
    print(f"原始布局形状: {target_layout.shape}")
    print(f"标准化布局范围: [{normalized_layout.min():.3f}, {normalized_layout.max():.3f}]")
    
    # 扩散过程测试
    timesteps = scheduler.sample_timesteps(B, device)
    noise = torch.randn_like(normalized_layout)
    noisy_layout = scheduler.add_noise(normalized_layout, noise, timesteps)
    
    print(f"加噪后布局形状: {noisy_layout.shape}")
    print(f"时间步: {timesteps}")
    
    # 反标准化测试
    recovered_layout = normalizer.denormalize_layout(normalized_layout)
    print(f"恢复布局与原始布局误差: {F.mse_loss(recovered_layout, target_layout):.6f}")
    
    print("扩散工具测试通过！")
