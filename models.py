"""
基于DiT (Diffusion Transformer)的多层布局轨迹优化模型
结合视觉特征和位置信息，预测元素的移动轨迹
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttentionBlock(nn.Module):
    """交叉注意力块，处理视觉特征和布局特征的交互"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, layout_features: torch.Tensor, 
                visual_features: torch.Tensor) -> torch.Tensor:
        # 自注意力
        normed = self.norm1(layout_features)
        attn_out, _ = self.self_attn(normed, normed, normed)
        layout_features = layout_features + attn_out
        
        # 交叉注意力
        normed = self.norm2(layout_features)
        cross_out, _ = self.cross_attn(normed, visual_features, visual_features)
        layout_features = layout_features + cross_out
        
        # FFN
        normed = self.norm3(layout_features)
        ffn_out = self.ffn(normed)
        layout_features = layout_features + ffn_out
        
        return layout_features


class LayoutDiffusionTransformer(nn.Module):
    """
    基于Diffusion Transformer的布局优化模型
    
    输入：
    - visual_features: 视觉元素特征 [B, N, visual_dim]
    - layout_state: 当前布局状态 [B, N, 6] (x, y, w, h, layer, confidence)
    - timestep: 扩散时间步 [B]
    
    输出：
    - layout_deltas: 布局调整量 [B, N, 6]
    """
    
    def __init__(
        self,
        visual_dim: int = 768,  # 视觉特征维度 (如CLIP特征)
        layout_dim: int = 6,    # 布局状态维度 (x,y,w,h,layer,conf)
        hidden_dim: int = 512,  # 隐藏层维度
        num_layers: int = 12,   # Transformer层数
        num_heads: int = 8,     # 注意力头数
        max_elements: int = 25, # 最大元素数量
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.layout_dim = layout_dim
        self.hidden_dim = hidden_dim
        self.max_elements = max_elements
        
        # 时间步嵌入
        self.time_embedding = SinusoidalPositionEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 视觉特征投影
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        
        # 布局状态编码
        self.layout_projection = nn.Linear(layout_dim, hidden_dim)
        
        # 位置编码（用于元素序列）
        self.position_embedding = nn.Parameter(
            torch.randn(max_elements, hidden_dim)
        )
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.action_head = ActionHead(hidden_dim, layout_dim)
        
    def forward(
        self,
        visual_features: torch.Tensor,    # [B, N, visual_dim]
        layout_state: torch.Tensor,      # [B, N, layout_dim]
        timestep: torch.Tensor,          # [B]
        element_mask: Optional[torch.Tensor] = None  # [B, N]
    ) -> torch.Tensor:
        
        B, N, _ = layout_state.shape
        
        # 时间步嵌入
        time_emb = self.time_embedding(timestep)  # [B, hidden_dim]
        time_emb = self.time_mlp(time_emb)        # [B, hidden_dim]
        
        # 视觉特征投影
        visual_feat = self.visual_projection(visual_features)  # [B, N, hidden_dim]
        
        # 布局状态编码
        layout_feat = self.layout_projection(layout_state)     # [B, N, hidden_dim]
        
        # 添加位置编码
        pos_emb = self.position_embedding[:N].unsqueeze(0).expand(B, -1, -1)
        layout_feat = layout_feat + pos_emb
        
        # 添加时间信息
        time_emb = time_emb.unsqueeze(1).expand(-1, N, -1)
        layout_feat = layout_feat + time_emb
        
        # 应用元素掩码
        if element_mask is not None:
            mask_expanded = element_mask.unsqueeze(-1).expand_as(layout_feat)
            layout_feat = layout_feat * mask_expanded
            visual_feat = visual_feat * mask_expanded
        
        # Transformer层
        for layer in self.transformer_layers:
            layout_feat = layer(layout_feat, visual_feat)
        
        # 输出预测
        layout_feat = self.output_norm(layout_feat)
        layout_deltas = self.action_head(layout_feat)
        
        # 应用掩码到输出
        if element_mask is not None:
            mask_expanded = element_mask.unsqueeze(-1).expand_as(layout_deltas)
            layout_deltas = layout_deltas * mask_expanded
        
        return layout_deltas


class ActionHead(nn.Module):
    """
    动作预测头，预测布局调整动作
    分别处理连续动作（位置）和离散动作（层序）
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        # 连续动作预测（x, y, w, h调整）
        self.continuous_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 4),  # (dx, dy, dw, dh)
            nn.Tanh()  # 限制调整范围
        )
        
        # 离散动作预测（层序调整）
        self.layer_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 1)  # 层序调整值
        )
        
        # 置信度预测
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 连续动作 (位置调整)
        continuous_actions = self.continuous_head(features)  # [B, N, 4]
        
        # 离散动作 (层序调整)
        layer_actions = self.layer_head(features)           # [B, N, 1]
        
        # 置信度
        confidence = self.confidence_head(features)         # [B, N, 1]
        
        # 合并所有动作
        actions = torch.cat([
            continuous_actions,  # x, y, w, h deltas
            layer_actions,       # layer delta
            confidence          # confidence
        ], dim=-1)  # [B, N, 6]
        
        return actions


class VisualFeatureExtractor(nn.Module):
    """
    视觉特征提取器
    可以基于预训练的CLIP或其他视觉模型
    """
    
    def __init__(
        self,
        backbone_name: str = "clip",
        feature_dim: int = 768,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        
        if backbone_name == "clip":
            # 使用CLIP作为backbone，后续会通过外部加载
            self.backbone = None  # 将通过外部设置
            self.projection = nn.Linear(512, feature_dim)  # CLIP默认512维
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
        
        self.freeze_backbone = freeze_backbone
    
    def set_backbone(self, backbone):
        """设置预训练的backbone模型"""
        self.backbone = backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取视觉特征
        Args:
            images: [B, N, C, H, W] 批次中每个样本的N个图像
        Returns:
            features: [B, N, feature_dim] 视觉特征
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not set. Call set_backbone() first.")
        
        B, N, C, H, W = images.shape
        
        # 重塑为 [B*N, C, H, W] 用于批处理
        images_flat = images.view(B * N, C, H, W)
        
        # 提取特征
        with torch.set_grad_enabled(not self.freeze_backbone):
            if self.backbone_name == "clip":
                features = self.backbone.encode_image(images_flat)
            else:
                features = self.backbone(images_flat)
        
        # 投影到目标维度
        features = self.projection(features)
        
        # 重塑回 [B, N, feature_dim]
        features = features.view(B, N, self.feature_dim)
        
        return features


def create_layout_model(
    visual_backbone: str = "clip",
    visual_dim: int = 768,
    hidden_dim: int = 512,
    num_layers: int = 12,
    max_elements: int = 25
) -> Tuple[LayoutDiffusionTransformer, VisualFeatureExtractor]:
    """
    创建完整的布局优化模型
    
    Returns:
        layout_model: 布局扩散模型
        visual_extractor: 视觉特征提取器
    """
    
    # 创建视觉特征提取器
    visual_extractor = VisualFeatureExtractor(
        backbone_name=visual_backbone,
        feature_dim=visual_dim,
        freeze_backbone=True
    )
    
    # 创建布局模型
    layout_model = LayoutDiffusionTransformer(
        visual_dim=visual_dim,
        layout_dim=6,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_elements=max_elements
    )
    
    return layout_model, visual_extractor


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    layout_model, visual_extractor = create_layout_model()
    layout_model = layout_model.to(device)
    visual_extractor = visual_extractor.to(device)
    
    # 测试数据
    B, N = 2, 5  # 批次大小=2，元素数量=5
    visual_features = torch.randn(B, N, 768).to(device)
    layout_state = torch.randn(B, N, 6).to(device)
    timestep = torch.randint(0, 1000, (B,)).to(device)
    
    # 前向传播
    layout_deltas = layout_model(visual_features, layout_state, timestep)
    
    print(f"输入形状:")
    print(f"  视觉特征: {visual_features.shape}")
    print(f"  布局状态: {layout_state.shape}")
    print(f"  时间步: {timestep.shape}")
    print(f"输出形状:")
    print(f"  布局调整: {layout_deltas.shape}")
    print("模型测试通过！")
