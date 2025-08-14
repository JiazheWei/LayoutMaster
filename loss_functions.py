"""
损失函数定义
包含位置损失、层序损失、美学损失等多种损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class LayoutDiffusionLoss(nn.Module):
    """
    布局扩散训练的复合损失函数
    结合多种损失来优化布局质量
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        layer_weight: float = 0.5,
        confidence_weight: float = 0.3,
        aesthetic_weight: float = 0.2,
        overlap_weight: float = 0.8,
        alignment_weight: float = 0.4,
        canvas_size: Tuple[int, int] = (1024, 1024)
    ):
        super().__init__()
        
        self.position_weight = position_weight
        self.layer_weight = layer_weight
        self.confidence_weight = confidence_weight
        self.aesthetic_weight = aesthetic_weight
        self.overlap_weight = overlap_weight
        self.alignment_weight = alignment_weight
        self.canvas_width, self.canvas_height = canvas_size
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.huber_loss = nn.SmoothL1Loss(reduction='mean')
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        predicted_layout: torch.Tensor,
        target_layout: torch.Tensor,
        element_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算复合损失
        
        Args:
            predicted_noise: [B, N, 6] 预测的噪声
            target_noise: [B, N, 6] 目标噪声
            predicted_layout: [B, N, 6] 预测的布局
            target_layout: [B, N, 6] 目标布局
            element_mask: [B, N] 元素掩码
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        
        # 应用掩码
        if element_mask is not None:
            mask = element_mask.unsqueeze(-1).expand_as(predicted_noise)
            predicted_noise = predicted_noise * mask
            target_noise = target_noise * mask
            predicted_layout = predicted_layout * mask
            target_layout = target_layout * mask
        
        # 1. 基础噪声预测损失
        noise_loss = self._compute_noise_loss(predicted_noise, target_noise)
        
        # 2. 位置损失
        position_loss = self._compute_position_loss(
            predicted_layout[..., :4], target_layout[..., :4]
        )
        
        # 3. 层序损失
        layer_loss = self._compute_layer_loss(
            predicted_layout[..., 4], target_layout[..., 4]
        )
        
        # 4. 置信度损失
        confidence_loss = self._compute_confidence_loss(
            predicted_layout[..., 5], target_layout[..., 5]
        )
        
        # 5. 美学损失
        aesthetic_loss = self._compute_aesthetic_loss(predicted_layout, element_mask)
        
        # 6. 重叠惩罚
        overlap_loss = self._compute_overlap_loss(predicted_layout, element_mask)
        
        # 7. 对齐损失
        alignment_loss = self._compute_alignment_loss(predicted_layout, element_mask)
        
        # 组合总损失
        total_loss = (
            noise_loss +
            self.position_weight * position_loss +
            self.layer_weight * layer_loss +
            self.confidence_weight * confidence_loss +
            self.aesthetic_weight * aesthetic_loss +
            self.overlap_weight * overlap_loss +
            self.alignment_weight * alignment_loss
        )
        
        # 损失字典
        loss_dict = {
            'total_loss': total_loss.item(),
            'noise_loss': noise_loss.item(),
            'position_loss': position_loss.item(),
            'layer_loss': layer_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'aesthetic_loss': aesthetic_loss.item(),
            'overlap_loss': overlap_loss.item(),
            'alignment_loss': alignment_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _compute_noise_loss(self, pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        """计算噪声预测损失"""
        return self.huber_loss(pred_noise, target_noise)
    
    def _compute_position_loss(self, pred_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """计算位置损失（包含IoU损失）"""
        # MSE损失
        mse_loss = self.mse_loss(pred_pos, target_pos)
        
        # IoU损失
        iou_loss = self._compute_iou_loss(pred_pos, target_pos)
        
        return mse_loss + 0.5 * iou_loss
    
    def _compute_iou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """计算IoU损失"""
        # boxes: [B, N, 4] (x, y, w, h)
        eps = 1e-8
        
        # 转换为 (x1, y1, x2, y2)
        def xyxy_from_xywh(boxes):
            return torch.stack([
                boxes[..., 0], boxes[..., 1],  # x1, y1
                boxes[..., 0] + boxes[..., 2], boxes[..., 1] + boxes[..., 3]  # x2, y2
            ], dim=-1)
        
        pred_xyxy = xyxy_from_xywh(pred_boxes)
        target_xyxy = xyxy_from_xywh(target_boxes)
        
        # 计算交集
        inter_x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
        inter_y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
        inter_x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
        inter_y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        pred_area = pred_boxes[..., 2] * pred_boxes[..., 3]
        target_area = target_boxes[..., 2] * target_boxes[..., 3]
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + eps)
        
        # IoU损失：1 - IoU
        return (1 - iou).mean()
    
    def _compute_layer_loss(self, pred_layer: torch.Tensor, target_layer: torch.Tensor) -> torch.Tensor:
        """计算层序损失"""
        # 排序损失：确保相对顺序正确
        mse_loss = self.mse_loss(pred_layer, target_layer)
        
        # 相对排序损失
        B, N = pred_layer.shape
        ranking_loss = 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                # 如果目标层序 i < j，那么预测也应该 i < j
                target_order = (target_layer[:, i] < target_layer[:, j]).float()
                pred_order = torch.sigmoid(pred_layer[:, j] - pred_layer[:, i])
                ranking_loss += F.binary_cross_entropy(pred_order, target_order)
        
        ranking_loss = ranking_loss / (N * (N - 1) / 2) if N > 1 else 0
        
        return mse_loss + 0.3 * ranking_loss
    
    def _compute_confidence_loss(self, pred_conf: torch.Tensor, target_conf: torch.Tensor) -> torch.Tensor:
        """计算置信度损失"""
        return self.mse_loss(pred_conf, target_conf)
    
    def _compute_aesthetic_loss(self, layout: torch.Tensor, element_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算美学损失（基于设计原则）"""
        # layout: [B, N, 6] (x, y, w, h, layer, confidence)
        B, N, _ = layout.shape
        
        if N < 2:
            return torch.tensor(0.0, device=layout.device)
        
        aesthetic_loss = 0.0
        
        # 1. 黄金比例损失
        golden_ratio = 1.618
        aspect_ratios = layout[..., 2] / (layout[..., 3] + 1e-8)  # w/h
        golden_loss = torch.abs(aspect_ratios - golden_ratio).mean()
        aesthetic_loss += 0.1 * golden_loss
        
        # 2. 尺寸协调损失（避免尺寸差异过大）
        areas = layout[..., 2] * layout[..., 3]
        if element_mask is not None:
            areas = areas * element_mask
        
        area_std = torch.std(areas, dim=1).mean()
        area_mean = torch.mean(areas, dim=1).mean()
        size_harmony_loss = area_std / (area_mean + 1e-8)
        aesthetic_loss += 0.2 * size_harmony_loss
        
        # 3. 视觉平衡损失（重心偏移）
        # 计算元素重心
        center_x = layout[..., 0] + layout[..., 2] / 2
        center_y = layout[..., 1] + layout[..., 3] / 2
        
        if element_mask is not None:
            center_x = center_x * element_mask
            center_y = center_y * element_mask
            valid_count = element_mask.sum(dim=1, keepdim=True)
        else:
            valid_count = N
        
        # 整体重心
        overall_center_x = center_x.sum(dim=1) / (valid_count + 1e-8)
        overall_center_y = center_y.sum(dim=1) / (valid_count + 1e-8)
        
        # 偏离画布中心的程度
        canvas_center_x = self.canvas_width / 2
        canvas_center_y = self.canvas_height / 2
        
        balance_loss = (
            torch.abs(overall_center_x - canvas_center_x) / canvas_center_x +
            torch.abs(overall_center_y - canvas_center_y) / canvas_center_y
        ).mean()
        aesthetic_loss += 0.15 * balance_loss
        
        return aesthetic_loss
    
    def _compute_overlap_loss(self, layout: torch.Tensor, element_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算重叠惩罚损失"""
        # layout: [B, N, 6] (x, y, w, h, layer, confidence)
        B, N, _ = layout.shape
        
        if N < 2:
            return torch.tensor(0.0, device=layout.device)
        
        overlap_loss = 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                if element_mask is not None:
                    # 跳过被掩码的元素
                    valid_pair = element_mask[:, i] * element_mask[:, j]
                    if valid_pair.sum() == 0:
                        continue
                
                # 计算两个box的重叠面积
                box1 = layout[:, i, :4]  # [B, 4]
                box2 = layout[:, j, :4]  # [B, 4]
                
                # 转换为 (x1, y1, x2, y2)
                box1_xyxy = torch.stack([
                    box1[..., 0], box1[..., 1],
                    box1[..., 0] + box1[..., 2], box1[..., 1] + box1[..., 3]
                ], dim=-1)
                
                box2_xyxy = torch.stack([
                    box2[..., 0], box2[..., 1],
                    box2[..., 0] + box2[..., 2], box2[..., 1] + box2[..., 3]
                ], dim=-1)
                
                # 计算交集
                inter_x1 = torch.max(box1_xyxy[..., 0], box2_xyxy[..., 0])
                inter_y1 = torch.max(box1_xyxy[..., 1], box2_xyxy[..., 1])
                inter_x2 = torch.min(box1_xyxy[..., 2], box2_xyxy[..., 2])
                inter_y2 = torch.min(box1_xyxy[..., 3], box2_xyxy[..., 3])
                
                inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
                
                # 计算重叠比例
                area1 = box1[..., 2] * box1[..., 3]
                area2 = box2[..., 2] * box2[..., 3]
                min_area = torch.min(area1, area2)
                
                overlap_ratio = inter_area / (min_area + 1e-8)
                
                if element_mask is not None:
                    overlap_ratio = overlap_ratio * valid_pair
                
                overlap_loss += overlap_ratio.mean()
        
        return overlap_loss / (N * (N - 1) / 2) if N > 1 else overlap_loss
    
    def _compute_alignment_loss(self, layout: torch.Tensor, element_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算对齐损失（鼓励元素对齐）"""
        # layout: [B, N, 6] (x, y, w, h, layer, confidence)
        B, N, _ = layout.shape
        
        if N < 2:
            return torch.tensor(0.0, device=layout.device)
        
        alignment_loss = 0.0
        alignment_threshold = 5.0  # 像素对齐阈值
        
        # 计算边界
        left_edges = layout[..., 0]                    # x
        right_edges = layout[..., 0] + layout[..., 2]  # x + w
        top_edges = layout[..., 1]                     # y
        bottom_edges = layout[..., 1] + layout[..., 3] # y + h
        center_x = layout[..., 0] + layout[..., 2] / 2 # center x
        center_y = layout[..., 1] + layout[..., 3] / 2 # center y
        
        edges = [left_edges, right_edges, top_edges, bottom_edges, center_x, center_y]
        
        for edge_type in edges:
            for i in range(N):
                for j in range(i + 1, N):
                    if element_mask is not None:
                        valid_pair = element_mask[:, i] * element_mask[:, j]
                        if valid_pair.sum() == 0:
                            continue
                    
                    # 计算边缘距离
                    edge_distance = torch.abs(edge_type[:, i] - edge_type[:, j])
                    
                    # 如果距离小于阈值，鼓励完全对齐
                    close_alignment = torch.exp(-edge_distance / alignment_threshold)
                    
                    if element_mask is not None:
                        close_alignment = close_alignment * valid_pair
                    
                    alignment_loss += edge_distance * close_alignment
        
        return alignment_loss.mean() / len(edges) if N > 1 else alignment_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.mse_loss(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_loss_function(config: Dict) -> LayoutDiffusionLoss:
    """
    根据配置创建损失函数
    
    Args:
        config: 损失函数配置字典
    
    Returns:
        loss_fn: 配置好的损失函数
    """
    return LayoutDiffusionLoss(
        position_weight=config.get('position_weight', 1.0),
        layer_weight=config.get('layer_weight', 0.5),
        confidence_weight=config.get('confidence_weight', 0.3),
        aesthetic_weight=config.get('aesthetic_weight', 0.2),
        overlap_weight=config.get('overlap_weight', 0.8),
        alignment_weight=config.get('alignment_weight', 0.4),
        canvas_size=config.get('canvas_size', (1024, 1024))
    )


if __name__ == "__main__":
    # 测试损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建损失函数
    loss_fn = LayoutDiffusionLoss()
    
    # 测试数据
    B, N = 2, 4
    predicted_noise = torch.randn(B, N, 6, device=device)
    target_noise = torch.randn(B, N, 6, device=device)
    
    predicted_layout = torch.tensor([
        [[100, 200, 50, 30, 3, 0.9],
         [200, 300, 80, 40, 5, 0.8],
         [50, 100, 60, 35, 1, 0.95],
         [300, 150, 70, 45, 7, 0.85]],
        [[120, 180, 45, 35, 4, 0.88],
         [250, 320, 75, 50, 6, 0.83],
         [80, 120, 65, 30, 2, 0.97],
         [320, 180, 60, 40, 8, 0.86]]
    ], dtype=torch.float32, device=device)
    
    target_layout = predicted_layout + torch.randn_like(predicted_layout) * 5
    
    element_mask = torch.ones(B, N, device=device)
    
    # 计算损失
    total_loss, loss_dict = loss_fn(
        predicted_noise, target_noise,
        predicted_layout, target_layout,
        element_mask
    )
    
    print("损失函数测试结果:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    print("损失函数测试通过！")
