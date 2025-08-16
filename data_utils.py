"""
数据处理工具
处理JSON数据，创建轨迹数据，数据增强等
"""

import torch
import torch.utils.data as data
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional, Union
import random
import os
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import logging
from tqdm import tqdm


class LayoutDataset(data.Dataset):
    """
    布局数据集类
    处理多层海报设计数据，生成训练样本
    """
    
    def __init__(
        self,
        data_file: str,
        image_root: str,
        canvas_size: Tuple[int, int] = (1024, 1024),
        max_elements: int = 25,
        augment: bool = True,
        trajectory_steps: int = 50,
        image_size: int = 224
    ):
        """
        Args:
            data_file: JSON数据文件路径
            image_root: 图片根目录
            canvas_size: 画布尺寸
            max_elements: 最大元素数量
            augment: 是否进行数据增强
            trajectory_steps: 轨迹步数
            image_size: 图片尺寸
        """
        self.data_file = data_file
        self.image_root = Path(image_root)
        self.canvas_size = canvas_size
        self.max_elements = max_elements
        self.augment = augment
        self.trajectory_steps = trajectory_steps
        self.image_size = image_size
        
        # 加载数据
        self.data = self._load_data()
        
        # 图片变换
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"加载了 {len(self.data)} 个布局样本")
    
    def _load_data(self) -> List[Dict]:
        """加载并预处理数据"""
        data = []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    
                    # 过滤掉元素过多的样本
                    if sample.get('total_layers', 0) <= self.max_elements:
                        processed_sample = self._process_sample(sample)
                        if processed_sample:
                            data.append(processed_sample)
                            
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"解析样本时出错: {e}")
                    continue
        
        return data
    
    def _process_sample(self, sample: Dict) -> Optional[Dict]:
        """处理单个样本"""
        try:
            # 解析JSON标签
            if isinstance(sample['label'], str):
                layout_data = json.loads(sample['label'])
            else:
                layout_data = sample['label']
            
            # 提取元素信息
            elements = layout_data.get('elements', [])
            if not elements:
                return None
            
            # 构建处理后的样本
            processed = {
                'elements': elements,
                'image_paths': sample.get('image_input', []),
                'total_layers': sample.get('total_layers', len(elements)),
                'canvas_size': self.canvas_size
            }
            
            return processed
            
        except Exception as e:
            print(f"处理样本时出错: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个训练样本"""
        sample = self.data[idx]
        
        # 加载图片
        images = self._load_images(sample['image_paths'])
        
        # 提取布局信息
        target_layout, element_mask = self._extract_layout(sample['elements'])
        
        # 创建起始布局（随机偏移）
        start_layout = self._create_start_layout(target_layout, element_mask)
        
        # 数据增强
        if self.augment:
            images, start_layout, target_layout = self._augment_data(
                images, start_layout, target_layout, element_mask
            )
        
        return {
            'images': images,                    # [N, 3, H, W]
            'start_layout': start_layout,       # [N, 6]
            'target_layout': target_layout,     # [N, 6]
            'element_mask': element_mask,       # [N]
            'num_elements': element_mask.sum().long()
        }
    
    def _load_images(self, image_paths: List[str]) -> torch.Tensor:
        """加载图片"""
        images = []
        
        for img_path in image_paths:
            try:
                # 完整路径
                full_path = self.image_root / img_path
                
                # 加载图片
                img = Image.open(full_path).convert('RGB')
                img_tensor = self.image_transform(img)
                images.append(img_tensor)
                
            except Exception as e:
                print(f"加载图片失败 {img_path}: {e}")
                # 创建空白图片作为占位符
                blank_img = torch.zeros(3, self.image_size, self.image_size)
                images.append(blank_img)
        
        # 补充到最大元素数量
        while len(images) < self.max_elements:
            blank_img = torch.zeros(3, self.image_size, self.image_size)
            images.append(blank_img)
        
        # 截断到最大数量
        images = images[:self.max_elements]
        
        return torch.stack(images)  # [N, 3, H, W]
    
    def _extract_layout(self, elements: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取布局信息"""
        layout = torch.zeros(self.max_elements, 6)  # [N, 6]
        mask = torch.zeros(self.max_elements)       # [N]
        
        for i, element in enumerate(elements[:self.max_elements]):
            if i >= self.max_elements:
                break
            
            # 提取边界框信息
            bbox = element.get('bounding_box', {})
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            width = bbox.get('width', 50)
            height = bbox.get('height', 50)
            
            # 提取层序信息
            layer = element.get('layer_order', i)
            
            # 置信度（可以根据元素类型或其他因素设置）
            confidence = 1.0
            
            layout[i] = torch.tensor([x, y, width, height, layer, confidence])
            mask[i] = 1.0
        
        return layout, mask
    
    def _create_start_layout(self, target_layout: torch.Tensor, element_mask: torch.Tensor) -> torch.Tensor:
        """创建起始布局（添加随机偏移）"""
        start_layout = target_layout.clone()
        
        # 只对有效元素添加偏移
        valid_indices = element_mask.bool()
        
        if valid_indices.sum() > 0:
            # 位置偏移
            position_noise = torch.randn_like(start_layout[valid_indices, :2]) * 50  # 50像素标准差
            start_layout[valid_indices, :2] += position_noise
            
            # 尺寸偏移
            size_noise = torch.randn_like(start_layout[valid_indices, 2:4]) * 20   # 20像素标准差
            start_layout[valid_indices, 2:4] += size_noise
            
            # 确保尺寸为正
            start_layout[valid_indices, 2:4] = torch.clamp(start_layout[valid_indices, 2:4], min=10)
            
            # 层序偏移
            layer_noise = torch.randn(valid_indices.sum()) * 2
            start_layout[valid_indices, 4] += layer_noise
            
            # 置信度偏移
            conf_noise = torch.randn(valid_indices.sum()) * 0.1
            start_layout[valid_indices, 5] += conf_noise
            start_layout[valid_indices, 5] = torch.clamp(start_layout[valid_indices, 5], 0, 1)
        
        return start_layout
    
    def _augment_data(
        self, 
        images: torch.Tensor, 
        start_layout: torch.Tensor, 
        target_layout: torch.Tensor,
        element_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """数据增强"""
        
        # 随机水平翻转
        if random.random() < 0.5:
            images = torch.flip(images, dims=[3])  # 水平翻转图片
            
            # 翻转布局坐标
            valid_indices = element_mask.bool()
            start_layout[valid_indices, 0] = self.canvas_size[0] - start_layout[valid_indices, 0] - start_layout[valid_indices, 2]
            target_layout[valid_indices, 0] = self.canvas_size[0] - target_layout[valid_indices, 0] - target_layout[valid_indices, 2]
        
        # 随机缩放
        if random.random() < 0.3:
            scale_factor = random.uniform(0.8, 1.2)
            valid_indices = element_mask.bool()
            
            # 缩放位置和尺寸
            start_layout[valid_indices, :4] *= scale_factor
            target_layout[valid_indices, :4] *= scale_factor
            
            # 确保在画布范围内
            start_layout[valid_indices, :4] = self._clamp_to_canvas(start_layout[valid_indices, :4])
            target_layout[valid_indices, :4] = self._clamp_to_canvas(target_layout[valid_indices, :4])
        
        # 随机旋转（小角度）
        if random.random() < 0.2:
            angle = random.uniform(-5, 5)  # 小角度旋转
            # 这里可以实现旋转变换，暂时跳过
            pass
        
        return images, start_layout, target_layout
    
    def _clamp_to_canvas(self, layout: torch.Tensor) -> torch.Tensor:
        """将布局限制在画布范围内"""
        # layout: [N, 4] (x, y, w, h)
        canvas_w, canvas_h = self.canvas_size
        
        # 限制位置
        layout[:, 0] = torch.clamp(layout[:, 0], 0, canvas_w - layout[:, 2])
        layout[:, 1] = torch.clamp(layout[:, 1], 0, canvas_h - layout[:, 3])
        
        # 限制尺寸
        layout[:, 2] = torch.clamp(layout[:, 2], 10, canvas_w - layout[:, 0])
        layout[:, 3] = torch.clamp(layout[:, 3], 10, canvas_h - layout[:, 1])
        
        return layout


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数，处理变长序列
    """
    # 提取各个字段
    images = torch.stack([sample['images'] for sample in batch])
    start_layouts = torch.stack([sample['start_layout'] for sample in batch])
    target_layouts = torch.stack([sample['target_layout'] for sample in batch])
    element_masks = torch.stack([sample['element_mask'] for sample in batch])
    num_elements = torch.stack([sample['num_elements'] for sample in batch])
    
    return {
        'images': images,           # [B, N, 3, H, W]
        'start_layouts': start_layouts,   # [B, N, 6]
        'target_layouts': target_layouts, # [B, N, 6]
        'element_masks': element_masks,   # [B, N]
        'num_elements': num_elements      # [B]
    }


class TrajectoryGenerator:
    """
    轨迹生成器
    基于起始和目标布局生成中间轨迹
    """
    
    def __init__(self, num_steps: int = 50):
        self.num_steps = num_steps
    
    def generate_trajectory(
        self, 
        start_layout: torch.Tensor, 
        target_layout: torch.Tensor
    ) -> torch.Tensor:
        """
        生成从起始到目标的轨迹
        
        Args:
            start_layout: [B, N, 6] 起始布局
            target_layout: [B, N, 6] 目标布局
        
        Returns:
            trajectory: [B, num_steps, N, 6] 完整轨迹
        """
        B, N, layout_dim = start_layout.shape
        device = start_layout.device
        
        # 创建时间插值
        t = torch.linspace(0, 1, self.num_steps, device=device)
        t = t.view(1, self.num_steps, 1, 1).expand(B, self.num_steps, N, layout_dim)
        
        # 线性插值
        start_expanded = start_layout.unsqueeze(1).expand(-1, self.num_steps, -1, -1)
        target_expanded = target_layout.unsqueeze(1).expand(-1, self.num_steps, -1, -1)
        
        trajectory = start_expanded + t * (target_expanded - start_expanded)
        
        return trajectory
    
    def generate_smooth_trajectory(
        self, 
        start_layout: torch.Tensor, 
        target_layout: torch.Tensor,
        smoothing_factor: float = 0.1
    ) -> torch.Tensor:
        """
        生成平滑轨迹（使用贝塞尔曲线或样条插值）
        """
        # 基础线性轨迹
        linear_trajectory = self.generate_trajectory(start_layout, target_layout)
        
        # 添加平滑控制点
        B, num_steps, N, layout_dim = linear_trajectory.shape
        device = linear_trajectory.device
        
        # 生成中间控制点
        mid_point = (start_layout + target_layout) / 2
        
        # 添加随机扰动到中间点
        noise = torch.randn_like(mid_point) * smoothing_factor
        control_point = mid_point + noise
        
        # 二次贝塞尔曲线插值
        t = torch.linspace(0, 1, num_steps, device=device)
        t = t.view(1, num_steps, 1, 1).expand(B, num_steps, N, layout_dim)
        
        # P(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        start_expanded = start_layout.unsqueeze(1).expand(-1, num_steps, -1, -1)
        control_expanded = control_point.unsqueeze(1).expand(-1, num_steps, -1, -1)
        target_expanded = target_layout.unsqueeze(1).expand(-1, num_steps, -1, -1)
        
        smooth_trajectory = (
            (1 - t) ** 2 * start_expanded +
            2 * (1 - t) * t * control_expanded +
            t ** 2 * target_expanded
        )
        
        return smooth_trajectory


def create_dataloader(
    data_file: str,
    image_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> data.DataLoader:
    """
    创建数据加载器
    
    Args:
        data_file: 数据文件路径
        image_root: 图片根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱数据
        **dataset_kwargs: 数据集额外参数
    
    Returns:
        dataloader: PyTorch数据加载器
    """
    dataset = LayoutDataset(
        data_file=data_file,
        image_root=image_root,
        **dataset_kwargs
    )
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def layout_to_json(layout: torch.Tensor, element_mask: torch.Tensor, element_ids: Optional[List[str]] = None) -> Dict:
    """
    将布局张量转换为JSON格式
    
    Args:
        layout: [N, 6] 布局张量 (x, y, w, h, layer, confidence)
        element_mask: [N] 元素掩码
        element_ids: 元素ID列表
    
    Returns:
        layout_json: JSON格式的布局描述
    """
    elements = []
    valid_indices = element_mask.bool()
    
    # 按层序排序
    valid_layout = layout[valid_indices]
    if len(valid_layout) > 0:
        sorted_indices = torch.argsort(valid_layout[:, 4])  # 按layer排序
        sorted_layout = valid_layout[sorted_indices]
        
        for i, element_layout in enumerate(sorted_layout):
            x, y, w, h, layer, confidence = element_layout.tolist()
            
            element_info = {
                "id": element_ids[i] if element_ids and i < len(element_ids) else f"element_{i}",
                "bounding_box": {
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h)
                },
                "layer_order": int(round(layer)),
                "confidence": float(confidence)
            }
            elements.append(element_info)
    
    return {
        "elements": elements,
        "canvas_size": {
            "width": 1024,
            "height": 1024
        },
        "total_elements": len(elements)
    }


class TrajectoryPreprocessor:
    """
    轨迹预处理器
    从海报数据集中生成轨迹数据并保存
    """
    
    def __init__(
        self,
        data_root: str = "/home/Data",
        canvas_size: Tuple[int, int] = (1024, 1024),
        num_trajectory_steps: int = 50,
        noise_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
        max_elements: int = 25
    ):
        """
        Args:
            data_root: 数据根目录
            canvas_size: 标准画布尺寸
            num_trajectory_steps: 轨迹步数
            noise_levels: 不同的噪声水平列表
            max_elements: 最大元素数量
        """
        self.data_root = Path(data_root)
        self.canvas_size = canvas_size
        self.num_trajectory_steps = num_trajectory_steps
        self.noise_levels = noise_levels
        self.max_elements = max_elements
        self.trajectory_gen = TrajectoryGenerator(num_steps=num_trajectory_steps)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_poster_data(self, poster_dir: Path) -> Optional[Dict]:
        """
        加载单个海报的数据
        
        Args:
            poster_dir: 海报目录路径
            
        Returns:
            poster_data: 包含布局信息和图片路径的字典，失败返回None
        """
        try:
            json_dir = poster_dir / "json"
            parse_dir = poster_dir / "parse"
            
            if not json_dir.exists() or not parse_dir.exists():
                return None
            
            # 查找JSON文件
            json_files = list(json_dir.glob("*.json"))
            if not json_files:
                return None
            
            json_file = json_files[0]
            
            # 读取JSON数据
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'layers' not in data:
                return None
            
            # 提取画布尺寸
            canvas_width = data.get('canvas_size', {}).get('width', 2480)
            canvas_height = data.get('canvas_size', {}).get('height', 3508)
            original_canvas = (canvas_width, canvas_height)
            
            # 处理图层信息
            elements = []
            image_paths = []
            
            for layer in data['layers']:
                # 跳过无效图层
                if 'files' not in layer or 'layer' not in layer['files']:
                    continue
                
                # 提取位置信息
                x = layer.get('x', 0)
                y = layer.get('y', 0)
                w = layer.get('w', 50)
                h = layer.get('h', 50)
                order = layer.get('order', 0)
                
                # 标准化到目标画布尺寸
                x_norm = (x / canvas_width) * self.canvas_size[0]
                y_norm = (y / canvas_height) * self.canvas_size[1]
                w_norm = (w / canvas_width) * self.canvas_size[0]
                h_norm = (h / canvas_height) * self.canvas_size[1]
                
                # 确保坐标为正数
                x_norm = max(0, x_norm)
                y_norm = max(0, y_norm)
                w_norm = max(10, w_norm)  # 最小宽度
                h_norm = max(10, h_norm)  # 最小高度
                
                # 确保不超出画布
                if x_norm + w_norm > self.canvas_size[0]:
                    w_norm = self.canvas_size[0] - x_norm
                if y_norm + h_norm > self.canvas_size[1]:
                    h_norm = self.canvas_size[1] - y_norm
                
                element = {
                    'bounding_box': {
                        'x': x_norm,
                        'y': y_norm,
                        'width': w_norm,
                        'height': h_norm
                    },
                    'layer_order': order,
                    'category': layer.get('category', 'unknown'),
                    'src': layer.get('src', 'unknown')
                }
                
                elements.append(element)
                
                # 图片路径 (转换为相对路径)
                layer_file = layer['files']['layer']
                if layer_file.startswith('/opt/liblibai-models/'):
                    # 转换为本地路径
                    relative_path = layer_file.replace(
                        '/opt/liblibai-models/user-workspace/laotianyu/layers/psd-parsed/',
                        ''
                    )
                    local_path = self.data_root / relative_path
                    if local_path.exists():
                        image_paths.append(str(local_path))
                    else:
                        # 尝试从parse目录查找
                        filename = Path(layer_file).name
                        local_file = parse_dir / filename
                        if local_file.exists():
                            image_paths.append(str(local_file))
                        else:
                            image_paths.append("")  # 空路径
                else:
                    image_paths.append(layer_file)
            
            # 按layer_order排序
            sorted_data = sorted(zip(elements, image_paths), key=lambda x: x[0]['layer_order'])
            elements, image_paths = zip(*sorted_data) if sorted_data else ([], [])
            
            # 限制元素数量
            elements = list(elements)[:self.max_elements]
            image_paths = list(image_paths)[:self.max_elements]
            
            return {
                'elements': elements,
                'image_paths': image_paths,
                'poster_name': poster_dir.name,
                'original_canvas': original_canvas,
                'total_layers': len(elements)
            }
            
        except Exception as e:
            self.logger.error(f"加载海报数据失败 {poster_dir}: {e}")
            return None
    
    def generate_noisy_layouts(self, target_layout: torch.Tensor, element_mask: torch.Tensor) -> List[torch.Tensor]:
        """
        生成不同噪声水平的起始布局
        
        Args:
            target_layout: 目标布局 [N, 6]
            element_mask: 元素掩码 [N]
            
        Returns:
            noisy_layouts: 不同噪声水平的起始布局列表
        """
        noisy_layouts = []
        valid_indices = element_mask.bool()
        
        for noise_level in self.noise_levels:
            noisy_layout = target_layout.clone()
            
            if valid_indices.sum() > 0:
                # 位置噪声
                pos_noise_scale = noise_level * min(self.canvas_size) * 0.1  # 基于画布尺寸的噪声
                position_noise = torch.randn_like(noisy_layout[valid_indices, :2]) * pos_noise_scale
                noisy_layout[valid_indices, :2] += position_noise
                
                # 尺寸噪声
                size_noise_scale = noise_level * min(self.canvas_size) * 0.05
                size_noise = torch.randn_like(noisy_layout[valid_indices, 2:4]) * size_noise_scale
                noisy_layout[valid_indices, 2:4] += size_noise
                
                # 确保尺寸为正
                noisy_layout[valid_indices, 2:4] = torch.clamp(noisy_layout[valid_indices, 2:4], min=10)
                
                # 层序噪声
                layer_noise = torch.randn(valid_indices.sum()) * noise_level * 3
                noisy_layout[valid_indices, 4] += layer_noise
                
                # 置信度噪声
                conf_noise = torch.randn(valid_indices.sum()) * noise_level * 0.2
                noisy_layout[valid_indices, 5] += conf_noise
                noisy_layout[valid_indices, 5] = torch.clamp(noisy_layout[valid_indices, 5], 0, 1)
                
                # 确保位置在画布范围内
                max_x = self.canvas_size[0] - noisy_layout[valid_indices, 2]
                max_y = self.canvas_size[1] - noisy_layout[valid_indices, 3]
                noisy_layout[valid_indices, 0] = torch.clamp(
                    noisy_layout[valid_indices, 0], 0, max_x.max().item()
                )
                noisy_layout[valid_indices, 1] = torch.clamp(
                    noisy_layout[valid_indices, 1], 0, max_y.max().item()
                )
            
            noisy_layouts.append(noisy_layout)
        
        return noisy_layouts
    
    def extract_layout_tensor(self, elements: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从元素列表提取布局张量
        
        Args:
            elements: 元素列表
            
        Returns:
            layout: 布局张量 [max_elements, 6]
            mask: 元素掩码 [max_elements]
        """
        layout = torch.zeros(self.max_elements, 6)
        mask = torch.zeros(self.max_elements)
        
        for i, element in enumerate(elements[:self.max_elements]):
            bbox = element['bounding_box']
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']
            layer = element['layer_order']
            confidence = 1.0
            
            layout[i] = torch.tensor([x, y, width, height, layer, confidence])
            mask[i] = 1.0
        
        return layout, mask
    
    def process_single_poster(self, poster_dir: Path) -> bool:
        """
        处理单个海报生成轨迹
        
        Args:
            poster_dir: 海报目录路径
            
        Returns:
            success: 是否成功
        """
        try:
            # 检查轨迹文件是否已存在
            trajectory_file = poster_dir / "trajectories.pkl"
            if trajectory_file.exists():
                return True
            
            # 加载海报数据
            poster_data = self.load_poster_data(poster_dir)
            if not poster_data:
                return False
            
            # 提取布局信息
            target_layout, element_mask = self.extract_layout_tensor(poster_data['elements'])
            
            # 跳过没有有效元素的海报
            if element_mask.sum() == 0:
                return False
            
            # 生成不同噪声水平的起始布局
            noisy_layouts = self.generate_noisy_layouts(target_layout, element_mask)
            
            # 生成轨迹数据
            trajectories = {}
            
            for i, (noise_level, start_layout) in enumerate(zip(self.noise_levels, noisy_layouts)):
                # 线性轨迹
                linear_trajectory = self.trajectory_gen.generate_trajectory(
                    start_layout.unsqueeze(0), target_layout.unsqueeze(0)
                ).squeeze(0)  # [num_steps, max_elements, 6]
                
                # 平滑轨迹
                smooth_trajectory = self.trajectory_gen.generate_smooth_trajectory(
                    start_layout.unsqueeze(0), target_layout.unsqueeze(0)
                ).squeeze(0)  # [num_steps, max_elements, 6]
                
                trajectories[f"noise_{noise_level:.1f}"] = {
                    'start_layout': start_layout,
                    'target_layout': target_layout,
                    'element_mask': element_mask,
                    'linear_trajectory': linear_trajectory,
                    'smooth_trajectory': smooth_trajectory,
                    'metadata': {
                        'poster_name': poster_data['poster_name'],
                        'num_elements': int(element_mask.sum()),
                        'canvas_size': self.canvas_size,
                        'trajectory_steps': self.num_trajectory_steps,
                        'noise_level': noise_level
                    }
                }
            
            # 保存轨迹数据
            with open(trajectory_file, 'wb') as f:
                pickle.dump(trajectories, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理海报失败 {poster_dir}: {e}")
            return False
    
    def process_all_posters(self, num_processes: Optional[int] = None, limit: Optional[int] = None):
        """
        使用多进程处理所有海报
        
        Args:
            num_processes: 进程数，None表示使用CPU核心数
            limit: 限制处理的海报数量，None表示处理全部
        """
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        # 获取所有海报目录
        poster_dirs = [d for d in self.data_root.iterdir() 
                      if d.is_dir() and d.name != 'transfer.log' and not d.name.startswith('.')]
        
        if limit:
            poster_dirs = poster_dirs[:limit]
        
        self.logger.info(f"开始处理 {len(poster_dirs)} 个海报，使用 {num_processes} 个进程")
        
        # 使用进程池处理
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 提交所有任务
            future_to_poster = {
                executor.submit(self.process_single_poster, poster_dir): poster_dir
                for poster_dir in poster_dirs
            }
            
            # 进度跟踪
            success_count = 0
            total_count = len(poster_dirs)
            
            with tqdm(total=total_count, desc="处理轨迹") as pbar:
                for future in as_completed(future_to_poster):
                    poster_dir = future_to_poster[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            'success': success_count,
                            'current': poster_dir.name[:20] + '...'
                        })
                    except Exception as e:
                        self.logger.error(f"处理 {poster_dir} 时出错: {e}")
                        pbar.update(1)
        
        self.logger.info(f"轨迹生成完成！成功处理: {success_count}/{total_count}")


def process_poster_worker(poster_dir_str: str) -> bool:
    """
    多进程工作函数
    """
    processor = TrajectoryPreprocessor()
    return processor.process_single_poster(Path(poster_dir_str))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成海报布局轨迹数据")
    parser.add_argument("--data_root", type=str, default="/home/data-action_head", help="数据根目录")
    parser.add_argument("--processes", type=int, default=None, help="进程数")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    parser.add_argument("--test", action="store_true", help="运行测试")
    
    args = parser.parse_args()
    
    if args.test:
        # 测试数据加载器
        print("测试数据处理工具...")
        
        # 测试轨迹生成器
        trajectory_gen = TrajectoryGenerator(num_steps=20)
        
        start = torch.tensor([[[100, 200, 50, 30, 3, 0.9],
                              [200, 300, 80, 40, 5, 0.8]]])
        target = torch.tensor([[[150, 250, 60, 35, 4, 0.95],
                               [250, 350, 90, 45, 6, 0.85]]])
        
        trajectory = trajectory_gen.generate_trajectory(start, target)
        print(f"轨迹形状: {trajectory.shape}")
        
        # 测试平滑轨迹
        smooth_trajectory = trajectory_gen.generate_smooth_trajectory(start, target)
        print(f"平滑轨迹形状: {smooth_trajectory.shape}")
        
        # 测试JSON转换
        layout = torch.tensor([[100, 200, 50, 30, 3, 0.9],
                              [200, 300, 80, 40, 5, 0.8]])
        mask = torch.tensor([1, 1])
        
        json_output = layout_to_json(layout, mask, ["img1", "img2"])
        print("JSON输出:")
        print(json.dumps(json_output, indent=2, ensure_ascii=False))
        
        print("数据处理工具测试通过！")
    else:
        # 运行轨迹生成
        processor = TrajectoryPreprocessor(data_root=args.data_root)
        processor.process_all_posters(num_processes=args.processes, limit=args.limit)
