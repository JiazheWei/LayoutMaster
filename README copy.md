# LayoutMaster：基于轨迹优化的多层海报布局生成

## 项目概述

LayoutMaster 是一个基于扩散模型（Diffusion Transformer）的多层海报布局生成系统，通过轨迹优化而非直接文本匹配来学习布局原则。系统能够处理多个视觉元素，预测它们的最优位置、尺寸和层序关系。

## 核心特性

- **轨迹优化训练**: 通过学习从随机偏移到最优位置的轨迹来训练模型
- **多模态处理**: 结合视觉特征和布局信息
- **层序管理**: 同时优化元素的空间位置和层叠顺序
- **美学约束**: 内置设计原则和美学损失函数
- **JSON输出**: 直接生成可用于渲染引擎的JSON格式布局描述

## 系统架构

```
输入：多个PNG图像 → 视觉特征提取 → 轨迹优化模型 → 布局调整 → JSON输出
```

### 核心组件

1. **LayoutDiffusionTransformer**: 基于DiT的主要模型
2. **VisualFeatureExtractor**: 视觉特征提取器（基于CLIP）
3. **ActionHead**: 预测位置和层序调整的动作头
4. **DiffusionScheduler**: 扩散过程调度器
5. **复合损失函数**: 包含位置、层序、美学等多种损失

## 安装和环境配置

### 依赖要求

```bash
pip install torch torchvision
pip install transformers
pip install PIL
pip install tqdm
pip install swanlab  # 可选，用于实验记录
pip install numpy
```

### 所需的Backbone模型

本项目需要预训练的视觉模型作为backbone。推荐使用以下模型之一：

#### 1. CLIP模型（推荐）

```python
# 使用Hugging Face下载
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# 保存到本地
model.save_pretrained("/path/to/clip-vit-base-patch32")
```

或直接下载：
- **OpenAI CLIP ViT-B/32**: `openai/clip-vit-base-patch32`
- **OpenAI CLIP ViT-L/14**: `openai/clip-vit-large-patch14`

#### 2. 其他可选模型

- **ALIGN**: Google的多模态对齐模型
- **BLIP**: Salesforce的视觉语言模型
- **Chinese CLIP**: 中文多模态模型

### 下载命令示例

```bash
# 使用git lfs下载CLIP模型
git lfs install
git clone https://huggingface.co/openai/clip-vit-base-patch32

# 或使用Python下载
python -c "
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model.save_pretrained('./clip-vit-base-patch32')
processor.save_pretrained('./clip-vit-base-patch32')
"
```

## 使用指南

### 1. 配置文件设置

修改 `config.json` 中的路径：

```json
{
  "data": {
    "train_file": "/path/to/your/psd_dataset_92000_fast.jsonl",
    "image_root": "/path/to/your/images",
    "batch_size": 4
  },
  "backbone_path": "/path/to/clip-vit-base-patch32"
}
```

### 2. 数据格式

训练数据应为JSONL格式，每行包含：

```json
{
  "image_input": ["path/to/image1.png", "path/to/image2.png"],
  "label": {
    "elements": [
      {
        "bounding_box": {"x": 100, "y": 200, "width": 50, "height": 30},
        "layer_order": 3,
        "id": "element_1"
      }
    ]
  },
  "total_layers": 5
}
```

### 3. 训练模型

```bash
# 基础训练
python train.py --config config.json

# 恢复训练
python train.py --config config.json --resume checkpoints/checkpoint_epoch_10.pth
```

### 4. 推理生成

```bash
# 基础推理
python inference.py --model checkpoints/best_model.pth --config config.json --images img1.png img2.png img3.png --output layout.json

# 迭代优化推理
python inference.py --model checkpoints/best_model.pth --config config.json --images img1.png img2.png img3.png --iterative --iterations 3 --output layout.json
```

## 输出格式

生成的JSON布局描述格式：

```json
{
  "elements": [
    {
      "id": "element_0",
      "bounding_box": {
        "x": 150.5,
        "y": 200.3,
        "width": 80.2,
        "height": 60.1
      },
      "layer_order": 3,
      "confidence": 0.95
    }
  ],
  "canvas_size": {
    "width": 1024,
    "height": 1024
  },
  "total_elements": 3
}
```

## 训练参数说明

### 模型参数
- `visual_dim`: 视觉特征维度（默认768，对应CLIP）
- `hidden_dim`: 隐藏层维度（默认512）
- `num_layers`: Transformer层数（默认12）
- `max_elements`: 最大元素数量（默认25）

### 扩散参数
- `num_timesteps`: 扩散时间步数（默认1000）
- `beta_start/beta_end`: 噪声调度参数
- `schedule_type`: 调度类型（linear/cosine）

### 损失权重
- `position_weight`: 位置损失权重（1.0）
- `layer_weight`: 层序损失权重（0.5）
- `aesthetic_weight`: 美学损失权重（0.2）
- `overlap_weight`: 重叠惩罚权重（0.8）
- `alignment_weight`: 对齐损失权重（0.4）

## 性能优化建议

### 训练优化
1. **批次大小**: 根据GPU内存调整，推荐4-8
2. **梯度累积**: 如果显存不足，增加`gradient_accumulation_steps`
3. **混合精度**: 启用`fp16`训练加速
4. **学习率**: 推荐1e-4，根据收敛情况调整

### 推理优化
1. **步数**: 生产环境可减少到20-30步
2. **迭代优化**: 对质量要求高的场景使用3-5次迭代
3. **批处理**: 多个布局任务可以批量处理

## 扩展和定制

### 添加新的视觉Backbone

1. 在`models.py`中的`VisualFeatureExtractor`类添加新的backbone支持
2. 实现对应的特征提取逻辑
3. 更新配置文件中的backbone设置

### 自定义损失函数

在`loss_functions.py`中添加新的损失项：

```python
def custom_loss(self, layout):
    # 实现自定义损失逻辑
    return loss_value
```

### 数据增强扩展

在`data_utils.py`的`_augment_data`方法中添加新的增强策略。

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 启用gradient_checkpointing
   - 使用更小的图片尺寸

2. **模型不收敛**
   - 检查学习率设置
   - 验证数据格式
   - 调整损失函数权重

3. **推理结果不理想**
   - 增加推理步数
   - 尝试迭代优化
   - 检查输入图片质量

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查模型输出
with torch.no_grad():
    output = model(visual_features, layout_state, timestep)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
```

## 许可证

本项目采用MIT许可证。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题或建议，请创建GitHub Issue。
