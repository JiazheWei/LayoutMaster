# LayoutMaster

基于VLA生成动作序列给出合理动作规划的想法，现对多图层生成模型做出对应的改进，主要为训练action head让模型学习如何将多图层从随机的初始位置一步步“移动”至ground truth位置。

## 一、Framework

### 核心思想
- **轨迹优化而非文本匹配**：通过学习从噪声布局到最优布局的去噪轨迹
- **多模态融合**：结合视觉特征和空间位置信息
- **扩散建模**：使用DDPM（Denoising Diffusion Probabilistic Model）框架

## 二、Module

### 1. **Data preprocessing** (`data_utils.py`)

**功能**：处理原始数据，转换为模型可用格式

**输入数据格式**：
```json
{
  "image_input": ["img1.png", "img2.png"],
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

**关键处理步骤**：
- **图像加载**：将PNG图像resize到224×224，标准化
- **布局提取**：将边界框转换为6维张量 `[x, y, w, h, layer, confidence]`
- **噪声添加**：从ground truth创建带噪声的起始布局
- **数据增强**：水平翻转、随机缩放等

**输出格式**：
```python
{
    'images': torch.Tensor[25, 3, 224, 224],      # 图像张量
    'start_layouts': torch.Tensor[25, 6],        # 起始布局
    'target_layouts': torch.Tensor[25, 6],       # 目标布局  
    'element_masks': torch.Tensor[25],           # 有效元素掩码
}
```

### 2. **Visual encoder** (`models.py` - `VisualFeatureExtractor`)

**功能**：从图像中提取语义特征

**架构**：
- **Backbone**：预训练的CLIP模型
- **投影层**：将CLIP特征映射到指定维度
- **冻结策略**：通常冻结backbone参数

**输入输出**：
- 输入：`[B, N, 3, H, W]` 图像批次
- 输出：`[B, N, 768]` 视觉特征

### 3. **LayoutDiffusionTransformer** (`models.py` - `LayoutDiffusionTransformer`)

**功能**：预测布局噪声，实现去噪

**核心架构**：
```python
# 关键组件
- 时间步嵌入: SinusoidalPositionEmbedding
- 视觉投影: Linear(768 -> 512)
- 布局投影: Linear(6 -> 512) 
- 位置编码: Learnable positional embedding
- Transformer层: 12层CrossAttentionBlock
- 输出头: ActionHead
```

**交叉注意力机制**：
- **自注意力**：布局元素间的空间关系
- **交叉注意力**：视觉特征指导布局调整
- **前馈网络**：特征变换和非线性

**输入输出**：
- 输入：视觉特征 + 噪声布局 + 时间步
- 输出：预测的噪声 `[B, N, 6]`

### 4. **Action head** (`models.py` - `ActionHead`)

**功能**：将Transformer特征转换为具体的布局调整动作

**分支设计**：
```python
# 连续动作（位置调整）
continuous_head: (x, y, w, h) 调整量
# 离散动作（层序调整）  
layer_head: layer 调整量
# 置信度预测
confidence_head: 元素置信度
```

### 5. **Diffusion Scheduler** (`diffusion_utils.py` - `DiffusionScheduler`)

**功能**：管理扩散过程的噪声调度

**关键参数**：
- `num_timesteps`: 1000（扩散步数）
- `beta_schedule`: linear/cosine（噪声调度策略）
- `beta_start/end`: 0.0001/0.02（噪声范围）

**核心方法**：
```python
# 前向扩散：添加噪声
add_noise(x_start, noise, timesteps)
# 反向推理：预测原始数据
predict_start_from_noise(x_t, t, noise)
```

### 6. **Layout Normalizer** (`diffusion_utils.py` - `LayoutNormalizer`)

**功能**：将布局坐标标准化到[-1,1]范围

**标准化规则**：
```python
# 位置和尺寸：映射到[-1,1]
normalized_x = (x / canvas_width) * 2 - 1
# 层序：归一化到[-1,1]  
normalized_layer = (layer / max_layer) * 2 - 1
# 置信度：[0,1] -> [-1,1]
normalized_conf = conf * 2 - 1
```

### 7. **Loss function** (`loss_functions.py` - `LayoutDiffusionLoss`)

**多重损失设计**：

```python
总损失 = 噪声损失 + 位置损失 + 层序损失 + 美学损失 + 重叠惩罚 + 对齐损失
```

**各项损失**：
- **噪声损失**：Huber Loss，预测噪声的主要损失
- **位置损失**：MSE + IoU，确保位置准确
- **层序损失**：排序一致性，保持相对层序
- **美学损失**：黄金比例、视觉平衡
- **重叠惩罚**：避免元素重叠
- **对齐损失**：鼓励边缘对齐

## 三、训练和推理流程

### 训练过程：
1. **加载数据**：JSONL → 图像+布局张量
2. **特征提取**：CLIP编码图像特征
3. **扩散过程**：
   - 标准化target layout
   - 随机采样时间步t
   - 添加噪声：`noisy = √(αₜ)×target + √(1-αₜ)×ε`
4. **模型预测**：预测添加的噪声ε'
5. **损失计算**：`||ε - ε'||²` + 其他损失项
6. **参数更新**：反向传播优化

### 推理过程：
1. **初始化**：随机布局或用户指定
2. **DDPM采样**：
   - 从t=1000开始
   - 逐步去噪：t → t-1
   - 模型预测噪声
   - 更新布局状态
3. **输出**：反标准化 → JSON格式

## 四、输入输出规格

### 系统输入：
- **图像文件**：多个PNG文件（设计元素）
- **配置参数**：模型超参数、画布尺寸等
- **可选初始布局**：用户指定的起始位置

### 系统输出：
```json
{
  "elements": [
    {
      "id": "element_0",
      "bounding_box": {
        "x": 150.5, "y": 200.3,
        "width": 80.2, "height": 60.1
      },
      "layer_order": 3,
      "confidence": 0.95
    }
  ],
  "canvas_size": {"width": 1024, "height": 1024},
  "total_elements": 3
}
```