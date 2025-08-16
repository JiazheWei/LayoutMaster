# LayoutMaster - data-action_head 数据格式训练指南

## 📁 数据格式

修改后的 `train.py` 现在支持 `data-action_head` 目录的数据格式：

```
/home/data-action_head/
├── Poster-Name-1/
│   ├── json/                    # Ground Truth JSON文件
│   │   └── xxx_parse_data.json
│   ├── parse/                   # PNG元素文件
│   │   ├── element1.png
│   │   ├── element2.png
│   │   └── ...
│   └── trajectories.pkl         # 预生成的轨迹数据
├── Poster-Name-2/
│   └── ...
└── ...
```

## 🚀 快速开始

### 1. 数据准备
确保你的数据已经按照上述格式组织，并且已经生成了轨迹文件：
```bash
# 验证数据格式
python test_data_loading.py
```

### 2. 训练模型
```bash
# 方法1: 使用配置文件直接训练
python train.py --config config_data_action_head.json --mode train

# 方法2: 使用交互式脚本
python run_training.py
```

### 3. 预测生成
```bash
# 批量预测（生成10个样本）
python train.py --config config_data_action_head.json --mode predict \
  --checkpoint ./checkpoints/best_model.pth --num_samples 10

# 单个海报预测
python train.py --config config_data_action_head.json --mode predict \
  --checkpoint ./checkpoints/best_model.pth \
  --poster_dir "/home/data-action_head/Ai Poster-xxx"
```

## 📊 数据集统计

根据测试结果：
- **训练集**: 800个样本 (80%)
- **验证集**: 200个样本 (20%)
- **元素数量**: 每个海报最多25个元素
- **图片尺寸**: 224x224像素
- **轨迹步数**: 50步
- **噪声水平**: 5个水平 (0.1-0.5)

## ⚙️ 配置文件

`config_data_action_head.json` 包含了针对轨迹数据的优化配置：

```json
{
  "data": {
    "use_trajectory_dataset": true,    // 使用新的轨迹数据集
    "data_root": "/home/data-action_head",
    "train_ratio": 0.8,
    "batch_size": 8,
    "image_size": 224
  },
  "model": {
    "max_elements": 25,
    "hidden_dim": 512
  },
  "training": {
    "num_epochs": 100,
    "save_every": 10
  }
}
```

## 🔧 主要功能

### 新增的TrajectoryDataset类
- 直接读取 `trajectories.pkl` 文件
- 自动加载对应的PNG元素图片
- 支持train/val数据集分割
- 随机选择不同噪声水平的轨迹数据

### 预测功能
- **批量预测**: 对验证集进行批量预测
- **单个预测**: 对指定海报进行预测
- **结果保存**: 预测结果自动保存为 `prediction.json` 到对应海报目录

### 输出文件格式
```json
{
  "elements": [
    {
      "id": "element_0",
      "bounding_box": {
        "x": 100.0,
        "y": 200.0,
        "width": 150.0,
        "height": 80.0
      },
      "layer_order": 3,
      "confidence": 0.95
    }
  ],
  "canvas_size": {
    "width": 1024,
    "height": 1024
  },
  "prediction_info": {
    "model": "LayoutMaster",
    "poster_name": "Ai Poster-xxx",
    "prediction_time": "2024-08-16T11:30:00",
    "total_elements": 15
  }
}
```

## 🎯 使用示例

### 完整训练流程
```bash
# 1. 测试数据加载
python test_data_loading.py

# 2. 开始训练
python train.py --config config_data_action_head.json --mode train

# 3. 训练完成后预测
python train.py --config config_data_action_head.json --mode predict \
  --checkpoint ./checkpoints/best_model.pth --num_samples 20
```

### 单个海报推理
```bash
python train.py --config config_data_action_head.json --mode predict \
  --checkpoint ./checkpoints/best_model.pth \
  --poster_dir "/home/data-action_head/Climate Change Poster-xxx"
```

## 📈 训练监控

训练过程中的关键指标：
- **total_loss**: 总损失
- **noise_loss**: 噪声预测损失  
- **position_loss**: 位置损失
- **layer_loss**: 层序损失
- **aesthetic_loss**: 美学损失

## 🐛 常见问题

### 1. 数据加载失败
确保每个海报目录都包含：
- `json/` 目录和JSON文件
- `parse/` 目录和PNG文件  
- `trajectories.pkl` 文件

### 2. GPU内存不足
减少配置文件中的 `batch_size`：
```json
{
  "data": {
    "batch_size": 4  // 从8减少到4
  }
}
```

### 3. 图片加载错误
检查 `parse/` 目录下的PNG文件是否完整，TrajectoryDataset会自动处理缺失的图片。

## ✨ 新特性

1. **高效数据加载**: 直接使用预生成的轨迹数据，无需训练时计算
2. **灵活预测模式**: 支持批量和单个预测
3. **自动结果保存**: 预测结果直接保存到对应目录
4. **完整的数据验证**: 包含数据加载测试脚本
5. **用户友好界面**: 交互式训练脚本

现在你可以使用修改后的代码对 `data-action_head` 数据进行高效训练和预测！🎉
