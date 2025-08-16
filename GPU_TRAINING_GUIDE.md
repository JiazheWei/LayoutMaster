# 🚀 LayoutMaster 8GPU训练完整指南

## 🎯 硬件配置
- **GPU**: 8x NVIDIA H20 (97GB显存/卡)
- **总显存**: 776GB
- **优化**: 专为大规模训练优化的配置

## 🏃‍♂️ 快速启动 (推荐)

### 方法1: 一键启动脚本
```bash
cd /home/LayoutMaster
./start_training.sh
```

### 方法2: 交互式启动
```bash
cd /home/LayoutMaster  
python train_multi_gpu.py
```

### 方法3: 直接启动
```bash
cd /home/LayoutMaster
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --config config_data_action_head.json --mode train
```

## ⚙️ 多GPU优化配置

### 关键配置参数
```json
{
  "use_multi_gpu": true,
  "data": {
    "batch_size": 32,        // 总批次大小 (每GPU 4个样本)
    "num_workers": 32        // 数据加载进程数
  },
  "optimizer": {
    "lr": 4e-4              // 4x基础学习率 (适应多GPU)
  }
}
```

### 性能优化设置
- **DataParallel**: 自动使用所有可用GPU
- **异步数据加载**: 32个worker进程
- **梯度累积**: 自动处理大批次
- **内存优化**: 自动管理多GPU显存

## 📊 训练监控

### 实时监控GPU状态
```bash
# 方法1: 实时显示
watch -n 1 nvidia-smi

# 方法2: 简化输出
watch -n 1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
```

### 查看训练进度
```bash
# 查看最新日志
tail -f logs/training_*.log

# 查看检查点
ls -la checkpoints/
```

### GPU使用率优化
- **目标GPU利用率**: 90-95%
- **显存使用率**: 70-80% (留出缓冲)
- **温度控制**: <80°C

## 🎛️ 运行参数调整

### 如果显存不足
```bash
# 减少批次大小
# 修改 config_data_action_head.json:
"batch_size": 16  # 从32减少到16
```

### 如果想要更快训练
```bash
# 增加学习率
"lr": 8e-4  # 进一步增加学习率
```

### 如果遇到数据加载瓶颈
```bash
# 增加数据加载进程
"num_workers": 64  # 充分利用128核CPU
```

## 📈 预期性能

### 训练速度估算
- **单个epoch**: ~5-10分钟 (800个训练样本)
- **100个epochs**: ~8-16小时
- **每秒样本数**: ~2-4个样本/秒 (8GPU并行)

### 显存使用预估
- **每GPU**: ~40-60GB (97GB总容量)
- **模型大小**: ~2-3GB
- **批次数据**: ~30-40GB/GPU

## 🚦 训练状态说明

### 正常运行标志
```
✅ 使用 8 个GPU进行训练
✅ 训练数据集包含 800 个样本  
✅ 验证数据集包含 200 个样本
✅ GPU利用率: 90%+
✅ 损失函数稳定下降
```

### 问题排查
```
❌ GPU利用率低 (<50%) → 检查数据加载速度
❌ 显存溢出 → 减少batch_size
❌ 温度过高 (>85°C) → 检查散热/降低功耗
❌ 训练速度慢 → 增加num_workers
```

## 💾 检查点管理

### 自动保存
- **最佳模型**: `checkpoints/best_model.pth`
- **定期保存**: 每10个epoch保存一次
- **断点续训**: 支持从任意检查点恢复

### 手动保存和加载
```bash
# 恢复训练
python train.py --config config_data_action_head.json --mode train --resume checkpoints/checkpoint_epoch_50.pth

# 预测模式
python train.py --config config_data_action_head.json --mode predict --checkpoint checkpoints/best_model.pth --num_samples 20
```

## 🎯 最佳实践

### 1. 启动前检查
```bash
# 检查数据
python test_data_loading.py

# 检查GPU状态
nvidia-smi

# 确认配置
cat config_data_action_head.json
```

### 2. 训练期间
- 定期查看GPU利用率
- 监控损失函数变化  
- 检查模型输出质量
- 确保检查点正常保存

### 3. 优化建议
- **第一次运行**: 使用默认配置
- **如果训练稳定**: 可增加batch_size到48-64
- **如果显存充足**: 可增加模型hidden_dim
- **如果数据丰富**: 可增加num_epochs到200

## 🔧 故障排除

### 常见问题
1. **CUDA内存不足**
   ```bash
   # 解决方案：减少batch_size
   "batch_size": 16
   ```

2. **数据加载慢**
   ```bash
   # 解决方案：增加worker数量
   "num_workers": 64
   ```

3. **模型不收敛**
   ```bash
   # 解决方案：调整学习率
   "lr": 2e-4  # 降低学习率
   ```

4. **GPU利用率低**
   ```bash
   # 检查数据管道
   htop  # 查看CPU使用情况
   ```

## 🎉 开始训练！

现在你已经准备好开始使用8GPU进行LayoutMaster训练了：

```bash
cd /home/LayoutMaster
./start_training.sh
```

训练愉快！🚀
