#!/usr/bin/env python3
"""
多GPU训练脚本
专门针对8GPU NVIDIA H20配置优化
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def setup_environment():
    """设置多GPU训练环境"""
    # 设置环境变量优化多GPU性能
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',  # 使用所有8个GPU
        'TORCH_CUDA_ARCH_LIST': '8.0;8.6;8.9;9.0',  # H20 支持的架构
        'CUDA_LAUNCH_BLOCKING': '0',  # 异步CUDA调用提高性能
        'TORCH_CUDNN_V8_API_ENABLED': '1',  # 启用cuDNN v8优化
        'NCCL_DEBUG': 'INFO',  # NCCL调试信息
        'NCCL_IB_DISABLE': '1',  # 如果InfiniBand有问题可以禁用
        'OMP_NUM_THREADS': '8',  # OpenMP线程数
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量: {key}={value}")

def check_gpu_memory():
    """检查GPU显存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("\n🔍 GPU显存状态:")
            for i, line in enumerate(lines):
                used, total = line.split(', ')
                usage_percent = (int(used) / int(total)) * 100
                print(f"  GPU {i}: {used}MB / {total}MB ({usage_percent:.1f}%)")
            return True
        return False
    except Exception as e:
        print(f"❌ 无法检查GPU状态: {e}")
        return False

def optimize_system():
    """系统优化设置"""
    optimizations = [
        # 设置GPU持久化模式
        "nvidia-smi -pm 1",
        # 设置最大性能模式
        "nvidia-smi -ac 877,1593",  # H20的内存和GPU频率
    ]
    
    print("\n⚡ 应用系统优化...")
    for cmd in optimizations:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ {cmd}")
            else:
                print(f"  ⚠️ {cmd} - {result.stderr.strip()}")
        except Exception as e:
            print(f"  ❌ {cmd} - {e}")

def start_training():
    """启动训练"""
    config_file = "config_data_action_head.json"
    
    # 检查配置文件
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    # 检查数据目录
    data_dir = "/home/data-action_head"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    # 创建检查点目录
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"\n🚀 开始8GPU训练...")
    print(f"📁 数据目录: {data_dir}")
    print(f"⚙️  配置文件: {config_file}")
    print(f"💾 检查点目录: ./checkpoints/")
    
    # 启动训练命令
    cmd = [
        sys.executable, "train.py",
        "--config", config_file,
        "--mode", "train"
    ]
    
    print(f"🔄 执行命令: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 启动训练
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True,
                                 bufsize=1)
        
        # 实时输出训练日志
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        end_time = time.time()
        
        if process.returncode == 0:
            training_time = end_time - start_time
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            print(f"\n✅ 训练完成！总用时: {hours}小时 {minutes}分钟")
            return True
        else:
            print(f"\n❌ 训练失败，退出码: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⛔ 用户中断训练")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        return False

def monitor_training():
    """监控训练过程"""
    print("\n📊 训练监控命令:")
    print("1. 实时查看GPU使用情况:")
    print("   watch -n 1 nvidia-smi")
    print()
    print("2. 查看训练日志:")
    print("   tail -f ./checkpoints/training.log")
    print()
    print("3. 监控系统资源:")
    print("   htop")
    print()

def main():
    print("=" * 80)
    print("🚀 LayoutMaster 多GPU训练启动器")
    print("🎯 配置: 8x NVIDIA H20 (97GB显存)")
    print("=" * 80)
    
    # 检查GPU
    if not check_gpu_memory():
        return
    
    # 设置环境
    setup_environment()
    
    # 系统优化
    optimize_system()
    
    # 显示监控信息
    monitor_training()
    
    # 询问是否开始训练
    print("\n准备开始训练...")
    choice = input("是否立即开始训练? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes', '是']:
        success = start_training()
        if success:
            print("\n🎉 训练任务完成！")
            print("💡 检查 ./checkpoints/ 目录查看保存的模型")
        else:
            print("\n💔 训练失败，请检查日志")
    else:
        print("🛑 训练取消")
    
    # 最终GPU状态
    print("\n🔍 最终GPU状态:")
    check_gpu_memory()

if __name__ == "__main__":
    main()
