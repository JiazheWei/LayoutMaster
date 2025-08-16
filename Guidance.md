# data_utils.py
## 测试功能
```
cd /home/LayoutMaster
python data_utils.py --test
```

## 生成少量轨迹测试
```
python data_utils.py --data_root /home/Data --limit 10 --processes 4
```

## 处理所有数据
```
# 使用所有CPU核心处理全部数据
python data_utils.py --data_root /home/Data --processes 64

# 或者限制进程数避免内存不足
python data_utils.py --data_root /home/Data --processes 32
```