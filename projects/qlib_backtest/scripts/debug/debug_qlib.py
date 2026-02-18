#!/usr/bin/env python3
"""
调试 qlib 数据加载
"""
import qlib
from qlib.config import REG_CN
from qlib.data import D

# 初始化 qlib
print("初始化 qlib...")
qlib.init(provider_uri="/mnt/d/quant/qlib_backtest/data/qlib_data", region=REG_CN)
print("初始化完成\n")

# 测试获取日历
print("测试获取日历...")
try:
    calendar = D.calendar(freq="1min", start_time="2023-01-03", end_time="2023-01-10")
    print(f"日历长度: {len(calendar)}")
    print(f"前5个交易日: {calendar[:5]}")
except Exception as e:
    print(f"获取日历失败: {e}")

# 测试获取特征
print("\n测试获取特征...")
instruments = ["RB9999.XSGE"]
fields = ["$close", "$high", "$low", "$volume"]

try:
    df = D.features(instruments, fields, freq="1min", start_time="2023-01-03", end_time="2023-01-10")
    print(f"数据形状: {df.shape}")
    print(f"列: {df.columns.tolist()}")
    print(f"前5行:")
    print(df.head())
except Exception as e:
    print(f"获取特征失败: {e}")
    import traceback
    traceback.print_exc()

# 检查 qlib 数据目录结构
print("\n检查数据目录结构...")
import os
data_dir = "/mnt/d/quant/qlib_backtest/data/qlib_data"
for root, dirs, files in os.walk(data_dir):
    level = root.replace(data_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:3]:  # 只显示前3个文件
        print(f'{subindent}{file}')
    if len(files) > 3:
        print(f'{subindent}... and {len(files) - 3} more files')
