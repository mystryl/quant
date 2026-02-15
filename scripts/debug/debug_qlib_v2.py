#!/usr/bin/env python3
"""
调试 qlib 数据加载（v2 格式）
"""
import qlib
from qlib.config import REG_CN
from qlib.data import D

# 初始化 qlib
print("初始化 qlib...")
qlib.init(provider_uri="/mnt/d/quant/qlib_backtest/data/qlib_data_v2", region=REG_CN)
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
