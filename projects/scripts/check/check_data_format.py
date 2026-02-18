#!/usr/bin/env python3
"""
检查 qlib 数据格式要求
"""
import pandas as pd
import os

# 读取我们的数据
data_file = "/mnt/d/quant/qlib_backtest/data/qlib_data/instruments/1min/RB9999.XSGE.csv"
print("=== 我们的数据格式 ===")
with open(data_file, 'r') as f:
    print("前10行:")
    for i, line in enumerate(f):
        if i >= 10:
            break
        print(line.rstrip())

print("\n=== 使用 pandas 读取 ===")
df = pd.read_csv(data_file, index_col=0, parse_dates=True)
print(f"形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"索引名称: {df.index.name}")
print(f"索引类型: {type(df.index)}")
print(f"前3行:")
print(df.head(3))

# qlib 可能需要特定的列名格式
# 让我们尝试重命名列
print("\n=== 尝试不同的列名格式 ===")

# qlib 可能需要没有 $ 前缀的列名
df_renamed = df.copy()
df_renamed.columns = [col.replace('$', '') for col in df_renamed.columns]
print(f"重命名后的列名: {df_renamed.columns.tolist()}")

# 或者 qlib 可能需要 multiindex
print("\n=== 检查 qlib 期望的格式 ===")
print("qlib 期望的格式通常是:")
print("- index: datetime")
print("- columns: MultiIndex (instrument, feature)")
print("- 或者: columns = feature, data 包含多列（每个合约一列）")
print("\n我们的数据格式:")
print(f"- index: {type(df.index)}")
print(f"- columns: {df.columns}")
print(f"- 这可能不是 qlib 期望的格式")
