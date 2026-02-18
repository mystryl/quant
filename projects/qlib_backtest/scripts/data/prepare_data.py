#!/usr/bin/env python3
"""
将 CSV 格式的数据转换为 qlib 格式
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil

# 配置路径
DATA_DIR = Path("/mnt/d/quant/qlib_data")
QLIB_DIR = Path("/mnt/d/quant/qlib_backtest/data/qlib_data")
INSTRUMENTS_DIR = QLIB_DIR / "instruments"

# qlib 需要的频率
FREQS = ["1min", "day"]

# 清理并重建 qlib 数据目录
if QLIB_DIR.exists():
    shutil.rmtree(QLIB_DIR)
QLIB_DIR.mkdir(parents=True, exist_ok=True)

# 创建 qlib 需要的目录结构
# qlib 期望: instruments/freq/instrument.csv
for freq in FREQS:
    freq_dir = INSTRUMENTS_DIR / freq
    freq_dir.mkdir(parents=True, exist_ok=True)

# 读取交易日历
day_calendar = pd.read_csv(DATA_DIR / "calendars" / "day.txt", header=None, names=['date'])
min_calendar = pd.read_csv(DATA_DIR / "calendars" / "1min.txt", header=None, names=['datetime'])

print(f"交易日历: {len(day_calendar)} 天")
print(f"分钟日历: {len(min_calendar)} 个时间点")

# 读取合约列表
with open(DATA_DIR / "instruments" / "all.txt", 'r') as f:
    instruments = [line.strip() for line in f if line.strip()]

print(f"合约数量: {len(instruments)}")

# 为每个合约创建 qlib 格式数据
for instrument in instruments:
    print(f"\n处理合约: {instrument}")

    instrument_dir = DATA_DIR / "instruments" / instrument
    if not instrument_dir.exists():
        print(f"  跳过: 数据目录不存在")
        continue

    # 读取各个字段
    fields = {}
    for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'open_interest']:
        field_file = instrument_dir / f"{field}.csv"
        if field_file.exists():
            df = pd.read_csv(field_file, index_col=0, parse_dates=True)
            fields[field] = df
            print(f"  {field}: {len(df)} 行")
        else:
            print(f"  跳过 {field}: 文件不存在")

    # 合并为单个 DataFrame（qlib 格式要求）
    # qlib 格式: datetime as index, columns as features
    if 'close' in fields:
        # 以 close 的 index 为基准
        df = fields['close'].copy()
        df.columns = ['$close']

        # 添加其他字段
        field_mapping = {
            'open': '$open',
            'high': '$high',
            'low': '$low',
            'volume': '$volume',
            'amount': '$amount',
            'vwap': '$vwap',
            'open_interest': '$open_interest'
        }

        for src, dst in field_mapping.items():
            if src in fields:
                df[dst] = fields[src].iloc[:, 0]

        # 保存为 qlib 格式
        # qlib 期望的格式: instruments/freq/instrument.csv
        for freq in FREQS:
            output_dir = INSTRUMENTS_DIR / freq
            df.to_csv(output_dir / f"{instrument}.csv")

        print(f"  保存: {len(df)} 行 × {len(df.columns)} 列")
        print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")

# 创建 calenders 目录并复制交易日历
calendars_dir = QLIB_DIR / "calendars"
calendars_dir.mkdir(parents=True, exist_ok=True)

# 复制日历文件
shutil.copy(DATA_DIR / "calendars" / "day.txt", calendars_dir / "day.txt")
shutil.copy(DATA_DIR / "calendars" / "1min.txt", calendars_dir / "1min.txt")

print(f"\n数据准备完成！")
print(f"qlib 数据目录: {QLIB_DIR}")
print(f"支持的频率: {', '.join(FREQS)}")
