#!/usr/bin/env python3
"""
将 CSV 格式的数据转换为 qlib 格式（正确版本）

qlib 期望的数据格式:
instruments/
  freq/                  # 频率（如 1min, day）
    feature_name/        # 特征名称（如 $close, $open）
      instrument.csv     # 每个合约一个文件
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil

# 配置路径
DATA_DIR = Path("/mnt/d/quant/qlib_data")
QLIB_DIR = Path("/mnt/d/quant/qlib_backtest/data/qlib_data_v2")
FREQS = ["1min", "day"]

# 清理并重建 qlib 数据目录
if QLIB_DIR.exists():
    shutil.rmtree(QLIB_DIR)
QLIB_DIR.mkdir(parents=True, exist_ok=True)

# 读取合约列表
with open(DATA_DIR / "instruments" / "all.txt", 'r') as f:
    instruments = [line.strip() for line in f if line.strip()]

print(f"合约数量: {len(instruments)}")

# qlib 期望的特征映射
field_mapping = {
    'open': '$open',
    'high': '$high',
    'low': '$low',
    'close': '$close',
    'volume': '$volume',
    'amount': '$amount',
    'vwap': '$vwap',
    'open_interest': '$open_interest'
}

# 为每个频率创建目录
for freq in FREQS:
    # 为每个特征创建目录
    for field in field_mapping.values():
        feature_dir = QLIB_DIR / "instruments" / freq / field
        feature_dir.mkdir(parents=True, exist_ok=True)

# 处理每个合约
for instrument in instruments:
    print(f"\n处理合约: {instrument}")

    instrument_dir = DATA_DIR / "instruments" / instrument
    if not instrument_dir.exists():
        print(f"  跳过: 数据目录不存在")
        continue

    # 读取各个字段
    fields_data = {}
    for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'open_interest']:
        field_file = instrument_dir / f"{field}.csv"
        if field_file.exists():
            df = pd.read_csv(field_file, index_col=0, parse_dates=True)
            fields_data[field] = df
            print(f"  {field}: {len(df)} 行")

    # 为每个频率和特征保存数据
    for freq in FREQS:
        for src_field, dst_feature in field_mapping.items():
            if src_field in fields_data:
                # qlib 期望的格式: 保存为单列，索引是 datetime
                df = fields_data[src_field].copy()

                # 如果是 1min 频率，直接使用
                # 如果是 day 频率，需要重采样
                if freq == "day":
                    # 使用每天 09:31 的收盘价（或者用 last/bfill）
                    df_daily = df.resample('D').last()
                    df_daily = df_daily.dropna()
                    df_to_save = df_daily
                else:
                    df_to_save = df

                # 保存文件
                output_file = QLIB_DIR / "instruments" / freq / dst_feature / f"{instrument}.csv"
                df_to_save.to_csv(output_file)

        print(f"  保存频率 {freq}")

# 创建 calendars 目录并复制交易日历
calendars_dir = QLIB_DIR / "calendars"
calendars_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(DATA_DIR / "calendars" / "day.txt", calendars_dir / "day.txt")
shutil.copy(DATA_DIR / "calendars" / "1min.txt", calendars_dir / "1min.txt")

print(f"\n数据准备完成！")
print(f"qlib 数据目录: {QLIB_DIR}")

# 显示目录结构
print(f"\n=== 目录结构 ===")
for root, dirs, files in os.walk(QLIB_DIR):
    level = root.replace(str(QLIB_DIR), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:3]:
        print(f'{subindent}{file}')
    if len(files) > 3:
        print(f'{subindent}... and {len(files) - 3} more files')
