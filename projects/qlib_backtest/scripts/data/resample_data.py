#!/usr/bin/env python3
"""
数据重采样脚本：将 1分钟数据重采样到 5min, 15min, 60min
"""
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# 配置路径
DATA_DIR = Path("/mnt/d/quant/qlib_data")
QLIB_DIR = Path("/mnt/d/quant/qlib_backtest/qlib_data_multi_freq")

# 目标频率
TARGET_FREQS = ["5min", "15min", "60min"]

# 字段映射（原始字段 → qlib 特征名）
FIELD_MAPPING = {
    'open': '$open',
    'high': '$high',
    'low': '$low',
    'close': '$close',
    'volume': '$volume',
    'amount': '$amount',
    'vwap': '$vwap',
    'open_interest': '$open_interest'
}

# 重采样规则
RESAMPLE_RULES = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'amount': 'sum',
    'open_interest': 'last'
}

print("="*60)
print("数据重采样脚本")
print("="*60)

# 读取合约列表
with open(DATA_DIR / "instruments" / "all.txt", 'r') as f:
    instruments = [line.strip() for line in f if line.strip()]

print(f"\n合约数量: {len(instruments)}")

# 处理每个合约
for instrument in instruments:
    print(f"\n{'='*60}")
    print(f"处理合约: {instrument}")
    print(f"{'='*60}")

    instrument_dir = DATA_DIR / "instruments" / instrument
    if not instrument_dir.exists():
        print(f"跳过: 数据目录不存在")
        continue

    # 读取 1 分钟数据
    data_1min = {}
    for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'open_interest']:
        field_file = instrument_dir / f"{field}.csv"
        if field_file.exists():
            df = pd.read_csv(field_file, index_col=0, parse_dates=True)
            data_1min[field] = df.iloc[:, 0]  # 获取第一列数据
            print(f"  读取 {field}: {len(df)} 行")

    # 合并为单个 DataFrame（1分钟）
    df_1min = pd.DataFrame(data_1min)
    df_1min = df_1min.sort_index()
    print(f"\n  1分钟数据: {len(df_1min)} 行")
    print(f"  时间范围: {df_1min.index.min()} ~ {df_1min.index.max()}")

    # 对每个目标频率进行重采样
    for freq in TARGET_FREQS:
        print(f"\n  重采样到 {freq}...")

        # 创建重采样规则字典
        resample_dict = {}
        for field, rule in RESAMPLE_RULES.items():
            if field in df_1min.columns:
                resample_dict[field] = rule

        # 重采样
        df_resampled = df_1min.resample(freq).agg(resample_dict)

        # 重新计算 vwap（如果存在）
        if 'amount' in df_resampled.columns and 'volume' in df_resampled.columns:
            df_resampled['vwap'] = df_resampled['amount'] / df_resampled['volume']
            # 处理除零情况
            df_resampled['vwap'] = df_resampled['vwap'].replace([np.inf, -np.inf], np.nan)
            df_resampled['vwap'] = df_resampled['vwap'].fillna(method='ffill')

        # 删除 NaN 行（非交易时间段）
        df_resampled = df_resampled.dropna(subset=['close'])

        print(f"    {freq} 数据: {len(df_resampled)} 行")
        print(f"    时间范围: {df_resampled.index.min()} ~ {df_resampled.index.max()}")

        # 保存为 qlib 格式
        freq_dir = QLIB_DIR / "instruments" / freq
        for field, feature_name in FIELD_MAPPING.items():
            if field in df_resampled.columns:
                feature_dir = freq_dir / feature_name
                feature_dir.mkdir(parents=True, exist_ok=True)

                # 保存单列数据
                df_feature = df_resampled[[field]].copy()
                df_feature.to_csv(feature_dir / f"{instrument}.csv")

        print(f"    保存到: {QLIB_DIR / 'instruments' / freq}")

# 复制 1分钟数据和日历
print(f"\n{'='*60}")
print("复制 1分钟数据和日历...")
print(f"{'='*60}")

# 复制 1min 数据
source_1min = DATA_DIR / "instruments" / instrument
target_1min = QLIB_DIR / "instruments" / "1min"

for feature_name in FIELD_MAPPING.values():
    feature_dir = target_1min / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)
    for field_file in source_1min.glob("*.csv"):
        shutil.copy(field_file, feature_dir / field_file.name)

# 生成不同频率的日历
calendars_dir = QLIB_DIR / "calendars"
calendars_dir.mkdir(parents=True, exist_ok=True)

# 复制原始日历
shutil.copy(DATA_DIR / "calendars" / "1min.txt", calendars_dir / "1min.txt")
shutil.copy(DATA_DIR / "calendars" / "day.txt", calendars_dir / "day.txt")

# 生成其他频率的日历
cal_1min = pd.read_csv(DATA_DIR / "calendars" / "1min.txt", header=None, names=['datetime'])
cal_1min['datetime'] = pd.to_datetime(cal_1min['datetime'])

for freq in TARGET_FREQS:
    # 获取重采样后的时间点
    cal_resampled = cal_1min.set_index('datetime').resample(freq).first().dropna()
    cal_resampled.reset_index(inplace=True)

    # 保存日历
    cal_file = calendars_dir / f"{freq}.txt"
    cal_resampled['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').to_csv(cal_file, index=False, header=False)

    print(f"  生成 {freq} 日历: {len(cal_resampled)} 个时间点")

print(f"\n{'='*60}")
print("重采样完成！")
print(f"{'='*60}")
print(f"数据目录: {QLIB_DIR}")

# 显示目录结构
print(f"\n目录结构:")
import os
for root, dirs, files in os.walk(QLIB_DIR):
    level = root.replace(str(QLIB_DIR), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level == 2:  # 只显示到频率级别
        for freq_dir in dirs:
            freq_path = Path(root) / freq_dir
            features = sorted([d for d in os.listdir(freq_path) if os.path.isdir(freq_path / d)])[:3]
            subindent = ' ' * 2 * (level + 1)
            print(f'{subindent}{freq_dir}/ (包含特征: {", ".join(features[:3])}{"..." if len(features) > 3 else ""})')
        break
