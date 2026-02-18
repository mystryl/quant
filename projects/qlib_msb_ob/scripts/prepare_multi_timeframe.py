#!/usr/bin/env python3
"""
准备多级别数据：15分钟和5分钟
从60分钟数据重采样
"""
import pandas as pd
import numpy as np
from pathlib import Path

def resample_data(df, freq):
    """重采样数据"""
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return df.resample(freq).agg(agg_dict).dropna()

def prepare_multi_timeframe():
    """准备多级别数据"""
    data_dir = Path(__file__).parent.parent / "data"

    years = [2023, 2024, 2025]

    for year in years:
        print(f"\n处理 {year} 年数据...")

        # 加载60分钟数据
        file_60m = data_dir / f"RB9999_{year}_60min.csv"
        if not file_60m.exists():
            print(f"  跳过 {year}：60分钟数据不存在")
            continue

        df = pd.read_csv(file_60m, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)

        print(f"  原始60分钟数据: {len(df)} 行")

        # 重采样为15分钟
        print("  重采样为15分钟...")
        df_15m = resample_data(df, '15min')
        file_15m = data_dir / f"RB9999_{year}_15min.csv"
        df_15m.reset_index().to_csv(file_15m, index=False)
        print(f"  15分钟数据: {len(df_15m)} 行")

        # 重采样为5分钟
        print("  重采样为5分钟...")
        df_5m = resample_data(df, '5min')
        file_5m = data_dir / f"RB9999_{year}_5min.csv"
        df_5m.reset_index().to_csv(file_5m, index=False)
        print(f"  5分钟数据: {len(df_5m)} 行")

    print("\n✅ 多级别数据准备完成！")

if __name__ == "__main__":
    prepare_multi_timeframe()
