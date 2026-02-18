#!/usr/bin/env python3
"""
从Qlib 1分钟数据重采样为5分钟、15分钟、60分钟
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_qlib_data(instrument_path):
    """从Qlib格式加载数据"""
    path = Path(instrument_path)

    # 读取各字段文件
    df_open = pd.read_csv(path / "open.csv", index_col=0, parse_dates=True)
    df_high = pd.read_csv(path / "high.csv", index_col=0, parse_dates=True)
    df_low = pd.read_csv(path / "low.csv", index_col=0, parse_dates=True)
    df_close = pd.read_csv(path / "close.csv", index_col=0, parse_dates=True)
    df_volume = pd.read_csv(path / "volume.csv", index_col=0, parse_dates=True)

    # 合并
    df = pd.concat([
        df_open['open'],
        df_high['high'],
        df_low['low'],
        df_close['close'],
        df_volume['volume']
    ], axis=1)

    df.index.name = 'datetime'
    return df.sort_index()

def resample_and_save(df, freq, year, output_dir):
    """重采样并保存"""
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # 过滤年份
    df_year = df[df.index.year == year]

    # 重采样
    df_resampled = df_year.resample(freq).agg(agg_dict).dropna()

    # 保存
    output_file = output_dir / f"RB9999_{year}_{freq.replace('min', 'min')}.csv"
    df_resampled.reset_index().to_csv(output_file, index=False)

    return len(df_resampled)

def main():
    instrument_path = r"D:\quant\data\qlib\instruments\RB9999.XSGE"
    output_dir = Path(r"D:\quant\projects\qlib_msb_ob\data")
    output_dir.mkdir(parents=True, exist_ok=True)

    years = [2023, 2024, 2025]
    freqs = ['5min', '15min', '60min']

    print("加载Qlib 1分钟数据...")
    df = load_qlib_data(instrument_path)
    print(f"总数据量: {len(df)} 行")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")

    for year in years:
        print(f"\n{'='*60}")
        print(f"处理 {year} 年")
        print(f"{'='*60}")

        df_year = df[df.index.year == year]
        print(f"1分钟数据: {len(df_year)} 行")

        for freq in freqs:
            rows = resample_and_save(df, freq, year, output_dir)
            print(f"  {freq:>6}: {rows:5} 行")

    print("\n完成！")

if __name__ == "__main__":
    main()
