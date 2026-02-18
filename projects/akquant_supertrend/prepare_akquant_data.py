"""
准备 AKQuant 需要的 RB9999 数据。
从 Qlib 格式合并并重采样为 15 分钟和 60 分钟级别。
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_qlib_to_dataframe(instrument_path: str) -> pd.DataFrame:
    """
    从 Qlib 格式（分开的 CSV 文件）加载并合并为单个 DataFrame。

    Args:
        instrument_path: 期货品种路径，如 '/mnt/d/quant/data/qlib/instruments/RB9999.XSGE'

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    path = Path(instrument_path)

    # 读取各字段文件
    df_open = pd.read_csv(path / "open.csv", index_col=0, parse_dates=True)
    df_high = pd.read_csv(path / "high.csv", index_col=0, parse_dates=True)
    df_low = pd.read_csv(path / "low.csv", index_col=0, parse_dates=True)
    df_close = pd.read_csv(path / "close.csv", index_col=0, parse_dates=True)
    df_volume = pd.read_csv(path / "volume.csv", index_col=0, parse_dates=True)

    # 合并为单表
    df = pd.concat([
        df_open['open'],
        df_high['high'],
        df_low['low'],
        df_close['close'],
        df_volume['volume']
    ], axis=1)

    # 确保索引是 datetime 类型
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'

    return df

def resample_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    重采样数据为指定频率。

    Args:
        df: 原始 DataFrame（1 分钟级别）
        freq: 重采样频率，如 '15min' 或 '60min'

    Returns:
        重采样后的 DataFrame
    """
    # 定义聚合规则
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # 重采样
    df_resampled = df.resample(freq).agg(agg_dict).dropna()

    return df_resampled

def filter_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """过滤指定年份的数据。"""
    return df[df.index.year == year]

def prepare_akquant_format(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    转换为 AKQuant 需要的格式。

    Args:
        df: 原始 DataFrame
        symbol: 品种代码

    Returns:
        AKQuant 格式的 DataFrame
    """
    df = df.copy()
    df.reset_index(inplace=True)
    df.rename(columns={'date': 'timestamp'}, inplace=True)
    df['symbol'] = symbol

    # 确保列顺序
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

    return df

def main():
    # 配置
    instrument_path = '/mnt/d/quant/data/qlib/instruments/RB9999.XSGE'
    symbol = 'RB9999.XSGE'
    year = 2024

    # 输出目录
    output_dir = Path('/mnt/d/quant/projects/akquant_supertrend/data')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    df = load_qlib_to_dataframe(instrument_path)
    print(f"原始数据: {len(df)} 行，从 {df.index[0]} 到 {df.index[-1]}")

    print(f"\n过滤 {year} 年数据...")
    df_year = filter_by_year(df, year)
    print(f"过滤后: {len(df_year)} 行")

    # 重采样为不同频率
    freqs = {
        '15min': '15min',
        '60min': '60min'
    }

    for name, freq in freqs.items():
        print(f"\n重采样为 {name}...")
        df_resampled = resample_data(df_year, freq)
        print(f"重采样后: {len(df_resampled)} 行")

        # 转换为 AKQuant 格式
        df_akquant = prepare_akquant_format(df_resampled, symbol)

        # 保存
        output_file = output_dir / f'RB9999_{year}_{name}.csv'
        df_akquant.to_csv(output_file, index=False)
        print(f"已保存: {output_file}")

    print("\n数据准备完成！")

if __name__ == '__main__':
    main()
