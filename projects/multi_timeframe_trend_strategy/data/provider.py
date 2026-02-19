"""
数据提供者模块
使用ParquetDataProvider加载商品指数合约数据
"""

import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

# 导入本地ParquetDataProvider
from .parquet_provider import ParquetDataProvider


class DataProvider:
    """数据提供者类"""

    def __init__(self, instrument: str, start_date: str, end_date: str, frequency: str = '1H'):
        """
        初始化数据提供者

        Args:
            instrument: 合约代码，如 "HC8888.XSGE"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            frequency: 数据频率，如 "1H", "4H", "1D"
        """
        self.instrument = instrument
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency

        # 使用ParquetDataProvider
        self.provider = ParquetDataProvider()

    def load_data(self, fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        加载合约数据

        Args:
            fields: 需要加载的字段列表，如 ["open", "high", "low", "close", "volume"]
                   如果为None，加载所有字段

        Returns:
            pd.DataFrame: K线数据，包含OHLCV字段
        """
        try:
            print(f"[DataProvider] 加载合约 {self.instrument} 数据...")
            print(f"  时间范围: {self.start_date} 到 {self.end_date}")
            print(f"  频率: {self.frequency}")

            # 使用ParquetDataProvider加载数据
            df = self.provider.get_data(
                self.instrument,
                self.start_date,
                self.end_date,
                fields=fields or ["open", "high", "low", "close", "volume"],
                format='parquet'
            )

            if df is None or len(df) == 0:
                raise ValueError(f"合约 {self.instrument} 没有数据或数据为空")

            # 按频率重采样
            if self.frequency != '1H':
                df = self._resample_data(df, self.frequency)

            print(f"[DataProvider] 数据加载成功: {len(df)} 条记录")
            print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")

            return df

        except Exception as e:
            print(f"[DataProvider] 数据加载失败: {str(e)}")
            raise

    def _resample_data(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        重采样数据到指定频率

        Args:
            df: 原始数据
            frequency: 目标频率

        Returns:
            pd.DataFrame: 重采样后的数据
        """
        freq_map = {
            '1D': '1D',
            '4H': '4H',
            '1H': '1H',
            '30min': '30T',
            '15min': '15T',
            '5min': '5T',
            '1min': '1T'
        }

        if frequency not in freq_map:
            raise ValueError(f"不支持的数据频率: {frequency}")

        resample_rule = freq_map[frequency]

        # 重采样OHLCV数据
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # 只重采样存在的列
        agg_dict = {col: agg_dict[col] for col in agg_dict if col in df.columns}

        df_resampled = df.resample(resample_rule).agg(agg_dict).dropna()

        print(f"[DataProvider] 数据重采样: {self.frequency} -> {len(df_resampled)} 条记录")

        return df_resampled

    @staticmethod
    def list_instruments() -> List[str]:
        """
        列出所有可用的商品指数合约

        Returns:
            List[str]: 合约代码列表
        """
        from .parquet_provider import ParquetDataProvider
        provider = ParquetDataProvider()
        instruments = provider.list_instruments()
        return instruments


if __name__ == "__main__":
    # 测试数据提供者
    provider = DataProvider(
        instrument="HC8888.XSGE",
        start_date="2024-01-01",
        end_date="2024-12-31",
        frequency="1H"
    )

    # 加载数据
    df = provider.load_data()

    # 显示数据摘要
    print("\n数据摘要:")
    print(df.describe())
    print("\n前5条:")
    print(df.head())
