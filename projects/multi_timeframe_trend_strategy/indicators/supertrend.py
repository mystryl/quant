"""
Supertrend指标模块
实现ATR和Supertrend动态止损计算
"""

import pandas as pd
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class SuperTrendValue:
    """Supertrend值数据结构"""
    trend: int  # 1=看涨, -1=看跌
    supertrend: float  # Supertrend线值
    atr: float  # ATR值


class SuperTrendIndicator:
    """Supertrend指标类"""

    def __init__(self, atr_period: int = 10, atr_factor: float = 3.0):
        """
        初始化Supertrend指标

        Args:
            atr_period: ATR周期（默认10）
            atr_factor: ATR因子（默认3.0）
        """
        self.atr_period = atr_period
        self.atr_factor = atr_factor

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Supertrend指标

        Args:
            df: K线数据，必须包含 high, low, close 列

        Returns:
            pd.DataFrame: 添加了Supertrend指标的数据框
        """
        df = df.copy()

        # 1. 计算ATR
        df['atr'] = self._calculate_atr(df, self.atr_period)

        # 2. 计算Supertrend上线和下线
        df['supertrend_upper'] = df['close'] + (self.atr_factor * df['atr'])
        df['supertrend_lower'] = df['close'] - (self.atr_factor * df['atr'])

        # 3. 计算Supertrend线和趋势（使用loc避免链式赋值错误）
        supertrend = np.zeros(len(df), dtype=float)
        supertrend_trend = np.zeros(len(df), dtype=int)

        for i in range(1, len(df)):
            if i == 1:
                # 第一根K线使用下线
                supertrend[i] = df['supertrend_lower'].iloc[i]
                supertrend_trend[i] = -1
            else:
                # 获取上一根K线的Supertrend值
                prev_supertrend = supertrend[i-1]
                prev_trend = supertrend_trend[i-1]

                # 根据上一根趋势选择上线或下线
                if prev_trend == 1:
                    # 之前是看涨，使用下线
                    current_supertrend = df['supertrend_lower'].iloc[i]
                else:
                    # 之前是看跌，使用上线
                    current_supertrend = df['supertrend_upper'].iloc[i]

                # Supertrend线只能向一个方向移动
                if prev_trend == 1:
                    # 看涨趋势，Supertrend只能上升
                    supertrend[i] = max(prev_supertrend, current_supertrend)
                else:
                    # 看跌趋势，Supertrend只能下降
                    supertrend[i] = min(prev_supertrend, current_supertrend)

                # 检测趋势反转
                current_close = df['close'].iloc[i]

                if prev_trend == 1 and current_close < supertrend[i]:
                    # 看涨反转为看跌
                    supertrend_trend[i] = -1
                elif prev_trend == -1 and current_close > supertrend[i]:
                    # 看跌反转为看涨
                    supertrend_trend[i] = 1
                else:
                    # 趋势继续
                    supertrend_trend[i] = prev_trend

        # 将结果赋值到DataFrame
        df['supertrend'] = supertrend
        df['supertrend_trend'] = supertrend_trend

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        计算平均真实波幅（ATR）

        Args:
            df: K线数据
            period: ATR周期

        Returns:
            pd.Series: ATR值
        """
        # 计算真实波幅（TR）
        df['tr'] = df['high'] - df['low']

        # 考虑前一日的收盘价
        df['prev_close'] = df['close'].shift(1)

        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['prev_close']),
                abs(df['low'] - df['prev_close'])
            )
        )

        # 计算ATR（使用EMA平滑）
        # ATR = EMA(TR, period)
        multiplier = 2 / (period + 1)
        atr = pd.Series(index=df.index, dtype=float)

        # 第一根K线的ATR
        atr.iloc[0] = df['tr'].iloc[0]

        # 滚动计算
        for i in range(1, len(df)):
            atr.iloc[i] = (df['tr'].iloc[i] * multiplier) + (atr.iloc[i-1] * (1 - multiplier))

        return atr

    def get_stop_loss(self, df: pd.DataFrame, index: int, position_type: str) -> Tuple[float, float]:
        """
        获取指定K线位置的止损价格

        Args:
            df: K线数据（已计算supertrend）
            index: K线索引
            position_type: 持仓类型（'long' or 'short'）

        Returns:
            Tuple[float, float]: (止损价格, ATR值）
        """
        if index >= len(df):
            raise IndexError(f"索引 {index} 超出范围")

        # 获取Supertrend值
        supertrend = df['supertrend'].iloc[index]
        atr = df['atr'].iloc[index]

        if position_type == 'long':
            # 看涨持仓：止损在Supertrend下线
            stop_loss = df['supertrend_lower'].iloc[index]
        elif position_type == 'short':
            # 看跌持仓：止损在Supertrend上线
            stop_loss = df['supertrend_upper'].iloc[index]
        else:
            raise ValueError(f"未知持仓类型: {position_type}")

        return stop_loss, atr

    def is_trend_reversal(self, df: pd.DataFrame, index: int, current_trend: int) -> bool:
        """
        检测趋势是否反转

        Args:
            df: K线数据（已计算supertrend_trend）
            index: K线索引
            current_trend: 当前趋势（1=看涨, -1=看跌）

        Returns:
            bool: 是否反转
        """
        if index >= len(df):
            return False

        current_supertrend_trend = df['supertrend_trend'].iloc[index]

        # 如果当前Supertrend趋势与持仓趋势相反，说明反转
        return current_supertrend_trend != current_trend


if __name__ == "__main__":
    # 测试Supertrend指标
    import sys
    from pathlib import Path

    # 添加data路径
    sys.path.append(str(Path(__file__).parent.parent / 'data'))

    from provider import DataProvider

    # 加载数据
    provider = DataProvider(
        instrument="HC8888.XSGE",
        start_date="2024-01-01",
        end_date="2024-12-31",
        frequency="1H"
    )

    df = provider.load_data()

    # 计算Supertrend
    indicator = SuperTrendIndicator(
        atr_period=10,
        atr_factor=3.0
    )

    df = indicator.calculate(df)

    # 显示结果
    print("\nSupertrend指标计算结果:")
    print(df[['close', 'atr', 'supertrend', 'supertrend_trend']].tail(10))

    # 测试止损获取
    stop_loss, atr = indicator.get_stop_loss(df, len(df)-1, 'long')
    print(f"\n最后一根K线:")
    print(f"  收盘价: {df['close'].iloc[-1]:.2f}")
    print(f"  ATR: {atr:.2f}")
    print(f"  Supertrend: {df['supertrend'].iloc[-1]:.2f}")
    print(f"  看涨止损: {stop_loss:.2f}")
