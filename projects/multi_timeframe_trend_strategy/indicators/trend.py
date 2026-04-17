"""
趋势指标模块
实现EMA60大周期趋势判断
"""

import pandas as pd
import numpy as np
from typing import Literal, List
from dataclasses import dataclass


@dataclass
class TrendState:
    """趋势状态数据结构"""
    direction: Literal['bullish', 'bearish', 'neutral']  # 趋势方向
    strength: Literal['strong', 'weak', 'none']  # 趋势强度
    since: pd.Timestamp  # 趋势开始时间
    ema_60: float  # EMA(60)值
    price: float  # 当前价格
    is_confirmed: bool  # 是否已确认
    bars_in_trend: int  # 当前趋势持续K线数


class TrendIndicator:
    """趋势指标类"""

    def __init__(self, ema_period: int = 60, direction_threshold: float = 0.0):
        """
        初始化趋势指标

        Args:
            ema_period: EMA周期（默认60）
            direction_threshold: 方向确认阈值（%）
        """
        self.ema_period = ema_period
        self.direction_threshold = direction_threshold

        # 当前趋势状态
        self.current_trend: TrendState = None
        self.trend_history: List[TrendState] = []

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势指标

        Args:
            df: K线数据，必须包含 close 列

        Returns:
            pd.DataFrame: 添加了趋势指标的数据框
        """
        df = df.copy()

        # 1. 计算EMA60
        df[f'ema_{self.ema_period}'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()

        # 2. 确定趋势方向
        df['trend_direction'] = self._determine_trend_direction(df)

        # 3. 计算趋势强度（基于价格距离EMA的距离）
        df['trend_strength'] = self._calculate_trend_strength(df)

        # 4. 更新趋势状态机
        self._update_trend_state_machine(df)

        return df

    def _determine_trend_direction(self, df: pd.DataFrame) -> pd.Series:
        """
        确定趋势方向（基于EMA60）

        Args:
            df: K线数据（已计算ema_60）

        Returns:
            pd.Series: 趋势方向（'bullish', 'bearish', 'neutral'）
        """
        # 基于价格与EMA60的位置确定趋势
        trend_direction = pd.Series('neutral', index=df.index, dtype=str)

        # 计算价格距离EMA60的距离（百分比）
        price_ema_distance = (df['close'] - df[f'ema_{self.ema_period}']) / df['close']

        # 上升趋势：价格 > EMA60
        bullish = df['close'] > df[f'ema_{self.ema_period}']
        # 同时需要距离阈值（可选）
        if self.direction_threshold > 0:
            bullish = bullish & (price_ema_distance > self.direction_threshold)
        trend_direction[bullish] = 'bullish'

        # 下降趋势：价格 < EMA60
        bearish = df['close'] < df[f'ema_{self.ema_period}']
        # 同时需要距离阈值（可选）
        if self.direction_threshold > 0:
            bearish = bearish & (price_ema_distance < -self.direction_threshold)
        trend_direction[bearish] = 'bearish'

        return trend_direction

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        计算趋势强度（基于价格距离EMA的距离）

        Args:
            df: K线数据（已计算ema_60）

        Returns:
            pd.Series: 趋势强度（'strong', 'weak', 'none'）
        """
        strength = pd.Series('none', index=df.index, dtype=str)

        # 计算价格距离EMA60的绝对距离（百分比）
        price_ema_distance_pct = abs(df['close'] - df[f'ema_{self.ema_period}']) / df['close']

        # 强趋势：差距 > 2%
        strong = price_ema_distance_pct > 0.02
        strength[strong] = 'strong'

        # 弱趋势：差距在0.5%-2%之间
        weak = (price_ema_distance_pct > 0.005) & (price_ema_distance_pct <= 0.02)
        strength[weak] = 'weak'

        return strength

    def _update_trend_state_machine(self, df: pd.DataFrame):
        """
        更新趋势状态机

        Args:
            df: K线数据（已计算所有趋势指标）
        """
        self.trend_history = []

        for i in range(len(df)):
            direction = df['trend_direction'].iloc[i]
            strength = df['trend_strength'].iloc[i]

            # 计算当前趋势持续K线数
            if self.current_trend is None or direction != self.current_trend.direction:
                bars_in_trend = 0
            else:
                bars_in_trend = self.current_trend.bars_in_trend + 1

            # 创建趋势状态
            trend_state = TrendState(
                direction=direction,
                strength=strength,
                since=df.index[i],
                ema_60=df[f'ema_{self.ema_period}'].iloc[i],
                price=df['close'].iloc[i],
                is_confirmed=True,  # EMA方向不需要额外确认
                bars_in_trend=bars_in_trend
            )

            # 更新当前趋势
            if self.current_trend is None or direction != self.current_trend.direction:
                # 新趋势开始
                self.current_trend = trend_state
            else:
                # 趋势继续
                self.current_trend.bars_in_trend = bars_in_trend

            self.trend_history.append(trend_state)

    def get_current_trend(self) -> TrendState:
        """
        获取当前趋势状态

        Returns:
            TrendState: 当前趋势状态
        """
        return self.current_trend

    def get_trend_at(self, timestamp: pd.Timestamp) -> TrendState:
        """
        获取指定时间点的趋势状态

        Args:
            timestamp: 时间戳

        Returns:
            TrendState: 趋势状态
        """
        # 查找最接近的时间点
        for i, trend_state in enumerate(self.trend_history):
            if trend_state.since >= timestamp:
                return trend_state

        # 如果没找到，返回最后一个
        if self.trend_history:
            return self.trend_history[-1]
        return None


if __name__ == "__main__":
    # 测试趋势指标
    import sys
    from pathlib import Path

    # 添加data路径
    sys.path.append(str(Path(__file__).parent.parent / 'data'))

    from provider import DataProvider

    # 加载日线数据
    provider = DataProvider(
        instrument="HC8888.XSGE",
        start_date="2023-01-01",
        end_date="2024-12-31",
        frequency="1D"
    )

    df = provider.load_data()

    # 计算趋势指标
    indicator = TrendIndicator(
        ema_period=60,
        direction_threshold=0.0
    )

    df = indicator.calculate(df)

    # 显示结果
    print("\n趋势指标计算结果:")
    print(df[['close', f'ema_{indicator.ema_period}', 'trend_direction',
              'trend_strength']].tail(10))

    # 当前趋势状态
    current_trend = indicator.get_current_trend()
    print(f"\n当前趋势:")
    print(f"  方向: {current_trend.direction}")
    print(f"  强度: {current_trend.strength}")
    print(f"  确认: {current_trend.is_confirmed}")
    print(f"  持续K线数: {current_trend.bars_in_trend}")
