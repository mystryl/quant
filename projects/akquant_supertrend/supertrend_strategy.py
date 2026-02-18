"""
SuperTrend 策略实现（AKQuant 版本）。

基于 ATR (Average True Range) 的趋势跟踪策略。
"""

import numpy as np
import pandas as pd
from typing import List
from akquant import Strategy, Bar


def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       period: int = 10, multiplier: float = 3.0) -> float:
    """
    计算 SuperTrend 指标线。

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: ATR 周期
        multiplier: ATR 倍数

    Returns:
        SuperTrend 线的最后一个值
    """
    # 确保是 numpy 数组
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)

    # 计算 ATR
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]

    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    atr = np.zeros(len(high))
    atr[0] = tr[0]

    for i in range(1, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    # 计算基本上下轨
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # 计算最终上下轨
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(high)):
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

    # 计算 SuperTrend 线
    supertrend = np.zeros(len(high))

    supertrend[0] = final_upper[0]

    for i in range(1, len(high)):
        if supertrend[i-1] == final_upper[i-1]:
            if close[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = final_lower[i]
        else:
            if close[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]

    return supertrend[-1]


class SuperTrendStrategy(Strategy):
    """
    SuperTrend 趋势跟踪策略。

    策略逻辑：
    - 当收盘价 > SuperTrend 时，做多
    - 当收盘价 < SuperTrend 时，平多仓
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        初始化策略。

        Args:
            period: ATR 周期（默认 10）
            multiplier: ATR 倍数（默认 3.0）
        """
        super().__init__()
        self.period = period
        self.multiplier = multiplier

        # 维护历史数据的列表
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.supertrend: float = None
        self.last_close_above: bool = None  # 上一次收盘价是否在 SuperTrend 上方

    def on_start(self) -> None:
        """策略启动时的初始化。"""
        print(f"SuperTrend 策略启动: period={self.period}, multiplier={self.multiplier}")

    def on_bar(self, bar: Bar) -> None:
        """
        处理每个 bar。

        Args:
            bar: 当前 bar 数据
        """
        # 收集数据
        self.highs.append(bar.high)
        self.lows.append(bar.low)
        self.closes.append(bar.close)

        # 需要至少 period 个数据点才能计算 SuperTrend
        if len(self.closes) < self.period:
            return

        # 计算最近的 SuperTrend 值
        recent_high = np.array(self.highs[-self.period:])
        recent_low = np.array(self.lows[-self.period:])
        recent_close = np.array(self.closes[-self.period:])

        self.supertrend = calculate_supertrend(recent_high, recent_low, recent_close,
                                           self.period, self.multiplier)

        # 判断当前状态
        close_above = bar.close > self.supertrend

        # 首次初始化
        if self.last_close_above is None:
            self.last_close_above = close_above
            return

        # 检测突破
        if close_above and not self.last_close_above:
            # 收盘价从下方突破到上方 -> 买入
            position = self.get_position(bar.symbol)
            if position == 0:
                self.buy(symbol=bar.symbol, quantity=1)
                print(f"[{bar.timestamp_str}] 突破上轨: 买入 @ {bar.close:.2f}, ST={self.supertrend:.2f}")

        elif not close_above and self.last_close_above:
            # 收盘价从上方跌破到下方 -> 平仓
            position = self.get_position(bar.symbol)
            if position > 0:
                self.close_position(symbol=bar.symbol)
                print(f"[{bar.timestamp_str}] 跌破下轨: 平仓 @ {bar.close:.2f}, ST={self.supertrend:.2f}")

        # 更新状态
        self.last_close_above = close_above
