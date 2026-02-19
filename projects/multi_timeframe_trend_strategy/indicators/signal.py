"""
信号识别模块
实现中周期（1小时）VWAP5和VWAP60金叉死叉入场
"""

import pandas as pd
import numpy as np
from typing import List, Literal
from dataclasses import dataclass


@dataclass
class EntrySignal:
    """入场信号数据结构"""
    id: str
    timestamp: pd.Timestamp
    bar_index: int
    trend_direction: Literal['bullish', 'bearish']  # 大周期趋势方向（日线EMA60）
    entry_type: Literal['golden_cross', 'death_cross']  # 入场类型（金叉/死叉）
    entry_price: float
    vwap_short: float  # VWAP5
    vwap_long: float  # VWAP60
    rsi: float
    volume: float
    volume_avg: float
    volume_multiplier: float
    quality_score: float  # 信号质量分数（0-100）
    is_confirmed: bool


class SignalGenerator:
    """信号生成器类"""

    def __init__(self, vwap_short_period: int = 5, vwap_long_period: int = 60,
                 confirmation_bars: int = 3, use_rsi: bool = True,
                 rsi_period: int = 14, rsi_overbought: float = 70,
                 rsi_oversold: float = 30):
        """
        初始化信号生成器

        Args:
            vwap_short_period: VWAP短周期（默认5）
            vwap_long_period: VWAP长周期（默认60）
            confirmation_bars: 交叉确认所需K线数（默认3）
            use_rsi: 是否使用RSI过滤
            rsi_period: RSI周期
            rsi_overbought: RSI超买阈值
            rsi_oversold: RSI超卖阈值
        """
        self.vwap_short_period = vwap_short_period
        self.vwap_long_period = vwap_long_period
        self.confirmation_bars = confirmation_bars
        self.use_rsi = use_rsi
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def generate_signals(self, df: pd.DataFrame, daily_trend: pd.Series) -> List[EntrySignal]:
        """
        生成入场信号列表

        Args:
            df: 1小时K线数据，必须包含：
                - open, high, low, close, volume
            daily_trend: 日线趋势数据（大周期EMA60方向）

        Returns:
            List[EntrySignal]: 入场信号列表
        """
        signals = []

        # 计算辅助指标
        df = self._calculate_indicators(df)

        # 将daily_trend合并到小时线
        # 需要按时间对齐
        df = self._align_daily_trend(df, daily_trend)

        for i in range(100, len(df)):  # 从第100根K线开始（保证有足够历史数据）
            # 只在确认趋势中寻找信号
            daily_direction = df['daily_trend_direction'].iloc[i]

            if daily_direction == 'neutral':
                continue

            # 检测VWAP金叉死叉
            if df['vwap_cross'].iloc[i] != 0:
                # 金叉：VWAP5上穿VWAP60
                if df['vwap_cross'].iloc[i] == 1:
                    # 只有在大周期也是看涨时才入场
                    if daily_direction == 'bullish':
                        signal = self._detect_golden_cross(df, i)
                        if signal:
                            signals.append(signal)

                # 死叉：VWAP5下穿VWAP60
                elif df['vwap_cross'].iloc[i] == -1:
                    # 只有在大周期也是看跌时才入场
                    if daily_direction == 'bearish':
                        signal = self._detect_death_cross(df, i)
                        if signal:
                            signals.append(signal)

        print(f"[SignalGenerator] 生成了 {len(signals)} 个入场信号")
        return signals

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算辅助指标

        Args:
            df: K线数据

        Returns:
            pd.DataFrame: 添加了辅助指标的数据框
        """
        df = df.copy()

        # 1. 计算VWAP5和VWAP60
        df['vwap_short'] = self._calculate_vwap(df, self.vwap_short_period)
        df['vwap_long'] = self._calculate_vwap(df, self.vwap_long_period)

        # 2. 检测VWAP交叉
        df['vwap_cross'] = self._detect_vwap_cross(df)

        # 3. 计算RSI（如果启用）
        if self.use_rsi:
            df['rsi'] = self._calculate_rsi(df, self.rsi_period)
        else:
            df['rsi'] = 50.0  # 默认中值

        # 4. 计算成交量平均值
        df['volume_avg'] = df['volume'].rolling(window=20, min_periods=1).mean()

        # 5. 计算成交量倍数
        df['volume_multiplier'] = df['volume'] / df['volume_avg']

        return df

    def _calculate_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        计算成交量加权平均价格（VWAP）

        Args:
            df: K线数据
            period: 会话长度（根K线数）

        Returns:
            pd.Series: VWAP值
        """
        # 计算典型价格 (TP)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # 计算成交量加权价格
        df['volume_weighted_price'] = df['typical_price'] * df['volume']

        # 滚动计算VWAP（基于会话长度）
        vwap = pd.Series(index=df.index, dtype=float)

        # 滚动窗口计算
        rolling_sum_vwp = df['volume_weighted_price'].rolling(window=period, min_periods=1).sum()
        rolling_sum_vol = df['volume'].rolling(window=period, min_periods=1).sum()

        # 避免除以零
        vwap = rolling_sum_vwp / rolling_sum_vol.replace(0, np.nan)

        return vwap

    def _detect_vwap_cross(self, df: pd.DataFrame) -> pd.Series:
        """
        检测VWAP交叉信号

        Args:
            df: K线数据（已计算vwap_short和vwap_long）

        Returns:
            pd.Series: 交叉信号（1=金叉, -1=死叉, 0=无）
        """
        cross_signals = pd.Series(0, index=df.index)

        # 向上交叉（金叉）
        cross_up = (df['vwap_short'] > df['vwap_long']) & \
                  (df['vwap_short'].shift(1) <= df['vwap_long'].shift(1))
        cross_signals[cross_up] = 1

        # 向下交叉（死叉）
        cross_down = (df['vwap_short'] < df['vwap_long']) & \
                   (df['vwap_short'].shift(1) >= df['vwap_long'].shift(1))
        cross_signals[cross_down] = -1

        return cross_signals

    def _detect_golden_cross(self, df: pd.DataFrame, index: int) -> EntrySignal:
        """
        检测看涨入场信号（金叉）

        Args:
            df: K线数据
            index: 当前K线索引

        Returns:
            EntrySignal: 入场信号（如果无则返回None）
        """
        current_price = df['close'].iloc[index]
        vwap_short = df['vwap_short'].iloc[index]
        vwap_long = df['vwap_long'].iloc[index]
        rsi = df['rsi'].iloc[index]
        volume = df['volume'].iloc[index]
        volume_avg = df['volume_avg'].iloc[index]
        volume_multiplier = df['volume_multiplier'].iloc[index]

        # RSI过滤（不能超买）
        if self.use_rsi and rsi >= self.rsi_overbought:
            return None

        # 成交量过滤（必须放大）
        if volume_multiplier < 1.2:
            return None

        # 入场价格：VWAP60（支撑位）
        entry_price = vwap_long

        # 计算信号质量分数
        quality_score = self._calculate_signal_quality(
            rsi=rsi,
            volume_multiplier=volume_multiplier,
            vwap_distance_pct=abs(current_price - vwap_long) / current_price,
            trend_strength='strong'  # 金叉通常信号强
        )

        signal = EntrySignal(
            id=f"golden_cross_{index}",
            timestamp=df.index[index],
            bar_index=index,
            trend_direction='bullish',
            entry_type='golden_cross',
            entry_price=entry_price,
            vwap_short=vwap_short,
            vwap_long=vwap_long,
            rsi=rsi,
            volume=volume,
            volume_avg=volume_avg,
            volume_multiplier=volume_multiplier,
            quality_score=quality_score,
            is_confirmed=True
        )

        return signal

    def _detect_death_cross(self, df: pd.DataFrame, index: int) -> EntrySignal:
        """
        检测看跌入场信号（死叉）

        Args:
            df: K线数据
            index: 当前K线索引

        Returns:
            EntrySignal: 入场信号（如果无则返回None）
        """
        current_price = df['close'].iloc[index]
        vwap_short = df['vwap_short'].iloc[index]
        vwap_long = df['vwap_long'].iloc[index]
        rsi = df['rsi'].iloc[index]
        volume = df['volume'].iloc[index]
        volume_avg = df['volume_avg'].iloc[index]
        volume_multiplier = df['volume_multiplier'].iloc[index]

        # RSI过滤（不能超卖）
        if self.use_rsi and rsi <= self.rsi_oversold:
            return None

        # 成交量过滤（必须放大）
        if volume_multiplier < 1.2:
            return None

        # 入场价格：VWAP60（阻力位）
        entry_price = vwap_long

        # 计算信号质量分数
        quality_score = self._calculate_signal_quality(
            rsi=rsi,
            volume_multiplier=volume_multiplier,
            vwap_distance_pct=abs(current_price - vwap_long) / current_price,
            trend_strength='strong'  # 死叉通常信号强
        )

        signal = EntrySignal(
            id=f"death_cross_{index}",
            timestamp=df.index[index],
            bar_index=index,
            trend_direction='bearish',
            entry_type='death_cross',
            entry_price=entry_price,
            vwap_short=vwap_short,
            vwap_long=vwap_long,
            rsi=rsi,
            volume=volume,
            volume_avg=volume_avg,
            volume_multiplier=volume_multiplier,
            quality_score=quality_score,
            is_confirmed=True
        )

        return signal

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        计算相对强弱指标（RSI）

        Args:
            df: K线数据
            period: RSI周期

        Returns:
            pd.Series: RSI值
        """
        # 计算价格变化
        delta = df['close'].diff()

        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 计算平均上涨和平均下跌
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        # 计算相对强度（RS）
        rs = avg_gain / avg_loss.replace(0, np.nan)

        # 计算RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_signal_quality(self, rsi: float, volume_multiplier: float,
                           vwap_distance_pct: float, trend_strength: str) -> float:
        """
        计算信号质量分数（0-100）

        Args:
            rsi: RSI值
            volume_multiplier: 成交量倍数
            vwap_distance_pct: 价格距离VWAP的百分比
            trend_strength: 趋势强度

        Returns:
            float: 质量分数（0-100）
        """
        score = 0

        # RSI得分（最佳区域：40-60）
        if 40 <= rsi <= 60:
            score += 30
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            score += 20
        else:
            score += 10

        # 成交量得分
        if volume_multiplier >= 2.0:
            score += 30
        elif volume_multiplier >= 1.5:
            score += 20
        elif volume_multiplier >= 1.2:
            score += 10

        # VWAP距离得分（距离越近越好）
        if vwap_distance_pct <= 0.005:  # 0.5%
            score += 20
        elif vwap_distance_pct <= 0.01:  # 1%
            score += 10
        else:
            score += 5

        # 趋势强度得分
        if trend_strength == 'strong':
            score += 20
        elif trend_strength == 'weak':
            score += 10

        return min(100, score)

    def _align_daily_trend(self, hourly_df: pd.DataFrame, daily_trend: pd.Series) -> pd.DataFrame:
        """
        将日线趋势对齐到小时线

        Args:
            hourly_df: 小时线数据
            daily_trend: 日线趋势数据（Series，包含trend_direction列）

        Returns:
            pd.DataFrame: 添加了日线趋势的小时线数据
        """
        # 创建新列
        hourly_df['daily_trend_direction'] = 'neutral'

        # 重新索引daily_trend
        # daily_trend是一个Series，包含trend_direction列的值
        # 对于每根小时线，找到对应的日线趋势（前向填充）

        # 使用numpy数组来快速查找
        daily_indices = daily_trend.index.values
        daily_values = daily_trend.values

        # 对于每个小时线时间戳，找到对应的日线索引
        from bisect import bisect_right
        for i, timestamp in enumerate(hourly_df.index):
            # 找到最后一个小于等于当前时间戳的日线索引
            idx = bisect_right(daily_indices, timestamp) - 1

            # 确保索引在有效范围内
            if 0 <= idx < len(daily_values):
                hourly_df['daily_trend_direction'].iloc[i] = daily_values[idx]
            else:
                hourly_df['daily_trend_direction'].iloc[i] = 'neutral'

        return hourly_df


if __name__ == "__main__":
    # 测试信号生成器
    import sys
    from pathlib import Path

    # 添加路径
    sys.path.append(str(Path(__file__).parent.parent / 'data'))

    from provider import DataProvider

    # 加载小时线数据
    provider = DataProvider(
        instrument="HC8888.XSGE",
        start_date="2023-01-01",
        end_date="2024-12-31",
        frequency="1H"
    )

    hourly_df = provider.load_data()

    # 加载日线数据（用于大周期趋势）
    daily_provider = DataProvider(
        instrument="HC8888.XSGE",
        start_date="2023-01-01",
        end_date="2024-12-31",
        frequency="1D"
    )

    daily_df = daily_provider.load_data()

    # 计算日线趋势
    from trend import TrendIndicator
    trend_indicator = TrendIndicator(ema_period=60)
    daily_df = trend_indicator.calculate(daily_df)

    # 生成入场信号
    signal_generator = SignalGenerator()
    signals = signal_generator.generate_signals(hourly_df, daily_df['trend_direction'])

    # 显示信号
    print(f"\n生成了 {len(signals)} 个入场信号:")
    for i, signal in enumerate(signals[:5]):  # 显示前5个
        print(f"\n信号 {i+1}:")
        print(f"  时间: {signal.timestamp}")
        print(f"  方向: {signal.trend_direction}")
        print(f"  类型: {signal.entry_type}")
        print(f"  入场价: {signal.entry_price:.2f}")
        print(f"  RSI: {signal.rsi:.2f}")
        print(f"  量倍: {signal.volume_multiplier:.2f}")
        print(f"  质量: {signal.quality_score:.1f}")
