#!/usr/bin/env python3
"""
MSB+OB 策略技术指标模块

严格对应 Pine 源码中的指标计算
"""
import pandas as pd
import numpy as np


def calculate_momentum_z(close: pd.Series, window: int = 50) -> pd.Series:
    """
    计算动量 Z-Score

    Pine 源码 (第142-146行):
        float priceChange = ta.change(close)
        float avgChange   = ta.sma(priceChange, 50)
        float stdChange   = ta.stdev(priceChange, 50)
        float momentumZ   = (priceChange - avgChange) / stdChange

    Args:
        close: 收盘价序列
        window: 计算窗口，默认50（与Pine源码一致）

    Returns:
        动量Z-Score序列
    """
    # ta.change(close) = 当前价格 - 前一根K线价格
    price_change = close.diff()

    # ta.sma(priceChange, 50) = 简单移动平均
    avg_change = price_change.rolling(window=window).mean()

    # ta.stdev(priceChange, 50) = 标准差
    std_change = price_change.rolling(window=window).std()

    # Z-Score = (price_change - avg) / std
    momentum_z = (price_change - avg_change) / std_change

    return momentum_z


def calculate_volume_percentile(volume: pd.Series, window: int = 100) -> pd.Series:
    """
    计算成交量百分位

    Pine 源码 (第147行):
        float volPercent  = ta.percentrank(volume, 100)

    Args:
        volume: 成交量序列
        window: 计算窗口，默认100（与Pine源码一致）

    Returns:
        成交量百分位序列 (0-100)
    """
    # ta.percentrank(volume, 100) = 在过去100根K线中的百分位排名
    vol_percentile = volume.rolling(window=window).rank(pct=True) * 100

    return vol_percentile


def detect_pivot_points(high: pd.Series, low: pd.Series, pivot_len: int = 7):
    """
    检测枢轴点（高点和低点）

    Pine 源码 (第150-163行):
        float ph = ta.pivothigh(high, pivotLenInput, pivotLenInput)
        float pl = ta.pivotlow(low, pivotLenInput, pivotLenInput)

    Args:
        high: 最高价序列
        low: 最低价序列
        pivot_len: 枢轴点回看期，默认7（与Pine源码一致）

    Returns:
        (pivot_high, pivot_low): 枢轴高点和低点序列
    """
    # 初始化
    pivot_high = pd.Series(np.nan, index=high.index)
    pivot_low = pd.Series(np.nan, index=low.index)

    # ta.pivothigh(high, pivot_len, pivot_len)
    # 检测高点：当前K线的high是左右各pivot_len根K线的最高值
    for i in range(pivot_len, len(high) - pivot_len):
        high_window = high.iloc[i - pivot_len:i + pivot_len + 1]
        if high.iloc[i] == high_window.max():
            pivot_high.iloc[i] = high.iloc[i]

    # ta.pivotlow(low, pivot_len, pivot_len)
    # 检测低点：当前K线的low是左右各pivot_len根K线的最低值
    for i in range(pivot_len, len(low) - pivot_len):
        low_window = low.iloc[i - pivot_len:i + pivot_len + 1]
        if low.iloc[i] == low_window.min():
            pivot_low.iloc[i] = low.iloc[i]

    return pivot_high, pivot_low


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算平均真实波幅(ATR)

    用于止损计算

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: ATR周期，默认14

    Returns:
        ATR序列
    """
    prev_close = close.shift(1)

    # 真实波幅 = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR = TR的简单移动平均
    atr = tr.rolling(window=period).mean()

    return atr
