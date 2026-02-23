#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势变化检测模块

检测最近N根K线内的趋势信号变化
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def classify_change_type(from_signal: str, to_signal: str) -> str:
    """
    分类趋势变化类型

    Args:
        from_signal: 原信号（'上涨', '下跌', '震荡'）
        to_signal: 新信号（'上涨', '下跌', '震荡'）

    Returns:
        变化类型:
        - '趋势启动': 震荡 → 上涨/下跌
        - '趋势反转': 上涨 → 下跌 或 下跌 → 上涨
        - '趋势结束': 上涨/下跌 → 震荡
        - '震荡延续': 震荡 → 震荡（通常不输出）
        - '无变化': 信号相同
    """
    # 震荡 → 上涨/下跌：趋势启动
    if from_signal == '震荡':
        if to_signal in ['上涨', '下跌']:
            return '趋势启动'

    # 上涨/下跌 → 震荡：趋势结束
    if to_signal == '震荡':
        if from_signal in ['上涨', '下跌']:
            return '趋势结束'

    # 上涨 → 下跌 或 下跌 → 上涨：趋势反转
    if from_signal == '上涨' and to_signal == '下跌':
        return '趋势反转'
    if from_signal == '下跌' and to_signal == '上涨':
        return '趋势反转'

    # 震荡 → 震荡
    if from_signal == '震荡' and to_signal == '震荡':
        return '震荡延续'

    # 其他情况
    return '信号转换'


def detect_recent_signal_changes(
    signals: List[str],
    timestamps: List[pd.Timestamp],
    prices: Optional[List[float]] = None,
    lookback_bars: int = 10
) -> List[Dict]:
    """
    检测最近N根K线内的信号变化

    Args:
        signals: 信号列表（如 ['震荡', '震荡', '上涨', '上涨', '下跌', ...]）
        timestamps: 对应的时间戳列表
        prices: 对应的价格列表（可选，用于输出价格信息）
        lookback_bars: 回溯K线数量（默认10根）

    Returns:
        变化事件列表，每个事件包含:
        {
            'time': timestamp,
            'from_signal': '震荡',
            'to_signal': '上涨',
            'position': -2,  # 距离最新的K线数（负数）
            'type': '趋势启动',  # 或 '趋势反转', '趋势结束'
            'price': 3580.00,  # 变化时的价格（可选）
            'from_price': 3575.00,  # 变化前价格（可选）
        }
    """
    if len(signals) != len(timestamps):
        raise ValueError("signals和timestamps长度必须相同")

    if prices is not None and len(prices) != len(signals):
        raise ValueError("prices长度必须与signals相同")

    if len(signals) < 2:
        return []

    changes = []

    # 只检查最近lookback_bars根K线
    # 例如，lookback_bars=10，则检查最后10个信号之间的变化
    start_idx = max(0, len(signals) - lookback_bars - 1)

    for i in range(start_idx, len(signals) - 1):
        from_signal = signals[i]
        to_signal = signals[i + 1]

        # 只关注信号变化的情况
        if from_signal != to_signal:
            change_type = classify_change_type(from_signal, to_signal)

            # 计算距离最新K线的位置（负数表示过去）
            # 例如，-1表示上一根K线，-2表示两根前
            position = -(len(signals) - 1 - i)

            # 构建变化事件
            change_event = {
                'time': timestamps[i + 1],  # 新信号的时间
                'from_signal': from_signal,
                'to_signal': to_signal,
                'position': position,
                'type': change_type,
            }

            # 添加价格信息（如果有）
            if prices is not None:
                change_event['price'] = prices[i + 1]  # 新信号时的价格
                change_event['from_price'] = prices[i]  # 原信号时的价格

            # 过滤掉"震荡延续"和"无变化"
            if change_type not in ['震荡延续', '无变化']:
                changes.append(change_event)

    return changes


def detect_recent_signal_changes_from_df(
    df_result: pd.DataFrame,
    lookback_bars: int = 10,
    price_col: str = 'Close'
) -> List[Dict]:
    """
    从DataFrame中检测最近N根K线的信号变化

    Args:
        df_result: 包含'signal'和价格列的DataFrame
        lookback_bars: 回溯K线数量
        price_col: 价格列名（默认'Close'）

    Returns:
        变化事件列表
    """
    # 只保留最近lookback_bars + 1根K线（需要多1根来检测变化）
    df_recent = df_result.tail(lookback_bars + 1).copy()

    signals = df_recent['signal'].tolist()
    timestamps = df_recent.index.tolist()
    prices = df_recent[price_col].tolist() if price_col in df_recent.columns else None

    changes = detect_recent_signal_changes(
        signals=signals,
        timestamps=timestamps,
        prices=prices,
        lookback_bars=lookback_bars
    )

    return changes


def format_change_events(changes: List[Dict], symbol_name: str = '') -> str:
    """
    格式化变化事件为可读文本

    Args:
        changes: 变化事件列表
        symbol_name: 品种名称（可选）

    Returns:
        格式化的文本
    """
    if not changes:
        return "  无趋势变化"

    lines = []

    for change in changes:
        position = change['position']
        time = change['time']

        # 计算位置描述
        if position == -1:
            position_desc = "上一根K线"
        else:
            position_desc = f"{abs(position)}根K线前"

        # 时间格式化
        time_str = time.strftime('%H:%M:%S') if hasattr(time, 'strftime') else str(time)

        # 变化描述
        change_desc = f"{change['from_signal']} → {change['to_signal']}"

        # 价格信息
        price_info = ""
        if 'price' in change:
            price = change['price']
            from_price = change.get('from_price', price)
            price_change = price - from_price
            price_change_pct = (price_change / from_price) * 100 if from_price != 0 else 0

            if price_change >= 0:
                price_info = f" (+{price_change:.2f}, +{price_change_pct:.2f}%)"
            else:
                price_info = f" ({price_change:.2f}, {price_change_pct:.2f}%)"

        lines.append(f"  [{position_desc}] {time_str}")
        lines.append(f"  类型: {change['type']}")
        lines.append(f"  变化: {change_desc}")
        if price_info:
            lines.append(f"  价格: {change['price']:.2f}{price_info}")
        lines.append("")

    return "\n".join(lines)


def get_current_signal(
    df_result: pd.DataFrame,
    price_col: str = 'Close'
) -> Dict:
    """
    获取当前（最新）信号状态

    Args:
        df_result: 包含'signal'和价格列的DataFrame
        price_col: 价格列名

    Returns:
        当前状态字典
    """
    if len(df_result) == 0:
        return {
            'signal': '未知',
            'price': None,
            'datetime': None,
        }

    last_row = df_result.iloc[-1]

    result = {
        'signal': last_row['signal'],
        'price': last_row[price_col] if price_col in last_row.index else None,
        'datetime': df_result.index[-1],
    }

    # 添加概率信息（如果有）
    proba_cols = ['P(震荡)', 'P(上涨)', 'P(下跌)']
    for col in proba_cols:
        if col in last_row.index:
            result[col] = last_row[col]

    return result


if __name__ == '__main__':
    """测试趋势变化检测功能"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("="*60)
    print("测试: 趋势变化检测")
    print("="*60)

    # 构造测试数据
    test_signals = ['震荡', '震荡', '上涨', '上涨', '下跌', '下跌', '震荡', '上涨']
    test_timestamps = pd.date_range('2026-02-21 09:00:00', periods=8, freq='H')
    test_prices = [3500 + i*10 for i in range(8)]

    print(f"\n测试信号序列:")
    for i, (sig, ts, price) in enumerate(zip(test_signals, test_timestamps, test_prices)):
        print(f"  {ts} | {sig} | {price:.2f}")

    # 检测最近5根K线的变化
    print(f"\n检测最近5根K线的信号变化:")
    changes = detect_recent_signal_changes(
        signals=test_signals,
        timestamps=test_timestamps,
        prices=test_prices,
        lookback_bars=5
    )

    print(f"\n检测到 {len(changes)} 个变化事件:")
    for change in changes:
        print(f"  {change['time']} | {change['type']} | {change['from_signal']} → {change['to_signal']}")

    # 格式化输出
    print(f"\n格式化输出:")
    print(format_change_events(changes, symbol_name='测试品种'))

    # 测试DataFrame接口
    print(f"\n{'='*60}")
    print("测试: DataFrame接口")
    print("="*60)

    df_test = pd.DataFrame({
        'signal': test_signals,
        'Close': test_prices,
    }, index=test_timestamps)

    changes_df = detect_recent_signal_changes_from_df(df_test, lookback_bars=5)
    print(f"\n从DataFrame检测到 {len(changes_df)} 个变化事件")

    current = get_current_signal(df_test)
    print(f"\n当前信号: {current['signal']}, 价格: {current['price']:.2f}")
