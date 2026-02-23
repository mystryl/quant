#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监控系统测试脚本

功能：
1. 测试数据获取模块
2. 测试趋势变化检测模块
3. 测试完整监控流程
4. 使用历史数据验证检测准确性
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# 添加项目路径
project_root = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry')
sys.path.insert(0, str(project_root))

from scripts.realtime_data_fetcher import RealtimeDataFetcher
from scripts.trend_change_detector import (
    detect_recent_signal_changes_from_df,
    classify_change_type,
    format_change_events
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_fetcher():
    """测试数据获取模块"""
    print("\n" + "="*80)
    print("测试1: 数据获取模块")
    print("="*80)

    fetcher = RealtimeDataFetcher(preferred_source='local')

    # 测试获取热卷数据
    df = fetcher.fetch_latest_data(symbol='HC0', bars=100)

    if df is not None:
        print(f"\n✓ 数据获取成功")
        print(f"  数据形状: {df.shape}")
        print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")
        print(f"  列名: {list(df.columns)}")
        print(f"\n最新5根K线:")
        print(df[['open', 'high', 'low', 'close', 'volume']].tail(5))
        return True
    else:
        print("\n✗ 数据获取失败")
        return False


def test_change_detector():
    """测试趋势变化检测模块"""
    print("\n" + "="*80)
    print("测试2: 趋势变化检测模块")
    print("="*80)

    # 构造测试数据
    test_signals = ['震荡', '震荡', '上涨', '上涨', '下跌', '下跌', '震荡', '上涨']
    test_timestamps = pd.date_range('2026-02-21 09:00:00', periods=8, freq='H')
    test_prices = [3500 + i*10 for i in range(8)]

    print(f"\n测试信号序列:")
    for i, (sig, ts, price) in enumerate(zip(test_signals, test_timestamps, test_prices)):
        print(f"  {i+1}. {ts.strftime('%H:%M')} | {sig} | {price:.2f}")

    # 测试变化类型分类
    print(f"\n变化类型分类测试:")
    test_cases = [
        ('震荡', '上涨', '趋势启动'),
        ('震荡', '下跌', '趋势启动'),
        ('上涨', '下跌', '趋势反转'),
        ('下跌', '上涨', '趋势反转'),
        ('上涨', '震荡', '趋势结束'),
        ('下跌', '震荡', '趋势结束'),
        ('震荡', '震荡', '震荡延续'),
    ]

    all_passed = True
    for from_sig, to_sig, expected in test_cases:
        result = classify_change_type(from_sig, to_sig)
        passed = result == expected
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"  {status} {from_sig} → {to_sig}: {result} (期望: {expected})")

    # 测试变化检测
    print(f"\n最近5根K线的信号变化检测:")
    from scripts.trend_change_detector import detect_recent_signal_changes

    changes = detect_recent_signal_changes(
        signals=test_signals,
        timestamps=test_timestamps,
        prices=test_prices,
        lookback_bars=5
    )

    print(f"  检测到 {len(changes)} 个变化事件")
    for change in changes:
        print(f"    - {change['time'].strftime('%H:%M')} | "
              f"{change['type']} | "
              f"{change['from_signal']} → {change['to_signal']}")

    return all_passed


def test_integration():
    """测试完整监控流程"""
    print("\n" + "="*80)
    print("测试3: 完整监控流程")
    print("="*80)

    try:
        # 导入监控脚本
        from scripts.realtime_monitor import monitor_symbol

        # 监控热卷
        result = monitor_symbol(symbol='HC0', bars=100, lookback_bars=10)

        if result and result.get('success'):
            print(f"\n✓ 监控成功")
            print(f"  品种: {result['symbol_name']}")
            print(f"  当前信号: {result['current']['signal']}")
            print(f"  当前价格: {result['current']['price']:.2f}")
            print(f"  检测到 {len(result['changes'])} 个趋势变化")
            return True
        else:
            print(f"\n✗ 监控失败")
            return False

    except Exception as e:
        print(f"\n✗ 监控出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_historical_data():
    """使用历史数据测试监控准确性"""
    print("\n" + "="*80)
    print("测试4: 历史数据验证")
    print("="*80)

    try:
        # 使用2026年1月的数据测试
        from scripts.realtime_monitor import monitor_symbol

        print("\n监控热卷 (HC0) 2026年1-2月数据...")
        result = monitor_symbol(symbol='HC0', bars=150, lookback_bars=20)

        if result and result.get('success'):
            df_result = result['df_result']

            print(f"\n✓ 历史数据加载成功")
            print(f"  总K线数: {len(df_result)}")
            print(f"  时间范围: {df_result.index[0]} 到 {df_result.index[-1]}")

            # 统计信号分布
            signal_counts = df_result['signal'].value_counts()
            print(f"\n信号分布:")
            for signal, count in signal_counts.items():
                pct = count / len(df_result) * 100
                print(f"  {signal}: {count} ({pct:.1f}%)")

            # 检测所有变化点（不限制回溯）
            print(f"\n检测所有趋势变化点...")
            all_changes = detect_recent_signal_changes_from_df(
                df_result,
                lookback_bars=len(df_result),
                price_col='Close'
            )

            print(f"  共检测到 {len(all_changes)} 个趋势变化点")

            # 按类型统计
            change_types = {}
            for change in all_changes:
                ctype = change['type']
                change_types[ctype] = change_types.get(ctype, 0) + 1

            print(f"\n变化类型分布:")
            for ctype, count in change_types.items():
                print(f"  {ctype}: {count}")

            return True
        else:
            print(f"\n✗ 历史数据测试失败")
            return False

    except Exception as e:
        print(f"\n✗ 历史数据测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("实时监控系统测试")
    print("="*80)

    results = []

    # 运行测试
    results.append(("数据获取模块", test_data_fetcher()))
    results.append(("趋势变化检测模块", test_change_detector()))
    results.append(("完整监控流程", test_integration()))
    results.append(("历史数据验证", test_with_historical_data()))

    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit(main())
