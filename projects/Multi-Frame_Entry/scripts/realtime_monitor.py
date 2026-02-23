#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时趋势监控脚本

功能：
1. 获取最新60分钟K线数据
2. 计算特征并使用window20模型预测趋势方向
3. 检测最近10根K线的趋势变化
4. 输出监控报告

支持品种：螺纹钢(RB0)、热卷(HC0)、铁矿石(I0)、黄金(AU0)、郑棉(CF0)

使用方法：
    # 监控单个品种
    python scripts/realtime_monitor.py --symbol RB0

    # 监控所有品种
    python scripts/realtime_monitor.py --all

    # 自定义参数
    python scripts/realtime_monitor.py --symbol RB0 --bars 150 --lookback 15
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from datetime import datetime
import warnings
import argparse
import sys

warnings.filterwarnings('ignore')

# 添加项目路径（使用相对路径）
# 获取项目根目录：当前文件的父目录的父目录
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from features.trend_features import TrendFeatures
from scripts.realtime_data_fetcher import RealtimeDataFetcher
from scripts.trend_change_detector import (
    detect_recent_signal_changes_from_df,
    format_change_events,
    get_current_signal
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置
DATA_FETCH_CONFIG = {
    'period': '60min',      # 60分钟K线
    'bars': 100,            # 获取100根K线
    'lookback_bars': 10,    # 检测最近10根K线
}

MODEL_DIR = project_root / 'models' / 'rolling_3month'
THRESHOLD = 0.5  # 二分类概率阈值

# 品种配置（与数据获取器保持一致）
SYMBOL_CONFIG = {
    'RB0': {'full_code': 'HC8888.XSGE', 'name': '螺纹钢', 'model_code': 'HC888'},
    'HC0': {'full_code': 'HC8888.XSGE', 'name': '热卷', 'model_code': 'HC888'},
    'I0': {'full_code': 'I8888.XDCE', 'name': '铁矿石', 'model_code': 'I888'},
    'AU0': {'full_code': 'AU8888.XSGE', 'name': '黄金', 'model_code': 'AU888'},
    'CF0': {'full_code': 'CF8888.XZCE', 'name': '郑棉', 'model_code': 'CF888'},
}


def load_model(symbol: str):
    """
    加载window20二分类模型

    Args:
        symbol: 品种代码（如 'RB0'）

    Returns:
        模型数据字典
    """
    model_code = SYMBOL_CONFIG[symbol]['model_code']
    full_code = SYMBOL_CONFIG[symbol]['full_code']
    model_file = MODEL_DIR / f'{full_code}_window20.pkl'

    logger.info(f"加载模型: {model_file}")

    if not model_file.exists():
        logger.error(f"模型文件不存在: {model_file}")
        raise FileNotFoundError(f"模型文件不存在: {model_file}")

    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    logger.info(f"  ✓ 模型加载成功")
    logger.info(f"  特征数量: {len(model_data['features'])}")

    return model_data


def predict_signals(
    df_features: pd.DataFrame,
    model_data: dict
) -> pd.DataFrame:
    """
    预测三分类信号（上涨/下跌/震荡）

    Args:
        df_features: 特征DataFrame
        model_data: 二分类模型数据

    Returns:
        结果DataFrame，包含三分类信号和概率
    """
    model = model_data['model']
    features = model_data['features']

    # 提取特征
    X = df_features[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # 预测概率
    binary_proba = model.predict_proba(X)[:, 1]  # P(有趋势)

    # 使用MACD直方图判断方向
    macd_hist = df_features['macd_histogram'].values

    # 组合预测
    signals = []
    p_range = []  # P(震荡)
    p_up = []     # P(上涨)
    p_down = []   # P(下跌)

    for i in range(len(df_features)):
        p_trend = binary_proba[i]  # P(有趋势)
        p_no_trend = 1 - p_trend  # P(震荡)

        if p_trend < THRESHOLD:
            # 震荡
            signals.append('震荡')
            p_range.append(p_no_trend)
            p_up.append(0)
            p_down.append(0)
        else:
            # 有趋势，用MACD判断方向
            if macd_hist[i] > 0:
                signals.append('上涨')
                p_range.append(0)
                p_up.append(p_trend)
                p_down.append(0)
            else:
                signals.append('下跌')
                p_range.append(0)
                p_up.append(0)
                p_down.append(p_trend)

    # 创建结果DataFrame
    df_result = pd.DataFrame({
        'signal': signals,
        'P(震荡)': p_range,
        'P(上涨)': p_up,
        'P(下跌)': p_down,
        'Close': df_features['close'].values,
    }, index=df_features.index)

    return df_result


def monitor_symbol(
    symbol: str,
    bars: int = 100,
    lookback_bars: int = 10
) -> dict:
    """
    监控单个品种

    Args:
        symbol: 品种代码（如 'RB0'）
        bars: 获取K线数量
        lookback_bars: 回溯K线数量

    Returns:
        监控结果字典
    """
    symbol_name = SYMBOL_CONFIG[symbol]['name']

    logger.info(f"\n{'='*80}")
    logger.info(f"监控品种: {symbol} ({symbol_name})")
    logger.info(f"{'='*80}")

    try:
        # 1. 获取最新数据（优先使用本地数据）
        fetcher = RealtimeDataFetcher(preferred_source='local')
        df = fetcher.fetch_latest_data(symbol=symbol, bars=bars, period='60min')

        if df is None or len(df) < 60:
            logger.error(f"数据不足，无法进行预测")
            return None

        # 2. 计算特征
        logger.info("计算技术指标特征...")
        calculator = TrendFeatures(
            ema_short=20,
            ema_long=60,
            adx_period=14,
            atr_period=14
        )
        df_features = calculator.compute_all_features(df, shift=True)

        # 删除包含NaN的行（特征计算导致的）
        df_features = df_features.dropna()

        if len(df_features) < lookback_bars:
            logger.warning(f"有效特征数据不足 {lookback_bars} 根K线")
            return None

        logger.info(f"  ✓ 有效特征数据: {len(df_features)} 条")

        # 3. 加载模型
        model_data = load_model(symbol)

        # 4. 预测信号
        logger.info("预测趋势信号...")
        df_result = predict_signals(df_features, model_data)

        # 5. 获取当前状态
        current = get_current_signal(df_result)

        # 6. 检测最近N根K线的变化
        logger.info(f"检测最近{lookback_bars}根K线的趋势变化...")
        changes = detect_recent_signal_changes_from_df(
            df_result,
            lookback_bars=lookback_bars,
            price_col='Close'
        )

        logger.info(f"  检测到 {len(changes)} 个变化事件")

        # 返回结果
        return {
            'symbol': symbol,
            'symbol_name': symbol_name,
            'current': current,
            'changes': changes,
            'df_result': df_result,
            'success': True,
        }

    except Exception as e:
        logger.error(f"监控 {symbol} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'symbol': symbol,
            'symbol_name': symbol_name,
            'success': False,
            'error': str(e),
        }


def format_monitor_report(result: dict) -> str:
    """
    格式化监控报告

    Args:
        result: 监控结果字典

    Returns:
        格式化的报告文本
    """
    if not result.get('success'):
        return f"\n❌ 监控失败: {result.get('symbol')} - {result.get('error', '未知错误')}"

    symbol = result['symbol']
    symbol_name = result['symbol_name']
    current = result['current']
    changes = result['changes']

    lines = []
    lines.append("=" * 60)
    lines.append(f"实时趋势监控报告")
    lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"品种: {symbol_name} ({symbol})")
    lines.append("=" * 60)
    lines.append("")

    # 当前状态
    lines.append("📊 当前状态:")
    lines.append(f"  信号: {current['signal']}")
    lines.append(f"  价格: {current['price']:.2f}" if current['price'] else "  价格: N/A")

    # 添加概率信息
    if 'P(震荡)' in current:
        lines.append(f"  P(上涨): {current['P(上涨)']:.2f}, "
                    f"P(震荡): {current['P(震荡)']:.2f}, "
                    f"P(下跌): {current['P(下跌)']:.2f}")

    lines.append("")

    # 趋势变化
    lines.append("🔔 最近10根K线内的趋势变化:")
    if changes:
        lines.append("")
        lines.append(format_change_events(changes, symbol_name=symbol_name))
    else:
        lines.append("  无趋势变化")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时趋势监控工具')
    parser.add_argument('--symbol', type=str, help='品种代码（如 RB0, HC0, I0, AU0, CF0）')
    parser.add_argument('--all', action='store_true', help='监控所有品种')
    parser.add_argument('--bars', type=int, default=100, help='获取K线数量（默认100）')
    parser.add_argument('--lookback', type=int, default=10, help='回溯K线数量（默认10）')

    args = parser.parse_args()

    # 验证参数
    if not args.symbol and not args.all:
        parser.print_help()
        print("\n错误: 请指定 --symbol 或 --all")
        return

    if args.symbol and args.symbol not in SYMBOL_CONFIG:
        print(f"错误: 不支持的品种 '{args.symbol}'")
        print(f"支持的品种: {', '.join(SYMBOL_CONFIG.keys())}")
        return

    logger.info("="*80)
    logger.info("实时趋势监控系统")
    logger.info("="*80)
    logger.info(f"数据获取配置: K线数量={args.bars}, 回溯周期={args.lookback}")
    logger.info(f"模型目录: {MODEL_DIR}")
    logger.info(f"信号阈值: {THRESHOLD}")
    logger.info("="*80 + "\n")

    # 确定要监控的品种
    if args.all:
        symbols_to_monitor = list(SYMBOL_CONFIG.keys())
    else:
        symbols_to_monitor = [args.symbol]

    # 监控各品种
    results = []
    for symbol in symbols_to_monitor:
        result = monitor_symbol(
            symbol=symbol,
            bars=args.bars,
            lookback_bars=args.lookback
        )
        if result:
            results.append(result)

    # 输出报告
    print("\n" + "="*80)
    print("监控报告汇总")
    print("="*80 + "\n")

    for result in results:
        report = format_monitor_report(result)
        print(report)

    # 汇总统计
    successful = sum(1 for r in results if r.get('success'))
    failed = len(results) - successful

    print(f"\n{'='*80}")
    print(f"监控完成: 成功 {successful} 个, 失败 {failed} 个")
    print(f"{'='*80}\n")

    return results


if __name__ == '__main__':
    main()
