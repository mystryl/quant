#!/usr/bin/env python3
"""
Qlib Supertrend 指标实现

基于 ssquant 的 supertrend 指标复刻，适配 qlib 数据格式
"""
import pandas as pd
import numpy as np
from pathlib import Path


def atr(high, low, close, period=14):
    """
    计算平均真实波幅(ATR)

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期 (默认14)

    Returns:
        ATR序列
    """
    # 计算真实波幅(TR)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 计算ATR (使用简单移动平均)
    atr_value = tr.rolling(period).mean()
    return atr_value


def supertrend(high, low, close, period=10, multiplier=3.0):
    """
    计算SuperTrend指标

    SuperTrend是一种趋势跟踪指标，基于ATR（平均真实波幅）计算。
    当价格位于SuperTrend线上方时为上涨趋势，下方时为下跌趋势。
    按照中国股市惯例：上涨趋势显示为红色，下跌趋势显示为绿色。

    Args:
        high: 最高价序列 (pandas Series)
        low: 最低价序列 (pandas Series)
        close: 收盘价序列 (pandas Series)
        period: ATR周期 (默认10)
        multiplier: ATR倍数 (默认3.0)

    Returns:
        (supertrend_line, trend): SuperTrend线和趋势方向
        - supertrend_line: SuperTrend线序列 (pandas Series)
        - trend: 趋势方向序列，1=上涨(红色), -1=下跌(绿色)
    """
    # 计算ATR
    atr_value = atr(high, low, close, period)

    # 计算基础上下带
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_value
    lower_band = hl2 - multiplier * atr_value

    # 初始化SuperTrend线和趋势
    supertrend_line = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)

    # 第一根K线的初始趋势
    if len(close) > 0:
        supertrend_line.iloc[0] = lower_band.iloc[0]
        trend.iloc[0] = 1

    # 迭代计算SuperTrend
    for i in range(1, len(close)):
        prev_trend = trend.iloc[i-1]
        prev_st = supertrend_line.iloc[i-1]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        curr_close = close.iloc[i]

        if prev_trend == 1:  # 之前是上涨趋势
            if curr_close < prev_st:  # 跌破SuperTrend线，转为下跌
                trend.iloc[i] = -1
                supertrend_line.iloc[i] = curr_upper
            else:  # 继续上涨
                trend.iloc[i] = 1
                # SuperTrend线取当前下带和之前SuperTrend线的较大值
                supertrend_line.iloc[i] = max(curr_lower, prev_st)
        else:  # 之前是下跌趋势
            if curr_close > prev_st:  # 突破SuperTrend线，转为上涨
                trend.iloc[i] = 1
                supertrend_line.iloc[i] = curr_lower
            else:  # 继续下跌
                trend.iloc[i] = -1
                # SuperTrend线取当前上带和之前SuperTrend线的较小值
                supertrend_line.iloc[i] = min(curr_upper, prev_st)

    return supertrend_line, trend


def supertrend_signals(supertrend_line, trend):
    """
    根据SuperTrend指标生成买卖信号

    当趋势从下跌转为上涨时产生买入信号，
    当趋势从上涨转为下跌时产生卖出信号。

    Args:
        supertrend_line: SuperTrend线序列 (pandas Series)
        trend: 趋势方向序列，1=上涨, -1=下跌 (pandas Series)

    Returns:
        (buy_signals, sell_signals): 买卖信号序列
        - buy_signals: 买入信号序列，True表示买入点 (pandas Series)
        - sell_signals: 卖出信号序列，True表示卖出点 (pandas Series)
    """
    # 计算趋势变化
    trend_change = trend.diff()

    # 买入信号：趋势从-1变为1
    buy_signals = (trend_change == 2)

    # 卖出信号：趋势从1变为-1
    sell_signals = (trend_change == -2)

    return buy_signals, sell_signals


class SupertrendStrategy:
    """
    SuperTrend 趋势跟踪策略

    参数:
    - period: ATR周期 (默认10)
    - multiplier: ATR倍数 (默认3.0)
    """
    def __init__(self, period=10, multiplier=3.0):
        self.period = period
        self.multiplier = multiplier
        self.name = f"SuperTrend({period},{multiplier})"

    def generate_signal(self, df):
        """
        生成交易信号

        Args:
            df: 包含 high, low, close 的 DataFrame

        Returns:
            添加了 position 列的 DataFrame
        """
        df = df.copy()

        # 计算 SuperTrend
        st_line, trend = supertrend(
            df['high'],
            df['low'],
            df['close'],
            period=self.period,
            multiplier=self.multiplier
        )

        df['supertrend'] = st_line
        df['trend'] = trend

        # 生成买卖信号
        buy_signals, sell_signals = supertrend_signals(st_line, trend)

        df['buy_signal'] = buy_signals
        df['sell_signal'] = sell_signals

        # 生成持仓信号
        # trend=1 表示上涨趋势，做多
        # trend=-1 表示下跌趋势，做空
        df['position'] = trend

        return df


def load_data(freq="1min", start_date="2023-01-01", end_date="2023-12-31"):
    """
    加载 qlib 数据

    Args:
        freq: 频率 (1min, 5min, 15min, 60min)
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame 包含 OHLCV 数据
    """
    DATA_DIR = Path("/mnt/d/quant/qlib_data")
    RESAMPLED_DIR = Path("/mnt/d/quant/qlib_backtest/qlib_data_multi_freq")

    FIELD_MAPPING = {
        'open': '$open',
        'high': '$high',
        'low': '$low',
        'close': '$close',
        'volume': '$volume',
        'amount': '$amount',
        'vwap': '$vwap',
        'open_interest': '$open_interest'
    }

    instrument = "RB9999.XSGE"

    if freq == "1min":
        data_dir = DATA_DIR / "instruments" / instrument
    else:
        data_dir = RESAMPLED_DIR / "instruments" / freq

    # 读取各个字段
    data = {}
    for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'open_interest']:
        if freq == "1min":
            field_file = data_dir / f"{field}.csv"
        else:
            feature_name = FIELD_MAPPING[field]
            field_file = data_dir / feature_name / f"{instrument}.csv"

        if field_file.exists():
            df = pd.read_csv(field_file, index_col=0, parse_dates=True)
            data[field] = df.iloc[:, 0]

    df = pd.DataFrame(data)
    df = df.sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    return df


def run_backtest(df, strategy_name):
    """
    运行回测

    Args:
        df: 包含 position 列的 DataFrame
        strategy_name: 策略名称

    Returns:
        回测结果字典
    """
    df = df.copy()

    # 计算收益
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['buy_hold_returns'] = df['returns']

    # 移除NaN
    df = df.dropna(subset=['position', 'strategy_returns'])

    # 统计结果
    total_trades = (df['position'].diff() != 0).sum()
    cumulative_return = (1 + df['strategy_returns']).prod() - 1
    buy_hold_cumulative = (1 + df['buy_hold_returns']).prod() - 1

    # 计算最大回撤
    cum_returns = (1 + df['strategy_returns']).cumprod()
    running_max = cum_returns.expanding().max()
    max_drawdown = (running_max - cum_returns) / running_max
    max_drawdown = max_drawdown.max()

    # 胜率
    win_rate = (df['strategy_returns'] > 0).sum() / (df['strategy_returns'] != 0).sum() * 100

    # 年化指标
    trading_periods_per_year = {
        '1min': 240 * 252,
        '5min': 48 * 252,
        '15min': 16 * 252,
        '60min': 4 * 252
    }

    # 默认使用1分钟
    freq = '1min'
    annual_trading_periods = trading_periods_per_year.get(freq, 240 * 252)

    annual_return = (1 + cumulative_return) ** (annual_trading_periods / len(df)) - 1
    sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(annual_trading_periods)

    results = {
        'strategy_name': strategy_name,
        'total_trades': int(total_trades),
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'buy_hold_return': buy_hold_cumulative
    }

    return results


def main():
    """主程序 - 测试 SuperTrend 指标"""
    print("="*60)
    print("SuperTrend 指标回测 - Qlib 版本")
    print("="*60)

    # 加载数据
    df = load_data(freq="1min", start_date="2023-01-01", end_date="2023-12-31")
    print(f"\n数据加载完成: {len(df)} 行")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")

    # 测试不同的参数组合
    params_list = [
        (10, 3.0),
        (10, 2.0),
        (7, 3.0),
        (14, 2.5),
    ]

    all_results = []

    for period, multiplier in params_list:
        print(f"\n{'='*60}")
        print(f"测试参数: period={period}, multiplier={multiplier}")
        print(f"{'='*60}")

        strategy = SupertrendStrategy(period=period, multiplier=multiplier)
        df_strategy = strategy.generate_signal(df.copy())

        # 移除前几行（用于计算指标）
        df_strategy = df_strategy.dropna(subset=['position'])

        # 运行回测
        results = run_backtest(df_strategy, strategy.name)
        all_results.append(results)

        print(f"\n{strategy.name} 回测结果:")
        print(f"  总交易次数: {results['total_trades']}")
        print(f"  累计收益: {results['cumulative_return']:.2%}")
        print(f"  年化收益: {results['annual_return']:.2%}")
        print(f"  最大回撤: {results['max_drawdown']:.2%}")
        print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"  胜率: {results['win_rate']:.2f}%")
        print(f"  买入持有收益: {results['buy_hold_return']:.2%}")

        # 统计买卖信号数量
        buy_count = df_strategy['buy_signal'].sum()
        sell_count = df_strategy['sell_signal'].sum()
        print(f"  买入信号数: {int(buy_count)}")
        print(f"  卖出信号数: {int(sell_count)}")

    # 汇总结果
    print(f"\n{'='*60}")
    print("参数对比汇总")
    print(f"{'='*60}")

    print(f"\n{'策略':<30} {'交易':<6} {'累计':<10} {'年化':<10} {'回撤':<10} {'夏普':<8} {'胜率':<8}")
    print("-"*80)

    for results in all_results:
        print(f"{results['strategy_name']:<30} {results['total_trades']:<6} "
              f"{results['cumulative_return']:>8.2%} {results['annual_return']:>8.2%} "
              f"{results['max_drawdown']:>8.2%} {results['sharpe_ratio']:>6.2f} {results['win_rate']:>6.2f}%")

    return all_results, df_strategy


if __name__ == "__main__":
    results, df = main()
