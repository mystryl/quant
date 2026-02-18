#!/usr/bin/env python3
"""
Qlib SuperTrend 指标增强版 - 基于 SF14Re 策略

主要增强功能：
1. 渐进式跟踪止损机制
2. 双重突破确认
3. 自适应参数调整
4. 持仓时间动态调整止损收紧度
5. 自动回测报告生成
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加父目录到路径，以便导入报告生成器
sys.path.insert(0, str(Path(__file__).parent.parent))
from other.report_generator import create_backtest_report


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


def supertrend_enhanced(high, low, close, period=50, multiplier=20, n=3):
    """
    计算增强版 SuperTrend 指标

    Args:
        high: 最高价序列 (pandas Series)
        low: 最低价序列 (pandas Series)
        close: 收盘价序列 (pandas Series)
        period: ATR周期 (默认50)
        multiplier: ATR倍数 (默认20)
        n: 突破确认系数，价格需要超过 SuperTrend 线 N 倍 ATR 才确认突破 (默认3)

    Returns:
        (supertrend_line, trend, atr_value): SuperTrend线、趋势方向和ATR序列
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
        curr_atr = atr_value.iloc[i]

        # 突破确认阈值
        breakthrough_threshold = n * curr_atr

        if prev_trend == 1:  # 之前是上涨趋势
            # 双重确认：收盘价跌破SuperTrend线 且 跌破幅度超过确认阈值
            if curr_close < prev_st - breakthrough_threshold:
                trend.iloc[i] = -1
                supertrend_line.iloc[i] = curr_upper
            else:
                trend.iloc[i] = 1
                supertrend_line.iloc[i] = max(curr_lower, prev_st)
        else:  # 之前是下跌趋势
            # 双重确认：收盘价突破SuperTrend线 且 突破幅度超过确认阈值
            if curr_close > prev_st + breakthrough_threshold:
                trend.iloc[i] = 1
                supertrend_line.iloc[i] = curr_lower
            else:
                trend.iloc[i] = -1
                supertrend_line.iloc[i] = min(curr_upper, prev_st)

    return supertrend_line, trend, atr_value


def calculate_liqka(holding_period, max_period=100, min_liqka=0.5, max_liqka=1.0):
    """
    计算动态止损系数 liQKA

    持仓时间越长，止损系数越小（止损越紧）
    liQKA 从 max_liqka 递减至 min_liqka

    Args:
        holding_period: 当前持仓时长（单位：K线数量）
        max_period: 达到最紧止损所需的持仓时长 (默认100)
        min_liqka: 最小止损系数 (默认0.5)
        max_liqka: 最大止损系数 (默认1.0)

    Returns:
        liQKA: 当前止损系数
    """
    if holding_period >= max_period:
        return min_liqka

    # 线性递减
    ratio = holding_period / max_period
    liqka = max_liqka - ratio * (max_liqka - min_liqka)
    return liqka


def supertrend_signals_with_trailing(
    df,
    trailing_stop_rate=80,
    max_holding_period=100,
    min_liqka=0.5,
    max_liqka=1.0
):
    """
    增强版信号生成 - 包含渐进式跟踪止损

    Args:
        df: 包含 supertrend, trend, atr, close 的 DataFrame
        trailing_stop_rate: 止损幅度百分比 (默认80, 范围40-100)
        max_holding_period: 达到最紧止损所需持仓时长 (默认100)
        min_liqka: 最小止损系数 (默认0.5)
        max_liqka: 最大止损系数 (默认1.0)

    Returns:
        添加了 position, stop_loss, holding_period 的 DataFrame
    """
    df = df.copy()

    # 初始化
    df['position'] = 0  # 1=做多, -1=做空, 0=空仓
    df['holding_period'] = 0  # 持仓时长
    df['stop_loss'] = np.nan  # 止损价格
    df['entry_price'] = np.nan  # 入场价格

    # 计算基础买卖信号（基于趋势反转）
    df['trend_change'] = df['trend'].diff()
    buy_signals = (df['trend_change'] == 2)  # 从-1转为1
    sell_signals = (df['trend_change'] == -2)  # 从1转为-1

    current_position = 0
    current_holding_period = 0
    current_entry_price = np.nan

    for i in range(1, len(df)):
        idx = df.index[i]

        # 获取当前数据
        close = df.loc[idx, 'close']
        trend = df.loc[idx, 'trend']
        supertrend_line = df.loc[idx, 'supertrend']
        atr_value = df.loc[idx, 'atr']

        # 检查是否有交易信号
        has_buy_signal = buy_signals.iloc[i]
        has_sell_signal = sell_signals.iloc[i]

        # 计算止损系数
        if current_position != 0:
            liqka = calculate_liqka(
                current_holding_period,
                max_holding_period,
                min_liqka,
                max_liqka
            )
        else:
            liqka = max_liqka

        # 计算动态止损价格
        if current_position == 1:  # 持有多仓
            stop_loss_price = current_entry_price * (1 - trailing_stop_rate / 1000 * liqka)
            # 止损价格不能低于 SuperTrend 线（趋势止损保护）
            stop_loss_price = max(stop_loss_price, supertrend_line * (1 - 0.01))  # 1% 容差
        elif current_position == -1:  # 持有空仓
            stop_loss_price = current_entry_price * (1 + trailing_stop_rate / 1000 * liqka)
            # 止损价格不能高于 SuperTrend 线
            stop_loss_price = min(stop_loss_price, supertrend_line * (1 + 0.01))
        else:
            stop_loss_price = np.nan

        # 检查止损
        stopped_out = False
        if current_position == 1 and close < stop_loss_price:
            stopped_out = True
        elif current_position == -1 and close > stop_loss_price:
            stopped_out = True

        # 执行交易逻辑
        if stopped_out:
            # 触发止损，平仓
            current_position = 0
            current_holding_period = 0
            current_entry_price = np.nan
            df.loc[idx, 'position'] = 0
            df.loc[idx, 'stop_loss'] = stop_loss_price
        elif has_buy_signal and current_position != 1:
            # 买入信号，开多仓
            current_position = 1
            current_holding_period = 0
            current_entry_price = close
            df.loc[idx, 'position'] = 1
            df.loc[idx, 'entry_price'] = close
            df.loc[idx, 'stop_loss'] = stop_loss_price
        elif has_sell_signal and current_position != -1:
            # 卖出信号，开空仓
            current_position = -1
            current_holding_period = 0
            current_entry_price = close
            df.loc[idx, 'position'] = -1
            df.loc[idx, 'entry_price'] = close
            df.loc[idx, 'stop_loss'] = stop_loss_price
        else:
            # 持续持仓
            if current_position != 0:
                current_holding_period += 1
            df.loc[idx, 'position'] = current_position
            df.loc[idx, 'holding_period'] = current_holding_period
            df.loc[idx, 'stop_loss'] = stop_loss_price
            if current_position != 0:
                df.loc[idx, 'entry_price'] = current_entry_price

    return df


class SupertrendEnhancedStrategy:
    """
    增强版 SuperTrend 趋势跟踪策略 - 基于 SF14Re

    参数:
    - period: ATR周期 (默认50，SF14Re标准)
    - multiplier: ATR倍数 (默认20，SF14Re标准)
    - n: 突破确认系数 (默认3，价格需要超过SuperTrend线N倍ATR才确认)
    - trailing_stop_rate: 止损幅度百分比 (默认80，范围40-100)
    - max_holding_period: 达到最紧止损所需持仓时长 (默认100)
    - min_liqka: 最小止损系数 (默认0.5)
    - max_liqka: 最大止损系数 (默认1.0)
    """
    def __init__(
        self,
        period=50,
        multiplier=20,
        n=3,
        trailing_stop_rate=80,
        max_holding_period=100,
        min_liqka=0.5,
        max_liqka=1.0
    ):
        self.period = period
        self.multiplier = multiplier
        self.n = n
        self.trailing_stop_rate = trailing_stop_rate
        self.max_holding_period = max_holding_period
        self.min_liqka = min_liqka
        self.max_liqka = max_liqka
        self.name = f"SuperTrend_SF14Re({period},{multiplier},n={n},ts={trailing_stop_rate})"

    def generate_signal(self, df):
        """
        生成交易信号

        Args:
            df: 包含 high, low, close 的 DataFrame

        Returns:
            添加了 position, stop_loss, holding_period 列的 DataFrame
        """
        df = df.copy()

        # 计算 SuperTrend
        st_line, trend, atr_value = supertrend_enhanced(
            df['high'],
            df['low'],
            df['close'],
            period=self.period,
            multiplier=self.multiplier,
            n=self.n
        )

        df['supertrend'] = st_line
        df['trend'] = trend
        df['atr'] = atr_value

        # 使用渐进式跟踪止损生成信号
        df = supertrend_signals_with_trailing(
            df,
            trailing_stop_rate=self.trailing_stop_rate,
            max_holding_period=self.max_holding_period,
            min_liqka=self.min_liqka,
            max_liqka=self.max_liqka
        )

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
    RESAMPLED_DIR = Path("/mnt/d/quant/qlib_backtest/data/qlib_data_multi_freq")

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
    total_trades = (df['position'].diff() != 0).sum() / 2  # 每次开仓+平仓算一次完整交易
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

    # 自动判断数据频率（基于时间戳间隔）
    time_diff = df.index.to_series().diff().median()

    if pd.isna(time_diff):
        freq = '1min'
    elif time_diff >= pd.Timedelta(minutes=50) and time_diff <= pd.Timedelta(minutes=70):
        freq = '60min'
    elif time_diff >= pd.Timedelta(minutes=10) and time_diff <= pd.Timedelta(minutes=20):
        freq = '15min'
    elif time_diff >= pd.Timedelta(minutes=4) and time_diff <= pd.Timedelta(minutes=6):
        freq = '5min'
    else:
        freq = '1min'

    annual_trading_periods = trading_periods_per_year.get(freq, 240 * 252)

    annual_return = (1 + cumulative_return) ** (annual_trading_periods / len(df)) - 1
    sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(annual_trading_periods)

    # 止损次数统计
    stopped_out_count = (df['position'].diff() != 0) & (df['position'] == 0) & (df['position'].shift(1) != 0)
    stopped_out_count = stopped_out_count.sum()

    results = {
        'strategy_name': strategy_name,
        'total_trades': int(total_trades),
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'buy_hold_return': buy_hold_cumulative,
        'stopped_out_count': int(stopped_out_count)
    }

    return results


def main():
    """主程序 - 测试增强版 SuperTrend 指标"""
    print("="*60)
    print("增强版 SuperTrend 指标回测 - 基于 SF14Re 策略")
    print("="*60)

    # 加载数据
    df = load_data(freq="1min", start_date="2023-01-01", end_date="2023-12-31")
    print(f"\n数据加载完成: {len(df)} 行")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")

    # 测试不同的参数组合
    params_list = [
        # SF14Re 标准参数
        (50, 20, 3, 80, 100, 0.5, 1.0),
        # 调整后的参数（更保守）
        (50, 15, 3, 70, 100, 0.5, 1.0),
        (30, 10, 2, 80, 100, 0.6, 1.0),
        # 测试不同止损系数
        (50, 20, 3, 60, 100, 0.5, 1.0),
        (50, 20, 3, 90, 100, 0.5, 1.0),
    ]

    all_results = []

    for period, multiplier, n, ts_rate, max_hold, min_liqka, max_liqka in params_list:
        print(f"\n{'='*60}")
        print(f"测试参数: period={period}, multiplier={multiplier}, n={n}, "
              f"ts_rate={ts_rate}, max_hold={max_hold}, "
              f"min_liqka={min_liqka}, max_liqka={max_liqka}")
        print(f"{'='*60}")

        strategy = SupertrendEnhancedStrategy(
            period=period,
            multiplier=multiplier,
            n=n,
            trailing_stop_rate=ts_rate,
            max_holding_period=max_hold,
            min_liqka=min_liqka,
            max_liqka=max_liqka
        )
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
        print(f"  止损平仓次数: {results['stopped_out_count']}")

        # 自动生成回测报告
        params = {
            'period': period,
            'multiplier': multiplier,
            'n': n,
            'trailing_stop_rate': ts_rate,
            'max_holding_period': max_hold,
            'min_liqka': min_liqka,
            'max_liqka': max_liqka,
            'freq': '15min',
            'year': 2023
        }

        try:
            report_dir = create_backtest_report(
                strategy_name=strategy.name,
                params=params,
                results=results,
                df=df_strategy,
                config={
                    'description': '增强版SuperTrend指标，基于SF14Re策略',
                    'notes': f'测试参数组合: period={period}, multiplier={multiplier}, n={n}'
                },
                script_path=__file__
            )
            print(f"  回测报告已自动生成: {report_dir}")
        except Exception as e:
            print(f"  ⚠️  生成回测报告失败: {e}")

    # 汇总结果
    print(f"\n{'='*60}")
    print("参数对比汇总")
    print(f"{'='*60}")

    print(f"\n{'策略':<50} {'交易':<6} {'累计':<10} {'年化':<10} {'回撤':<10} {'夏普':<8} {'胜率':<8} {'止损':<6}")
    print("-"*110)

    for results in all_results:
        print(f"{results['strategy_name']:<50} {results['total_trades']:<6} "
              f"{results['cumulative_return']:>8.2%} {results['annual_return']:>8.2%} "
              f"{results['max_drawdown']:>8.2%} {results['sharpe_ratio']:>6.2f} "
              f"{results['win_rate']:>6.2f}% {results['stopped_out_count']:<6}")

    return all_results, df_strategy


if __name__ == "__main__":
    results, df = main()
