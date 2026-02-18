#!/usr/bin/env python3
"""
V02: 简化入场策略
MSB+OB策略，使用简化入场规则（MSB信号后下一根反向K线入场，3根K线后平仓）

2023-2025年 多时间周期回测
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def calculate_momentum_z(close, window=50):
    """计算动量Z-Score"""
    price_change = close.diff()
    avg_change = price_change.rolling(window).mean()
    std_change = price_change.rolling(window).std()
    return (price_change - avg_change) / std_change


def detect_pivots(high, low, pivot_len=7):
    """检测枢轴点"""
    pivot_high = pd.Series(np.nan, index=high.index)
    pivot_low = pd.Series(np.nan, index=low.index)

    for i in range(pivot_len, len(high) - pivot_len):
        if high.iloc[i] == high.iloc[i-pivot_len:i+pivot_len+1].max():
            pivot_high.iloc[i] = high.iloc[i]
        if low.iloc[i] == low.iloc[i-pivot_len:i+pivot_len+1].min():
            pivot_low.iloc[i] = low.iloc[i]

    return pivot_high, pivot_low


def simple_msb_ob_strategy(df, pivot_len=7, msb_zscore=0.3):
    """超简化的MSB+OB策略"""
    df = df.copy()

    # 1. 计算指标
    df['momentum_z'] = calculate_momentum_z(df['close'])
    df['vol_percentile'] = df['volume'].rolling(100).rank(pct=True) * 100

    # 2. 检测枢轴点
    df['pivot_high'], df['pivot_low'] = detect_pivots(df['high'], df['low'], pivot_len)

    # 3. 检测MSB
    df['msb_bull'] = False
    df['msb_bear'] = False

    last_ph = None
    last_pl = None

    for i in range(1, len(df)):
        # 更新枢轴点
        if not pd.isna(df['pivot_high'].iloc[i]):
            last_ph = df['pivot_high'].iloc[i]
        if not pd.isna(df['pivot_low'].iloc[i]):
            last_pl = df['pivot_low'].iloc[i]

        # MSB信号
        if last_ph is not None:
            is_msb_bull = (df['close'].iloc[i] > last_ph and
                          df['close'].iloc[i-1] <= last_ph and
                          df['momentum_z'].iloc[i] > msb_zscore)
            if is_msb_bull:
                df.at[df.index[i], 'msb_bull'] = True

        if last_pl is not None:
            is_msb_bear = (df['close'].iloc[i] < last_pl and
                          df['close'].iloc[i-1] >= last_pl and
                          df['momentum_z'].iloc[i] < -msb_zscore)
            if is_msb_bear:
                df.at[df.index[i], 'msb_bear'] = True

    # 4. 生成入场信号（超简化：MSB信号后，下一个反向K线入场）
    df['position'] = 0
    current_pos = 0
    entry_bar_idx = -1

    for i in range(2, len(df)):
        if current_pos == 0:
            # 检查MSB信号
            if df['msb_bull'].iloc[i-1]:
                # 看涨MSB：下一根K线如果有阴线就做多
                if df['close'].iloc[i] < df['open'].iloc[i]:
                    current_pos = 1
                    entry_bar_idx = i
            elif df['msb_bear'].iloc[i-1]:
                # 看跌MSB：下一根K线如果有阳线就做空
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    current_pos = -1
                    entry_bar_idx = i
        else:
            # 简单出场：3根K线后平仓
            bars_held = i - entry_bar_idx
            if bars_held >= 3:
                current_pos = 0
                entry_bar_idx = -1

        df.at[df.index[i], 'position'] = current_pos

    return df


def _identify_trades(df):
    """识别每笔完整交易"""
    trades = []
    in_position = False
    entry_idx = None
    entry_price = None
    pos_type = None

    for i in range(len(df)):
        if not in_position and df['position'].iloc[i] != 0:
            # 开仓
            in_position = True
            entry_idx = i
            entry_price = df['close'].iloc[i]
            pos_type = 'long' if df['position'].iloc[i] > 0 else 'short'
        elif in_position and df['position'].iloc[i] == 0:
            # 平仓
            exit_price = df['close'].iloc[i]
            if pos_type == 'long':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price
            trades.append({'entry': entry_idx, 'exit': i, 'pnl': pnl, 'type': pos_type})
            in_position = False

    return trades


def calculate_metrics(df, freq='15min'):
    """计算回测指标"""
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df = df.dropna(subset=['position', 'strategy_returns'])

    if len(df) == 0:
        return None

    cumulative_return = (1 + df['strategy_returns']).prod() - 1
    cum_returns = (1 + df['strategy_returns']).cumprod()
    max_drawdown = ((cum_returns.expanding().max() - cum_returns) / cum_returns.expanding().max()).max()

    trading_periods = {'5min': 48*252, '15min': 16*252, '60min': 4*252}
    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(trading_periods.get(freq, 16*252)) if df['strategy_returns'].std() > 0 else 0

    # 修正：基于完整交易计算胜率
    trades = _identify_trades(df)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'total_trades': total_trades
    }


def load_data(year, freq='60min'):
    """加载指定年份的数据"""
    data_file = project_root / "data" / f"RB9999_{year}_{freq}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    df = pd.read_csv(data_file, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def main():
    """主函数"""
    print("="*80)
    print("V02: 简化入场策略 - MSB+OB (MSB后反向K线入场，3根K线平仓)")
    print("="*80)

    years = [2023, 2024, 2025]
    freqs = ['60min', '15min', '5min']

    all_results = []

    for freq in freqs:
        print(f"\n{'='*80}")
        print(f"时间周期: {freq}")
        print(f"{'='*80}")

        for year in years:
            try:
                # 加载数据
                df = load_data(year, freq)

                print(f"\n{year}年...", end=' ')

                # 运行策略
                df_sig = simple_msb_ob_strategy(df)
                metrics = calculate_metrics(df_sig, freq=freq)

                if metrics:
                    metrics['year'] = year
                    metrics['freq'] = freq
                    all_results.append(metrics)

                    msb_count = df_sig['msb_bull'].sum() + df_sig['msb_bear'].sum()
                    print(f"MSB{int(msb_count):3}  交易{metrics['total_trades']:3}  "
                          f"胜率{metrics['win_rate']:5.2f}%  "
                          f"收益{metrics['cumulative_return']:6.2f}%  "
                          f"回撤{metrics['max_drawdown']:6.2f}%")

            except Exception as e:
                print(f"\n{year}年: 失败 - {e}")
                import traceback
                traceback.print_exc()

    # 汇总
    if all_results:
        print(f"\n{'='*80}")
        print("汇总对比")
        print(f"{'='*80}\n")

        print(f"{'周期':>6}  {'交易':>6}  {'胜率':>8}  {'总收益':>8}  {'回撤':>8}  {'夏普':>8}")
        print("-"*60)

        for freq in freqs:
            freq_res = [r for r in all_results if r['freq'] == freq]
            if not freq_res:
                continue

            total_trades = sum(r['total_trades'] for r in freq_res)
            avg_win = np.mean([r['win_rate'] for r in freq_res])
            total_ret = sum(r['cumulative_return'] for r in freq_res)
            max_dd = max(r['max_drawdown'] for r in freq_res)
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in freq_res])

            print(f"{freq:>6}  {total_trades:6}  {avg_win:7.2f}%  {total_ret:8.2f}%  {max_dd:8.2f}%  {avg_sharpe:8.2f}")

        # 保存
        df_res = pd.DataFrame(all_results)
        output = Path(__file__).parent / "report_v02_simple_entry.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(output, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {output}")

        return all_results


if __name__ == "__main__":
    main()
