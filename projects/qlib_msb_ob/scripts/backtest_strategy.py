#!/usr/bin/env python3
"""
MSB+OB 策略回测脚本
2023-2025年 60分钟线回测
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from strategy.msb_ob_strategy import MSBOBStrategy, calculate_metrics, calculate_ob_stats


# ============================================
# 数据加载
# ============================================

def load_data(year):
    """加载指定年份的数据"""
    data_file = Path(__file__).parent.parent / f"data/RB9999_{year}_60min.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    df = pd.read_csv(data_file, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    return df


# ============================================
# 回测执行
# ============================================

def run_backtest(df, strategy_name, params):
    """
    运行单次回测

    返回包含以下指标的字典：
    - 总交易次数
    - 盈利交易数
    - 亏损交易数
    - 胜率
    - 平均盈利
    - 平均亏损
    - 盈亏比
    - 累计收益
    - 最大回撤
    - 夏普比率
    - OB可靠性（源码Reliability）
    - HP-OB数量
    """
    # 初始化策略
    strategy = MSBOBStrategy(**params)

    # 生成信号
    df_signals = strategy.generate_signal(df.copy())

    # 计算指标
    metrics = calculate_metrics(df_signals, freq='60min')
    ob_stats = calculate_ob_stats(df_signals)

    # 统计盈利和亏损交易数
    df_signals_copy = df_signals.copy()
    df_signals_copy['returns'] = df_signals_copy['close'].pct_change()
    df_signals_copy['strategy_returns'] = df_signals_copy['position'].shift(1) * df_signals_copy['returns']
    df_signals_copy = df_signals_copy.dropna(subset=['strategy_returns'])

    winning_trades = (df_signals_copy['strategy_returns'] > 0).sum()
    losing_trades = (df_signals_copy['strategy_returns'] < 0).sum()

    # 合并结果
    result = {
        'strategy_name': strategy_name,
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        **metrics,
        **ob_stats
    }

    return result, df_signals


# ============================================
# 详细分析
# ============================================

def analyze_trades(df):
    """
    分析每笔交易

    返回：
    - 交易列表
    - 持仓时长分布
    - 盈利分布
    - OB质量与交易结果关系
    """
    trades = []

    df_copy = df.copy()
    df_copy['position_change'] = df_copy['position'].diff().fillna(0)

    entry_idx = None
    entry_price = None
    entry_type = None  # 'long' or 'short'
    ob_quality = None

    for idx, row in df_copy.iterrows():
        if row['position_change'] == 1:  # 开多仓
            entry_idx = idx
            entry_price = row['close']
            entry_type = 'long'
            if not pd.isna(row['ob_bullish_quality']):
                ob_quality = row['ob_bullish_quality']

        elif row['position_change'] == -1:  # 开空仓
            entry_idx = idx
            entry_price = row['close']
            entry_type = 'short'
            if not pd.isna(row['ob_bearish_quality']):
                ob_quality = row['ob_bearish_quality']

        elif (row['position_change'] == -1 and entry_type == 'long') or \
             (row['position_change'] == 1 and entry_type == 'short') or \
             (row['position'] == 0 and entry_idx is not None):  # 平仓
            if entry_idx is not None:
                exit_price = row['close']
                exit_idx = idx

                if entry_type == 'long':
                    profit = exit_price - entry_price
                    profit_pct = profit / entry_price * 100
                else:
                    profit = entry_price - exit_price
                    profit_pct = profit / entry_price * 100

                trades.append({
                    'entry_time': entry_idx,
                    'exit_time': exit_idx,
                    'type': entry_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'ob_quality': ob_quality
                })

                entry_idx = None
                entry_price = None
                entry_type = None
                ob_quality = None

    return trades


def analyze_ob_performance(df):
    """
    分析OB表现

    计算：
    - OB生成总数
    - OB失效总数
    - OB可靠性
    - HP-OB数量
    - 基于OB的交易胜率
    """
    return calculate_ob_stats(df)


# ============================================
# 报告生成
# ============================================

def generate_reports(all_results, trades_list, ob_stats):
    """
    生成多格式报告

    输出文件：
    1. CSV格式（详细数据）
    2. JSON格式（结构化数据）
    3. TXT格式（可读摘要）
    """
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 1. CSV报告
    csv_file = reports_dir / "backtest_results.csv"
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')

    # 2. JSON报告
    json_file = reports_dir / "backtest_results.json"

    # 计算汇总统计
    total_trades = sum(r['total_trades'] for r in all_results)
    avg_win_rate = np.mean([r['win_rate'] for r in all_results])
    total_return = sum(r['cumulative_return'] for r in all_results)
    max_drawdown = max(r['max_drawdown'] for r in all_results)
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
    avg_ob_reliability = np.mean([r['ob_reliability'] for r in all_results])
    avg_hp_ob = np.mean([r['hp_ob_count'] for r in all_results])

    # 提取参数
    params = {
        'pivot_len': 7,
        'msb_zscore': 0.5,
        'ob_max_count': 10,
        'hpz_threshold': 80,
        'atr_period': 14,
        'atr_multiplier': 1.0,
        'tp1_multiplier': 0.5,
        'tp2_multiplier': 1.0,
        'tp3_multiplier': 1.5
    }

    json_data = {
        "strategy_name": "MSB+OB",
        "parameters": params,
        "backtest_period": "2023-2025",
        "frequency": "60min",
        "instrument": "RB9999.XSGE",
        "yearly_results": all_results,
        "overall_stats": {
            "total_trades": total_trades,
            "avg_win_rate": round(avg_win_rate, 2),
            "total_return": round(total_return, 4),
            "max_drawdown": round(max_drawdown, 4),
            "avg_sharpe_ratio": round(avg_sharpe, 2),
            "avg_ob_reliability": round(avg_ob_reliability, 2),
            "avg_hp_ob_count": round(avg_hp_ob, 1)
        },
        "generated_at": datetime.now().isoformat()
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # 3. TXT摘要
    txt_file = reports_dir / "backtest_summary.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MSB+OB 策略回测报告 (2023-2025)\n")
        f.write("=" * 80 + "\n\n")

        f.write("策略参数:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("年度结果:\n")
        for result in all_results:
            year = result.get('year', 'N/A')
            f.write(f"  {year}: 收益{result['cumulative_return']:.2%}, "
                   f"回撤{result['max_drawdown']:.2%}, "
                   f"夏普{result['sharpe_ratio']:.2f}, "
                   f"胜率{result['win_rate']:.2f}%\n")
        f.write("\n")

        f.write("汇总:\n")
        f.write(f"  总交易: {total_trades}笔\n")
        f.write(f"  平均胜率: {avg_win_rate:.2f}%\n")
        f.write(f"  总收益: {total_return:.2%}\n")
        f.write(f"  最大回撤: {max_drawdown:.2%}\n")
        f.write(f"  平均夏普: {avg_sharpe:.2f}\n\n")

        f.write("OB统计:\n")
        f.write(f"  平均可靠性: {avg_ob_reliability:.2f}%\n")
        f.write(f"  平均HP-OB数: {avg_hp_ob:.1f}\n\n")

        # 生成结论
        f.write("结论:\n")
        if total_return > 0:
            f.write("  策略在2023-2025年期间实现正收益。\n")
        else:
            f.write("  策略在2023-2025年期间出现亏损。\n")

        if avg_win_rate > 50:
            f.write("  胜率较高，说明策略入场信号质量较好。\n")
        else:
            f.write("  胜率偏低，需要关注盈亏比是否足以弥补胜率不足。\n")

        if avg_sharpe > 1.0:
            f.write("  夏普比率大于1，策略风险调整后收益表现良好。\n")
        else:
            f.write("  夏普比率较低，策略风险收益比需要优化。\n")

        if max_drawdown < -0.10:
            f.write(f"  最大回撤{max_drawdown:.2%}较大，建议优化风险控制参数。\n")
        else:
            f.write(f"  最大回撤{max_drawdown:.2%}在可接受范围内。\n")

    return csv_file, json_file, txt_file


def print_summary(all_results):
    """
    打印汇总报告
    """
    print("=" * 80)
    print("MSB+OB 策略回测报告")
    print("=" * 80)

    # 提取参数
    params = {
        'pivot_len': 7,
        'msb_zscore': 0.5,
        'ob_max_count': 10,
        'hpz_threshold': 80,
        'atr_period': 14,
        'atr_multiplier': 1.0,
        'tp1_multiplier': 0.5,
        'tp2_multiplier': 1.0,
        'tp3_multiplier': 1.5
    }

    print("\n策略参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    print("\n" + "-" * 80)
    print(f"{'年份':<8} {'交易次数':<8} {'胜率':<10} {'累计收益':<12} {'最大回撤':<12} {'夏普比率':<10} {'OB可靠性':<12} {'HP-OB数':<8}")
    print("-" * 80)

    for result in all_results:
        year = result.get('year', 'N/A')
        print(f"{year:<8} {result['total_trades']:<8} "
              f"{result['win_rate']:<10.2f} "
              f"{result['cumulative_return']:<12.2%} "
              f"{result['max_drawdown']:<12.2%} "
              f"{result['sharpe_ratio']:<10.2f} "
              f"{result['ob_reliability']:<12.2f} "
              f"{result['hp_ob_count']:<8}")

    print("-" * 80)

    # 汇总统计
    total_trades = sum(r['total_trades'] for r in all_results)
    avg_win_rate = np.mean([r['win_rate'] for r in all_results])
    total_return = sum(r['cumulative_return'] for r in all_results)
    max_drawdown = max(r['max_drawdown'] for r in all_results)
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
    avg_ob_reliability = np.mean([r['ob_reliability'] for r in all_results])

    print("\n汇总统计:")
    print(f"  总交易次数: {total_trades}")
    print(f"  平均胜率: {avg_win_rate:.2f}%")
    print(f"  总收益: {total_return:.2%}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    print(f"  平均夏普: {avg_sharpe:.2f}")
    print(f"  平均OB可靠性: {avg_ob_reliability:.2f}%")

    print("\n" + "=" * 80)
    print("结论:")
    if total_return > 0:
        print("  策略整体盈利，表现良好。")
    else:
        print("  策略整体亏损，需要优化参数。")

    if avg_sharpe > 1.0:
        print("  夏普比率大于1，风险调整后收益表现良好。")
    else:
        print("  夏普比率较低，建议优化风险收益比。")

    if abs(max_drawdown) > 0.15:
        print(f"  最大回撤{max_drawdown:.2%}偏大，建议优化止损策略。")
    else:
        print(f"  最大回撤{max_drawdown:.2%}在可接受范围内。")
    print("=" * 80)


# ============================================
# 主程序
# ============================================

def main():
    """主回测流程"""

    # 标准参数
    params = {
        'pivot_len': 7,
        'msb_zscore': 0.5,
        'ob_max_count': 10,
        'hpz_threshold': 80,
        'atr_period': 14,
        'atr_multiplier': 1.0,
        'tp1_multiplier': 0.5,
        'tp2_multiplier': 1.0,
        'tp3_multiplier': 1.5
    }

    years = [2023, 2024, 2025]
    all_results = []
    all_trades = []

    for year in years:
        print(f"\n{'='*80}")
        print(f"回测 {year} 年")
        print(f"{'='*80}")

        # 1. 加载数据
        df = load_data(year)
        print(f"数据加载: {len(df)} 行")
        print(f"时间范围: {df.index.min()} ~ {df.index.max()}")

        # 2. 运行回测
        result, df_signals = run_backtest(df, f"MSB+OB_{year}", params)

        # 3. 分析交易
        trades = analyze_trades(df_signals)

        # 4. 汇总结果
        result['year'] = year
        all_results.append(result)
        all_trades.extend(trades)

        # 5. 打印年度结果
        print(f"\n{year}年回测结果:")
        print(f"  总交易次数: {result['total_trades']}")
        print(f"  盈利交易: {result['winning_trades']}")
        print(f"  亏损交易: {result['losing_trades']}")
        print(f"  胜率: {result['win_rate']:.2f}%")
        print(f"  累计收益: {result['cumulative_return']:.2%}")
        print(f"  最大回撤: {result['max_drawdown']:.2%}")
        print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"  盈亏比: {result['profit_loss_ratio']:.2f}")
        print(f"  OB生成总数: {result['total_ob_created']}")
        print(f"  OB失效总数: {result['total_ob_mitigated']}")
        print(f"  OB可靠性: {result['ob_reliability']:.2f}%")
        print(f"  HP-OB数量: {result['hp_ob_count']}")

    # 6. 生成报告
    csv_file, json_file, txt_file = generate_reports(all_results, all_trades, None)

    print(f"\n{'='*80}")
    print("回测完成")
    print(f"{'='*80}")
    print(f"报告已生成:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")
    print(f"  TXT: {txt_file}")

    # 7. 打印汇总
    print_summary(all_results)

    return all_results, all_trades


if __name__ == "__main__":
    results = main()
