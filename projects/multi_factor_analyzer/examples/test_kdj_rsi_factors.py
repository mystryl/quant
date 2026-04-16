"""
KDJ 和 RSI 金叉死叉因子测试

测试两个经典的技术指标因子：
1. KDJ 金叉死叉因子
2. RSI 金叉死叉因子

金叉：买入信号（因子值为正）
死叉：卖出信号（因子值为负）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('/Users/mystryl/Documents/Quant/projects/multi_factor_analyzer')

from src.core import FactorManager, PerformanceEvaluator, CycleAligner, StrategyAnalyzer
from src.core.reliability import ReliabilityEvaluator, DEFAULT_WEIGHTS
from src.report import ReportGenerator, Visualizer
from src.utils.helpers import calculate_returns


def create_mock_data(instruments=None, start_date='2020-01-01', end_date='2023-12-31'):
    """
    创建模拟数据用于测试

    在实际使用时，应该替换为真实数据：
    from src.data import FactorDataProvider
    provider = FactorDataProvider(data_dir="/path/to/data")
    data = provider.get_factor_data(instruments, fields, start_date, end_date)
    """
    if instruments is None:
        instruments = ['STOCK001', 'STOCK002', 'STOCK003', 'STOCK004', 'STOCK005']

    # 创建日期索引
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # 只保留工作日

    # 生成模拟价格数据
    np.random.seed(42)
    data = {}

    for instrument in instruments:
        # 随机游走生成价格
        returns = np.random.normal(0.0005, 0.02, len(dates))
        price = 100 * np.cumprod(1 + returns)

        # 生成 OHLC 数据
        df = pd.DataFrame({
            'open': price * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': price * (1 + np.abs(np.random.uniform(0, 0.02, len(dates)))),
            'low': price * (1 - np.abs(np.random.uniform(0, 0.02, len(dates)))),
            'close': price,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)

        # 确保 high >= close >= low
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.uniform(0, 0.01, len(dates))))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.uniform(0, 0.01, len(dates))))

        data[instrument] = df

    return data


def calculate_kdj(high, low, close, n=9, m1=3, m2=3):
    """
    计算 KDJ 指标

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: RSV 周期
        m1: K 值平滑周期
        m2: D 值平滑周期

    Returns:
        K, D, J 三个指标值
    """
    # 计算 RSV (Raw Stochastic Value)
    low_n = low.rolling(window=n, min_periods=1).min()
    high_n = high.rolling(window=n, min_periods=1).max()
    rsv = (close - low_n) / (high_n - low_n) * 100

    # 计算 K 值 (SMA of RSV)
    K = rsv.ewm(alpha=1/m1, adjust=False).mean()

    # 计算 D 值 (SMA of K)
    D = K.ewm(alpha=1/m2, adjust=False).mean()

    # 计算 J 值
    J = 3 * K - 2 * D

    return K, D, J


def calculate_rsi(close, n=14):
    """
    计算 RSI 指标

    Args:
        close: 收盘价序列
        n: RSI 周期

    Returns:
        RSI 值
    """
    # 计算价格变化
    delta = close.diff()

    # 分离上涨和下跌
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # 计算平均涨跌幅
    avg_gains = gains.rolling(window=n, min_periods=1).mean()
    avg_losses = losses.rolling(window=n, min_periods=1).mean()

    # 计算 RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_kdj_cross_signal(df, n=9, m1=3, m2=3):
    """
    计算 KDJ 金叉死叉因子

    金叉：J 线上穿 K 线和 D 线（买入信号，因子值为正）
    死叉：J 线下穿 K 线和 D 线（卖出信号，因子值为负）

    Args:
        df: 包含 high, low, close 的 DataFrame
        n: KDJ 周期参数
        m1: K 值平滑周期
        m2: D 值平滑周期

    Returns:
        factor: KDJ 金叉死叉因子值
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # 计算 KDJ
    K, D, J = calculate_kdj(high, low, close, n, m1, m2)

    # 计算金叉死叉信号
    # 金叉：J 上穿 K 且 J 上穿 D，且 K 和 D 都在低位（< 20）
    golden_cross = (J > K) & (J.shift(1) <= K.shift(1)) & (J > D) & (J.shift(1) <= D.shift(1)) & (K < 20) & (D < 20)

    # 死叉：J 下穿 K 且 J 下穿 D，且 K 和 D 都在高位（> 80）
    death_cross = (J < K) & (J.shift(1) >= K.shift(1)) & (J < D) & (J.shift(1) >= D.shift(1)) & (K > 80) & (D > 80)

    # 生成因子值
    factor = pd.Series(0.0, index=close.index)

    # 金叉时因子值为 J 值（正相关）
    factor[golden_cross] = J[golden_cross]

    # 死叉时因子值为 -J 值（负相关）
    factor[death_cross] = -J[death_cross]

    # 对于其他情况，使用 (J - 50) 作为因子值（反映超买超卖）
    factor[(factor == 0) & (~golden_cross) & (~death_cross)] = (J - 50) / 50

    return factor


def calculate_rsi_cross_signal(df, fast_n=6, slow_n=14):
    """
    计算 RSI 金叉死叉因子

    金叉：快线 RSI 上穿慢线 RSI（买入信号，因子值为正）
    死叉：快线 RSI 下穿慢线 RSI（卖出信号，因子值为负）

    Args:
        df: 包含 close 的 DataFrame
        fast_n: 快线 RSI 周期
        slow_n: 慢线 RSI 周期

    Returns:
        factor: RSI 金叉死叉因子值
    """
    close = df['close']

    # 计算快慢 RSI
    rsi_fast = calculate_rsi(close, fast_n)
    rsi_slow = calculate_rsi(close, slow_n)

    # 计算金叉死叉信号
    golden_cross = (rsi_fast > rsi_slow) & (rsi_fast.shift(1) <= rsi_slow.shift(1)) & (rsi_fast < 30)
    death_cross = (rsi_fast < rsi_slow) & (rsi_fast.shift(1) >= rsi_slow.shift(1)) & (rsi_fast > 70)

    # 生成因子值
    factor = pd.Series(0.0, index=close.index)

    # 金叉时因子值为 (RSI快线 - 50)（正相关）
    factor[golden_cross] = (rsi_fast[golden_cross] - 50) / 50

    # 死叉时因子值为 -(RSI快线 - 50)（负相关）
    factor[death_cross] = -(rsi_fast[death_cross] - 50) / 50

    # 对于其他情况，使用 (RSI快线 - RSI慢线) 作为因子值
    factor[(factor == 0) & (~golden_cross) & (~death_cross)] = (rsi_fast - rsi_slow) / 50

    return factor


def prepare_factor_data(data_dict, factor_func, factor_name):
    """
    准备因子数据

    Args:
        data_dict: 股票数据字典 {instrument: df}
        factor_func: 因子计算函数
        factor_name: 因子名称

    Returns:
        factor_df: 因子 DataFrame (MultiIndex: datetime, instrument)
        price_df: 价格 DataFrame (MultiIndex: datetime, instrument)
    """
    factor_data = []
    price_data = []

    for instrument, df in data_dict.items():
        # 计算因子
        factor = factor_func(df)

        # 创建临时 DataFrame
        temp_factor = pd.DataFrame({
            'datetime': factor.index,
            'instrument': instrument,
            'factor': factor.values
        })
        factor_data.append(temp_factor)

        # 获取收盘价
        temp_price = pd.DataFrame({
            'datetime': df.index,
            'instrument': instrument,
            'close': df['close'].values
        })
        price_data.append(temp_price)

    # 合并所有股票的数据
    factor_df = pd.concat(factor_data, ignore_index=True)
    price_df = pd.concat(price_data, ignore_index=True)

    # 设置 MultiIndex
    factor_df = factor_df.set_index(['datetime', 'instrument'])
    price_df = price_df.set_index(['datetime', 'instrument'])

    # 删除 NaN
    factor_df = factor_df.dropna()
    price_df = price_df.dropna()

    return factor_df, price_df


def analyze_factor(factor_name, factor_df, price_df, output_dir='./output'):
    """
    分析单个因子

    Args:
        factor_name: 因子名称
        factor_df: 因子数据
        price_df: 价格数据
        output_dir: 输出目录
    """
    print(f"\n{'='*80}")
    print(f"分析因子: {factor_name}")
    print(f"{'='*80}\n")

    # 1. 周期对齐
    print("1. 周期对齐...")
    aligner = CycleAligner()
    factor_aligned, returns_aligned = aligner.align(
        factor_df, price_df, method='default'
    )

    # 打印对齐摘要
    print(f"   - 因子数据点数: {len(factor_aligned)}")
    print(f"   - 收益率数据点数: {len(returns_aligned)}")
    print(f"   - 对齐方式: default (T+1 to T+2)")

    # 2. 准备收益率数据
    print("\n2. 准备收益率数据...")
    # returns_aligned 已经是未来收益率了（通过 cycle_aligner 计算）
    returns_final = returns_aligned

    # 对齐因子和收益率
    common_index = factor_aligned.index.intersection(returns_final.index)
    factor_final = factor_aligned.loc[common_index]
    returns_final = returns_final.loc[common_index]

    print(f"   - 最终数据点数: {len(common_index)}")

    # 3. 性能评估
    print("\n3. 性能评估...")
    evaluator = PerformanceEvaluator()

    # 直接使用 Series（如果是 DataFrame，提取第一列）
    if isinstance(factor_final, pd.DataFrame):
        pred = factor_final.iloc[:, 0]
    else:
        pred = factor_final

    if isinstance(returns_final, pd.DataFrame):
        label = returns_final.iloc[:, 0]
    else:
        label = returns_final

    metrics = evaluator.calculate_all(pred, label)

    print(f"   - IC 均值: {metrics['ic_mean']:.4f}")
    print(f"   - IC 标准差: {metrics['ic_std']:.4f}")
    print(f"   - ICIR: {metrics['icir']:.4f}")
    print(f"   - Rank IC 均值: {metrics['rank_ic_mean']:.4f}")
    print(f"   - Rank ICIR: {metrics['rank_icir']:.4f}")

    # 4. 策略场景分析
    print("\n4. 策略场景分析...")
    strategy_analyzer = StrategyAnalyzer()

    # 看涨策略
    bull_result = strategy_analyzer.analyze_bull_strategy(
        factor_final, returns_final, top_pct=0.2
    )
    print(f"   - 看涨策略年化收益: {bull_result['annual_return']:.2%}")
    print(f"   - 看涨策略夏普比率: {bull_result['sharpe_ratio']:.2f}")

    # 看跌策略
    bear_result = strategy_analyzer.analyze_bear_strategy(
        factor_final, returns_final, bottom_pct=0.2
    )
    print(f"   - 看跌策略年化收益: {bear_result['annual_return']:.2%}")
    print(f"   - 看跌策略夏普比率: {bear_result['sharpe_ratio']:.2f}")

    # 多空策略
    long_short_result = strategy_analyzer.analyze_long_short_strategy(
        factor_final, returns_final, top_pct=0.2, bottom_pct=0.2
    )
    print(f"   - 多空策略年化收益: {long_short_result['annual_return']:.2%}")
    print(f"   - 多空策略夏普比率: {long_short_result['sharpe_ratio']:.2f}")

    scenario_results = {
        'bull': bull_result,
        'bear': bear_result,
        'long_short': long_short_result
    }

    # 5. 可靠性评估
    print("\n5. 可靠性评估...")
    reliability_evaluator = ReliabilityEvaluator(weights=DEFAULT_WEIGHTS)
    evaluation = reliability_evaluator.evaluate(metrics, scenario_results)

    print(f"   - 综合评分: {evaluation['total_score']:.2f}")
    print(f"   - 可靠性等级: {evaluation['reliability']}")
    print(f"   - 建议: {evaluation['recommendation']}")

    # 6. 生成报告
    print("\n6. 生成报告...")
    import os
    os.makedirs(output_dir, exist_ok=True)

    report_gen = ReportGenerator(output_dir=output_dir)
    visualizer = Visualizer(output_dir=output_dir)

    # 生成文本报告
    reports = report_gen.generate_full_report(
        factor_name=factor_name,
        metrics=metrics,
        scenario_results=scenario_results,
        reliability_result=evaluation
    )

    # 保存 Markdown 报告（不添加 output_dir，因为 ReportGenerator 已经处理了）
    md_filename = f'{factor_name}_report.md'
    report_gen.save_markdown(reports['markdown'], md_filename)
    print(f"   - Markdown 报告已保存: {md_filename}")

    # 保存 HTML 报告
    html_filename = f'{factor_name}_report.html'
    report_gen.save_html(reports['html'], html_filename)
    print(f"   - HTML 报告已保存: {html_filename}")

    # 生成图表
    print("\n7. 生成图表...")
    charts = visualizer.save_all_charts(
        metrics, scenario_results, factor_name
    )
    print(f"   - 已生成 {len(charts)} 个图表")

    return metrics, scenario_results, evaluation


def main():
    """主函数"""
    print("\n" + "="*80)
    print("KDJ 和 RSI 金叉死叉因子测试")
    print("="*80)

    # 1. 创建模拟数据
    print("\n正在创建模拟数据...")
    data_dict = create_mock_data(
        instruments=['STOCK001', 'STOCK002', 'STOCK003', 'STOCK004', 'STOCK005'],
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    print(f"已创建 {len(data_dict)} 只股票的模拟数据")

    # 2. 测试 KDJ 金叉死叉因子
    print("\n" + "="*80)
    print("测试 KDJ 金叉死叉因子")
    print("="*80)

    factor_kdj_df, price_df = prepare_factor_data(
        data_dict,
        lambda df: calculate_kdj_cross_signal(df, n=9, m1=3, m2=3),
        'KDJ'
    )

    kdj_metrics, kdj_scenarios, kdj_eval = analyze_factor(
        'KDJ_Cross',
        factor_kdj_df,
        price_df,
        output_dir='./output/kdj'
    )

    # 3. 测试 RSI 金叉死叉因子
    print("\n" + "="*80)
    print("测试 RSI 金叉死叉因子")
    print("="*80)

    factor_rsi_df, price_df = prepare_factor_data(
        data_dict,
        lambda df: calculate_rsi_cross_signal(df, fast_n=6, slow_n=14),
        'RSI'
    )

    rsi_metrics, rsi_scenarios, rsi_eval = analyze_factor(
        'RSI_Cross',
        factor_rsi_df,
        price_df,
        output_dir='./output/rsi'
    )

    # 4. 对比分析
    print("\n" + "="*80)
    print("因子对比分析")
    print("="*80)

    comparison = pd.DataFrame({
        'KDJ_Cross': [
            kdj_metrics['ic_mean'],
            kdj_metrics['icir'],
            kdj_metrics['rank_ic_mean'],
            kdj_scenarios['bull']['annual_return'],
            kdj_eval['total_score']
        ],
        'RSI_Cross': [
            rsi_metrics['ic_mean'],
            rsi_metrics['icir'],
            rsi_metrics['rank_ic_mean'],
            rsi_scenarios['bull']['annual_return'],
            rsi_eval['total_score']
        ]
    }, index=['IC 均值', 'ICIR', 'Rank IC', '年化收益', '综合评分'])

    print("\n因子对比:")
    print(comparison.round(4))

    # 5. 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    print("\nKDJ 金叉死叉因子:")
    print(f"  - IC 均值: {kdj_metrics['ic_mean']:.4f}")
    print(f"  - ICIR: {kdj_metrics['icir']:.4f}")
    print(f"  - 可靠性: {kdj_eval['reliability']} ({kdj_eval['total_score']:.2f}分)")

    print("\nRSI 金叉死叉因子:")
    print(f"  - IC 均值: {rsi_metrics['ic_mean']:.4f}")
    print(f"  - ICIR: {rsi_metrics['icir']:.4f}")
    print(f"  - 可靠性: {rsi_eval['reliability']} ({rsi_eval['total_score']:.2f}分)")

    # 判断哪个因子更好
    if kdj_eval['total_score'] > rsi_eval['total_score']:
        print("\n结论: KDJ 金叉死叉因子表现更好! 🏆")
    else:
        print("\n结论: RSI 金叉死叉因子表现更好! 🏆")

    print("\n" + "="*80)
    print("测试完成! 报告已保存到 output/ 目录")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
