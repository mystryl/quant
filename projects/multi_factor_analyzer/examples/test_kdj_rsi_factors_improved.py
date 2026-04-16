"""
KDJ 和 RSI 金叉死叉因子测试（改进版）

主要改进：
1. ✅ 修复所有已知的 bug
2. ✅ 添加完整的错误处理
3. ✅ 添加数据验证
4. ✅ 修复 KDJ 除零错误
5. ✅ 优化性能
6. ✅ 添加详细日志

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
import os
import logging
from typing import Dict, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加路径
sys.path.append('/Users/mystryl/Documents/Quant/projects/multi_factor_analyzer')

from src.core import PerformanceEvaluator, CycleAligner, StrategyAnalyzer
from src.core.reliability import ReliabilityEvaluator
from src.core.config import DEFAULT_WEIGHTS
from src.report import ReportGenerator, Visualizer


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
        high = price * (1 + np.abs(np.random.uniform(0, 0.02, len(dates))))
        low = price * (1 - np.abs(np.random.uniform(0, 0.02, len(dates))))

        df = pd.DataFrame({
            'open': price * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)

        # 确保 high >= close >= low
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.uniform(0, 0.01, len(dates))))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.uniform(0, 0.01, len(dates))))

        data[instrument] = df

    logger.info(f"✓ 创建了 {len(data)} 只股票的模拟数据")
    logger.info(f"  时间范围: {dates[0].date()} 到 {dates[-1].date()}")
    logger.info(f"  总数据点: {len(dates) * len(instruments):,}")

    return data


def validate_data(data_dict: Dict) -> bool:
    """
    验证数据有效性

    Args:
        data_dict: 股票数据字典

    Returns:
        bool: 数据是否有效
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']

    for instrument, df in data_dict.items():
        # 检查必需列
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"❌ {instrument} 缺少必需列: {missing_cols}")
            return False

        # 检查数据逻辑
        if (df['high'] < df['low']).any():
            logger.error(f"❌ {instrument} 存在 high < low 的异常数据")
            return False

        if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
            logger.error(f"❌ {instrument} close 超出 high-low 范围")
            return False

        # 检查缺失值
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"⚠️ {instrument} 存在缺失值: {null_counts.to_dict()}")

    logger.info("✓ 数据验证通过")
    return True


def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
                  n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算 KDJ 指标（改进版，避免除零错误）

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

    # 避免除零错误
    denominator = high_n - low_n
    rsv = np.where(
        denominator != 0,
        (close - low_n) / denominator * 100,
        50  # 当 high == low 时，设为中性值
    )
    rsv = pd.Series(rsv, index=close.index)

    # 计算 K 值 (SMA of RSV)
    K = rsv.ewm(alpha=1/m1, adjust=False).mean()

    # 计算 D 值 (SMA of K)
    D = K.ewm(alpha=1/m2, adjust=False).mean()

    # 计算 J 值
    J = 3 * K - 2 * D

    return K, D, J


def calculate_rsi(close: pd.Series, n: int = 14) -> pd.Series:
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

    # 避免除零
    rs = np.where(avg_losses != 0, avg_gains / avg_losses, 0)
    rsi = 100 - (100 / (1 + rs))

    return pd.Series(rsi, index=close.index)


def calculate_kdj_cross_signal(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.Series:
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
    golden_cross = (
        (J > K) & (J.shift(1) <= K.shift(1)) &
        (J > D) & (J.shift(1) <= D.shift(1)) &
        (K < 20) & (D < 20)
    )

    death_cross = (
        (J < K) & (J.shift(1) >= K.shift(1)) &
        (J < D) & (J.shift(1) >= D.shift(1)) &
        (K > 80) & (D > 80)
    )

    # 生成因子值
    factor = pd.Series(0.0, index=close.index)
    factor[golden_cross] = J[golden_cross]
    factor[death_cross] = -J[death_cross]

    # 对于其他情况，使用 (J - 50) / 50 作为因子值
    mask = (factor == 0) & (~golden_cross) & (~death_cross)
    factor[mask] = (J - 50) / 50

    return factor


def calculate_rsi_cross_signal(df: pd.DataFrame, fast_n: int = 6, slow_n: int = 14) -> pd.Series:
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
    golden_cross = (
        (rsi_fast > rsi_slow) & (rsi_fast.shift(1) <= rsi_slow.shift(1)) &
        (rsi_fast < 30)
    )

    death_cross = (
        (rsi_fast < rsi_slow) & (rsi_fast.shift(1) >= rsi_slow.shift(1)) &
        (rsi_fast > 70)
    )

    # 生成因子值
    factor = pd.Series(0.0, index=close.index)
    factor[golden_cross] = (rsi_fast[golden_cross] - 50) / 50
    factor[death_cross] = -(rsi_fast[death_cross] - 50) / 50

    # 对于其他情况，使用 (RSI快线 - RSI慢线) / 50
    mask = (factor == 0) & (~golden_cross) & (~death_cross)
    factor[mask] = (rsi_fast[mask] - rsi_slow[mask]) / 50

    return factor


def prepare_factor_data(data_dict: Dict, factor_func, factor_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    准备因子数据（改进版，性能更优）

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
        try:
            # 计算因子
            factor = factor_func(df)

            # 移除 NaN
            factor = factor.dropna()

            # 创建 MultiIndex (用 from_arrays 避免 from_product 的对齐问题)
            n = len(factor)
            index = pd.MultiIndex.from_arrays(
                [factor.index, [instrument] * n],
                names=['datetime', 'instrument']
            )

            factor_data.append(pd.DataFrame({'factor': factor.values}, index=index))
            price_data.append(pd.DataFrame({'close': df.loc[factor.index, 'close'].values}, index=index))

        except Exception as e:
            logger.warning(f"⚠️ {instrument} 因子计算失败: {e}")
            continue

    # 合并所有股票的数据
    if not factor_data:
        raise ValueError(f"没有成功计算任何 {factor_name} 因子数据")

    factor_df = pd.concat(factor_data)
    price_df = pd.concat(price_data)

    # 删除 NaN
    factor_df = factor_df.dropna()
    price_df = price_df.dropna()

    logger.info(f"✓ {factor_name} 因子数据准备完成: {len(factor_df)} 个数据点")

    return factor_df, price_df


def analyze_factor(factor_name: str, factor_df: pd.DataFrame, price_df: pd.DataFrame,
                  output_dir: str = './output') -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    分析单个因子（改进版，包含完整错误处理）

    Args:
        factor_name: 因子名称
        factor_df: 因子数据
        price_df: 价格数据
        output_dir: 输出目录

    Returns:
        metrics, scenario_results, evaluation 或 (None, None, None) 如果失败
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"分析因子: {factor_name}")
        logger.info(f"{'='*80}\n")

        # 1. 周期对齐
        logger.info("1. 周期对齐...")
        aligner = CycleAligner()
        factor_aligned, returns_aligned = aligner.align(
            factor_df, price_df, method='default'
        )

        logger.info(f"   ✓ 因子数据点数: {len(factor_aligned):,}")
        logger.info(f"   ✓ 收益率数据点数: {len(returns_aligned):,}")
        logger.info(f"   ✓ 对齐方式: default (T+1 to T+2)")

        # 2. 准备收益率数据
        logger.info("\n2. 准备收益率数据...")
        returns_final = returns_aligned
        common_index = factor_aligned.index.intersection(returns_final.index)
        factor_final = factor_aligned.loc[common_index]
        returns_final = returns_final.loc[common_index]

        logger.info(f"   ✓ 最终数据点数: {len(common_index):,}")

        # 3. 性能评估
        logger.info("\n3. 性能评估...")
        evaluator = PerformanceEvaluator()

        # 确保 Series 格式
        pred = factor_final.iloc[:, 0] if isinstance(factor_final, pd.DataFrame) else factor_final
        label = returns_final.iloc[:, 0] if isinstance(returns_final, pd.DataFrame) else returns_final

        metrics = evaluator.calculate_all(pred, label)

        logger.info(f"   ✓ IC 均值: {metrics['ic_mean']:.4f}")
        logger.info(f"   ✓ IC 标准差: {metrics['ic_std']:.4f}")
        logger.info(f"   ✓ ICIR: {metrics['icir']:.4f}")
        logger.info(f"   ✓ Rank IC 均值: {metrics['rank_ic_mean']:.4f}")
        logger.info(f"   ✓ Rank ICIR: {metrics['rank_icir']:.4f}")

        # 4. 策略场景分析
        logger.info("\n4. 策略场景分析...")
        strategy_analyzer = StrategyAnalyzer()

        bull_result = strategy_analyzer.analyze_bull_strategy(
            factor_final, returns_final, top_pct=0.2
        )
        logger.info(f"   ✓ 看涨策略年化收益: {bull_result['annual_return']:.2%}")
        logger.info(f"   ✓ 看涨策略夏普比率: {bull_result['sharpe_ratio']:.2f}")

        bear_result = strategy_analyzer.analyze_bear_strategy(
            factor_final, returns_final, bottom_pct=0.2
        )
        logger.info(f"   ✓ 看跌策略年化收益: {bear_result['annual_return']:.2%}")
        logger.info(f"   ✓ 看跌策略夏普比率: {bear_result['sharpe_ratio']:.2f}")

        long_short_result = strategy_analyzer.analyze_long_short_strategy(
            factor_final, returns_final, top_pct=0.2, bottom_pct=0.2
        )
        logger.info(f"   ✓ 多空策略年化收益: {long_short_result['annual_return']:.2%}")
        logger.info(f"   ✓ 多空策略夏普比率: {long_short_result['sharpe_ratio']:.2f}")

        scenario_results = {
            'bull': bull_result,
            'bear': bear_result,
            'long_short': long_short_result
        }

        # 5. 可靠性评估
        logger.info("\n5. 可靠性评估...")
        reliability_evaluator = ReliabilityEvaluator(weights=DEFAULT_WEIGHTS)
        evaluation = reliability_evaluator.evaluate(metrics, scenario_results)

        logger.info(f"   ✓ 综合评分: {evaluation['total_score']:.2f}")
        logger.info(f"   ✓ 可靠性等级: {evaluation['reliability']}")
        logger.info(f"   ✓ 建议: {evaluation['recommendation'][:50]}...")

        # 6. 生成报告
        logger.info("\n6. 生成报告...")
        os.makedirs(output_dir, exist_ok=True)

        report_gen = ReportGenerator(output_dir=output_dir)
        visualizer = Visualizer(output_dir=output_dir)

        # 生成报告
        reports = report_gen.generate_full_report(
            factor_name=factor_name,
            metrics=metrics,
            scenario_results=scenario_results,
            reliability_result=evaluation
        )

        # 保存报告
        md_filename = f'{factor_name}_report.md'
        report_gen.save_markdown(reports['markdown'], md_filename)
        logger.info(f"   ✓ Markdown 报告: {md_filename}")

        html_filename = f'{factor_name}_report.html'
        report_gen.save_html(reports['html'], html_filename)
        logger.info(f"   ✓ HTML 报告: {html_filename}")

        # 7. 生成图表
        logger.info("\n7. 生成图表...")
        charts = visualizer.save_all_charts(
            metrics, scenario_results, factor_name
        )
        logger.info(f"   ✓ 已生成 {len(charts)} 个图表")

        return metrics, scenario_results, evaluation

    except Exception as e:
        logger.error(f"❌ 因子分析失败 ({factor_name}): {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    """主函数（改进版）"""
    logger.info("\n" + "="*80)
    logger.info("KDJ 和 RSI 金叉死叉因子测试（改进版）")
    logger.info("="*80)

    # 1. 创建模拟数据
    logger.info("\n正在创建模拟数据...")
    data_dict = create_mock_data(
        instruments=['STOCK001', 'STOCK002', 'STOCK003', 'STOCK004', 'STOCK005'],
        start_date='2020-01-01',
        end_date='2023-12-31'
    )

    # 2. 验证数据
    if not validate_data(data_dict):
        logger.error("❌ 数据验证失败，退出")
        return

    # 3. 测试 KDJ 金叉死叉因子
    logger.info("\n" + "="*80)
    logger.info("测试 KDJ 金叉死叉因子")
    logger.info("="*80)

    try:
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

        if kdj_metrics is None:
            logger.error("❌ KDJ 因子分析失败")
            return

    except Exception as e:
        logger.error(f"❌ KDJ 因子测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 测试 RSI 金叉死叉因子
    logger.info("\n" + "="*80)
    logger.info("测试 RSI 金叉死叉因子")
    logger.info("="*80)

    try:
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

        if rsi_metrics is None:
            logger.error("❌ RSI 因子分析失败")
            return

    except Exception as e:
        logger.error(f"❌ RSI 因子测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 对比分析
    logger.info("\n" + "="*80)
    logger.info("因子对比分析")
    logger.info("="*80)

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

    logger.info("\n因子对比:")
    logger.info(f"\n{comparison.round(4)}")

    # 6. 总结
    logger.info("\n" + "="*80)
    logger.info("测试总结")
    logger.info("="*80)

    logger.info(f"\nKDJ 金叉死叉因子:")
    logger.info(f"  - IC 均值: {kdj_metrics['ic_mean']:.4f}")
    logger.info(f"  - ICIR: {kdj_metrics['icir']:.4f}")
    logger.info(f"  - 可靠性: {kdj_eval['reliability']} ({kdj_eval['total_score']:.2f}分)")

    logger.info(f"\nRSI 金叉死叉因子:")
    logger.info(f"  - IC 均值: {rsi_metrics['ic_mean']:.4f}")
    logger.info(f"  - ICIR: {rsi_metrics['icir']:.4f}")
    logger.info(f"  - 可靠性: {rsi_eval['reliability']} ({rsi_eval['total_score']:.2f}分)")

    # 判断哪个因子更好
    if kdj_eval['total_score'] > rsi_eval['total_score']:
        logger.info("\n结论: KDJ 金叉死叉因子表现更好! 🏆")
    else:
        logger.info("\n结论: RSI 金叉死叉因子表现更好! 🏆")

    logger.info("\n" + "="*80)
    logger.info("✅ 测试完成! 报告已保存到 output/ 目录")
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()
