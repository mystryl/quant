"""
周期对齐和策略分析示例

演示如何使用 CycleAligner 和 StrategyAnalyzer 进行因子分析
"""

import numpy as np
import pandas as pd
from typing import Tuple
import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.core.cycle_aligner import CycleAligner
from src.core.strategy_analyzer import StrategyAnalyzer


def generate_mock_data(
    n_dates: int = 252,
    n_stocks: int = 100,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    生成模拟数据用于演示

    Args:
        n_dates: 交易日数量
        n_stocks: 股票数量
        seed: 随机种子

    Returns:
        (price_df, factor_df, market_index)
    """
    np.random.seed(seed)

    # 生成日期
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')

    # 生成股票代码
    stocks = [f'STOCK_{i:04d}' for i in range(n_stocks)]

    # 创建 MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, stocks],
        names=['datetime', 'instrument']
    )

    # 生成价格数据（随机游走）
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.02, size=(n_dates, n_stocks))
    returns = pd.DataFrame(returns, index=dates, columns=stocks)

    prices = initial_price * (1 + returns).cumprod()
    price_df = prices.stack().reindex(index).to_frame('close')

    # 生成因子数据（动量因子：过去 20 日收益率）
    factor = returns.rolling(window=20).mean().stack().reindex(index)

    # 添加一些噪声
    factor += np.random.normal(0, 0.01, size=len(index))

    factor_df = factor.to_frame('momentum')

    # 生成市场指数
    market_returns = returns.mean(axis=1)
    market_index = (1 + market_returns).cumprod()
    market_index.name = 'market_index'

    return price_df, factor_df, market_index


def example_1_default_alignment():
    """示例 1: 使用默认对齐方式（Qlib T+1 to T+2）"""
    print("=" * 80)
    print("示例 1: 默认对齐方式（Qlib T+1 to T+2）")
    print("=" * 80)

    # 生成模拟数据
    price_df, factor_df, _ = generate_mock_data()

    # 创建对齐器
    aligner = CycleAligner()

    # 默认对齐
    factor_aligned, returns_aligned = aligner.align(
        factor_df,
        price_df,
        method='default'
    )

    print(f"原始数据形状: {factor_df.shape}")
    print(f"对齐后数据形状: {factor_aligned.shape}")
    print(f"\n因子统计:")
    print(factor_aligned.describe())
    print(f"\n收益率统计:")
    print(returns_aligned.describe())

    # 验证对齐结果
    validation = aligner.validate_alignment(factor_aligned, returns_aligned)
    print(f"\n验证结果:")
    print(f"  - 是否有效: {validation['is_valid']}")
    print(f"  - 因子 NaN 比例: {validation['statistics']['factor_nan_ratio']:.2%}")
    print(f"  - 收益率 NaN 比例: {validation['statistics']['returns_nan_ratio']:.2%}")

    if validation['warnings']:
        print(f"  - 警告: {validation['warnings']}")

    print()


def example_2_flexible_alignment():
    """示例 2: 使用灵活对齐方式（自定义偏移量）"""
    print("=" * 80)
    print("示例 2: 灵活对齐方式（自定义偏移量）")
    print("=" * 80)

    # 生成模拟数据
    price_df, factor_df, _ = generate_mock_data()

    # 创建对齐器
    aligner = CycleAligner()

    # 尝试不同的偏移量
    for shift in [1, 2, 5, 10]:
        factor_aligned, returns_aligned = aligner.align(
            factor_df,
            price_df,
            method='flexible',
            shift=shift
        )

        # 计算 IC
        ic = aligner._calculate_ic(factor_aligned, returns_aligned)

        print(f"Shift={shift} (T+{shift-1} to T+{shift}):")
        print(f"  - IC 均值: {ic.mean():.4f}")
        print(f"  - IC 标准差: {ic.std():.4f}")
        print(f"  - ICIR: {ic.mean()/ic.std():.4f}")
        print()


def example_3_auto_alignment():
    """示例 3: 自动检测最优对齐方式"""
    print("=" * 80)
    print("示例 3: 自动检测最优对齐方式")
    print("=" * 80)

    # 生成模拟数据
    price_df, factor_df, _ = generate_mock_data()

    # 创建对齐器
    aligner = CycleAligner()

    # 自动检测
    factor_aligned, returns_aligned, best_shift = aligner.align(
        factor_df,
        price_df,
        method='auto',
        auto_search_range=(1, 10),
        ic_calc_method='pearson'
    )

    print(f"自动检测最优偏移量: {best_shift}")
    print(f"对齐方式: T+{best_shift-1} to T+{best_shift}")

    # 计算 IC
    ic = aligner._calculate_ic(factor_aligned, returns_aligned)
    print(f"IC 均值: {ic.mean():.4f}")
    print(f"IC 标准差: {ic.std():.4f}")
    print(f"ICIR: {ic.mean()/ic.std():.4f}")

    # 获取对齐摘要
    summary = aligner.get_alignment_summary(factor_df, price_df, best_shift)
    print(f"\n对齐摘要:")
    print(f"  - 原始日期数: {summary['original_dates']}")
    print(f"  - 对齐后日期数: {summary['aligned_dates']}")
    print(f"  - 数据损失: {summary['data_loss']:.2%}")

    print()


def example_4_bull_strategy():
    """示例 4: 看涨策略分析"""
    print("=" * 80)
    print("示例 4: 看涨策略分析")
    print("=" * 80)

    # 生成模拟数据
    price_df, factor_df, _ = generate_mock_data()

    # 对齐数据
    aligner = CycleAligner()
    factor_aligned, returns_aligned = aligner.align(
        factor_df, price_df, method='default'
    )

    # 创建策略分析器
    analyzer = StrategyAnalyzer()

    # 分析看涨策略
    bull_result = analyzer.analyze_bull_strategy(
        factor_aligned,
        returns_aligned,
        top_pct=0.2
    )

    print("看涨策略表现:")
    print(f"  - 总收益率: {bull_result['total_return']:.2%}")
    print(f"  - 年化收益率: {bull_result['annual_return']:.2%}")
    print(f"  - 夏普比率: {bull_result['sharpe_ratio']:.2f}")
    print(f"  - 最大回撤: {bull_result['max_drawdown']:.2%}")
    print(f"  - 胜率: {bull_result['win_rate']:.2%}")
    print(f"  - 卡玛比率: {bull_result['calmar_ratio']:.2f}")

    print()


def example_5_long_short_strategy():
    """示例 5: 多空策略分析"""
    print("=" * 80)
    print("示例 5: 多空策略分析")
    print("=" * 80)

    # 生成模拟数据
    price_df, factor_df, _ = generate_mock_data()

    # 对齐数据
    aligner = CycleAligner()
    factor_aligned, returns_aligned = aligner.align(
        factor_df, price_df, method='default'
    )

    # 创建策略分析器
    analyzer = StrategyAnalyzer()

    # 分析多空策略
    ls_result = analyzer.analyze_long_short_strategy(
        factor_aligned,
        returns_aligned,
        top_pct=0.2
    )

    print("多空策略表现:")
    print(f"  - 总收益率: {ls_result['total_return']:.2%}")
    print(f"  - 年化收益率: {ls_result['annual_return']:.2%}")
    print(f"  - 夏普比率: {ls_result['sharpe_ratio']:.2f}")
    print(f"  - 最大回撤: {ls_result['max_drawdown']:.2%}")
    print(f"  - 胜率: {ls_result['win_rate']:.2%}")

    print("\n多头 vs 空头:")
    print(f"  - 多头年化收益: {ls_result['long_returns'].mean() * 252:.2%}")
    print(f"  - 空头年化收益: {ls_result['short_returns'].mean() * 252:.2%}")

    print()


def example_6_volatility_strategy():
    """示例 6: 波动率策略分析"""
    print("=" * 80)
    print("示例 6: 波动率策略分析")
    print("=" * 80)

    # 生成模拟数据
    price_df, factor_df, _ = generate_mock_data()

    # 对齐数据
    aligner = CycleAligner()
    factor_aligned, returns_aligned = aligner.align(
        factor_df, price_df, method='default'
    )

    # 创建策略分析器
    analyzer = StrategyAnalyzer()

    # 分析波动率策略
    vol_result = analyzer.analyze_volatility_strategy(
        factor_aligned,
        returns_aligned,
        price_df,
        top_pct=0.2,
        position_method='inverse'
    )

    print("波动率策略表现:")
    print(f"  - 总收益率: {vol_result['total_return']:.2%}")
    print(f"  - 年化收益率: {vol_result['annual_return']:.2%}")
    print(f"  - 夏普比率: {vol_result['sharpe_ratio']:.2f}")
    print(f"  - 最大回撤: {vol_result['max_drawdown']:.2%}")

    print("\n与基础策略对比:")
    base_annual = vol_result['base_returns'].mean() * 252
    vol_annual = vol_result['strategy_returns'].mean() * 252
    print(f"  - 基础策略年化收益: {base_annual:.2%}")
    print(f"  - 波动率调整后年化收益: {vol_annual:.2%}")
    print(f"  - 收益提升: {(vol_annual - base_annual):.2%}")

    print()


def example_7_market_regime():
    """示例 7: 牛熊市场景分析"""
    print("=" * 80)
    print("示例 7: 牛熊市场景分析")
    print("=" * 80)

    # 生成模拟数据
    price_df, factor_df, market_index = generate_mock_data()

    # 对齐数据
    aligner = CycleAligner()
    factor_aligned, returns_aligned = aligner.align(
        factor_df, price_df, method='default'
    )

    # 创建策略分析器
    analyzer = StrategyAnalyzer()

    # 分析牛熊市表现
    regime_result = analyzer.analyze_market_regime(
        factor_aligned,
        returns_aligned,
        market_index,
        bull_threshold=0.0,
        window=20
    )

    print("牛熊市表现对比:")
    for regime_name, metrics in regime_result.items():
        print(f"\n{regime_name}:")
        print(f"  - 交易日数: {metrics['days']}")
        print(f"  - 总收益率: {metrics['total_return']:.2%}")
        print(f"  - 夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"  - 胜率: {metrics['win_rate']:.2%}")
        print(f"  - 平均日收益: {metrics['avg_daily_return']:.4%}")

    print()


def example_8_all_scenarios():
    """示例 8: 分析所有策略场景"""
    print("=" * 80)
    print("示例 8: 分析所有策略场景")
    print("=" * 80)

    # 生成模拟数据
    price_df, factor_df, market_index = generate_mock_data()

    # 对齐数据
    aligner = CycleAligner()
    factor_aligned, returns_aligned = aligner.align(
        factor_df, price_df, method='default'
    )

    # 创建策略分析器
    analyzer = StrategyAnalyzer()

    # 分析所有场景
    all_results = analyzer.analyze_all_scenarios(
        factor_aligned,
        returns_aligned,
        price_df,
        market_index=market_index,
        top_pct=0.2
    )

    print("策略场景汇总:")
    print("-" * 80)

    # 看涨策略
    print("\n1. 看涨策略:")
    bull = all_results['bull']
    print(f"   年化收益: {bull['annual_return']:.2%}, 夏普: {bull['sharpe_ratio']:.2f}")

    # 看跌策略
    print("\n2. 看跌策略:")
    bear = all_results['bear']
    print(f"   年化收益: {bear['annual_return']:.2%}, 夏普: {bear['sharpe_ratio']:.2f}")

    # 多空策略
    print("\n3. 多空策略:")
    ls = all_results['long_short']
    print(f"   年化收益: {ls['annual_return']:.2%}, 夏普: {ls['sharpe_ratio']:.2f}")

    # 波动率策略
    print("\n4. 波动率策略:")
    vol = all_results['volatility']
    print(f"   年化收益: {vol['annual_return']:.2%}, 夏普: {vol['sharpe_ratio']:.2f}")

    # 牛熊市
    print("\n5. 牛熊市表现:")
    for regime_name, metrics in all_results['market_regime'].items():
        print(f"   {regime_name}: 年化收益 {metrics['total_return']:.2%}, 夏普 {metrics['sharpe_ratio']:.2f}")

    print()


def example_9_convenient_functions():
    """示例 9: 使用便捷函数"""
    print("=" * 80)
    print("示例 9: 使用便捷函数")
    print("=" * 80)

    # 生成模拟数据
    price_df, factor_df, _ = generate_mock_data()

    # 使用便捷函数对齐
    from src.core.cycle_aligner import align_factor_returns

    factor_aligned, returns_aligned = align_factor_returns(
        factor_df,
        price_df,
        method='default'
    )

    print(f"对齐完成，数据形状: {factor_aligned.shape}")

    # 使用便捷函数分析策略
    from src.core.strategy_analyzer import analyze_factor_strategies

    results = analyze_factor_strategies(
        factor_aligned,
        returns_aligned,
        price_df,
        top_pct=0.2
    )

    print("\n快速分析结果:")
    print(f"  - 看涨策略夏普: {results['bull']['sharpe_ratio']:.2f}")
    print(f"  - 多空策略夏普: {results['long_short']['sharpe_ratio']:.2f}")
    print(f"  - 波动率策略夏普: {results['volatility']['sharpe_ratio']:.2f}")

    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("周期对齐和策略分析示例集")
    print("=" * 80 + "\n")

    examples = [
        ("默认对齐", example_1_default_alignment),
        ("灵活对齐", example_2_flexible_alignment),
        ("自动对齐", example_3_auto_alignment),
        ("看涨策略", example_4_bull_strategy),
        ("多空策略", example_5_long_short_strategy),
        ("波动率策略", example_6_volatility_strategy),
        ("牛熊市场景", example_7_market_regime),
        ("所有场景", example_8_all_scenarios),
        ("便捷函数", example_9_convenient_functions),
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"示例 {i} ({name}) 出错: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print("所有示例运行完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
