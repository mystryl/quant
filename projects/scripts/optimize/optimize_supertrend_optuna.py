#!/usr/bin/env python3
"""
使用 Optuna 优化 SuperTrend 策略参数

贝叶斯优化比网格搜索更高效，能够更快找到最优参数组合。

当前基准参数（SF14Re）：
- period: 20
- multiplier: 2
- n: 1.5
- trailing_stop_rate: 80
- max_holding_period: 100
- min_liqka: 0.5
- max_liqka: 1.0

优化搜索空间围绕这些参数展开。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import optuna.visualization as vis
from qlib_supertrend_enhanced import SupertrendEnhancedStrategy, run_backtest


def load_data_multi_freq(freq="1min", start_date="2023-01-01", end_date="2023-12-31"):
    """加载 qlib 数据（支持多频率）"""
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

    if len(data) == 0:
        return None

    df = pd.DataFrame(data)
    df = df.sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    return df


def objective(trial, data, strategy_name="supertrend_enhanced"):
    """
    Optuna 优化目标函数

    Args:
        trial: Optuna trial 对象
        data: 回测数据
        strategy_name: 策略名称

    Returns:
        优化目标值（最小化负夏普比率 = 最大化夏普比率）
    """
    # 定义参数搜索空间（只优化这三个参数）
    # period: 20 -> [12, 28]
    period = trial.suggest_int('period', 12, 28)

    # multiplier: 2 -> [1.5, 2.5]
    multiplier = trial.suggest_float('multiplier', 1.5, 2.5)

    # n: 1.5 -> [1, 2] (整数)
    n = trial.suggest_int('n', 1, 2)

    # 以下参数固定不变
    trailing_stop_rate = 80
    max_holding_period = 100
    min_liqka = 0.5
    max_liqka = 1.0

    # 创建策略
    strategy = SupertrendEnhancedStrategy(
        period=period,
        multiplier=multiplier,
        n=n,
        trailing_stop_rate=trailing_stop_rate,
        max_holding_period=max_holding_period,
        min_liqka=min_liqka,
        max_liqka=max_liqka
    )

    # 生成信号
    df_strategy = strategy.generate_signal(data.copy())
    df_strategy = df_strategy.dropna(subset=['position'])

    if len(df_strategy) == 0:
        # 如果没有有效信号，返回最差值
        return float('inf')

    # 运行回测
    results = run_backtest(df_strategy, strategy.name)

    # 记录附加指标
    trial.set_user_attr('cumulative_return', results['cumulative_return'])
    trial.set_user_attr('max_drawdown', results['max_drawdown'])
    trial.set_user_attr('total_trades', results['total_trades'])
    trial.set_user_attr('win_rate', results['win_rate'])

    # 返回优化目标（最小化负夏普比率 = 最大化夏普比率）
    return -results['sharpe_ratio']


def optimize_with_optuna(freq="15min", years=[2023, 2024, 2025],
                         n_trials=100, timeout=None, objective_metric='sharpe_ratio'):
    """
    使用 Optuna 优化 SuperTrend 参数

    Args:
        freq: 测试频率
        years: 测试年份列表
        n_trials: 最大试验次数
        timeout: 最大优化时间（秒）
        objective_metric: 优化目标指标
    """
    print("="*80)
    print("SuperTrend 策略参数优化（Optuna 贝叶斯优化）")
    print("="*80)

    print(f"\n优化配置：")
    print(f"   频率: {freq}")
    print(f"   年份: {years}")
    print(f"   最大试验次数: {n_trials}")
    print(f"   优化目标: {objective_metric}")
    print(f"\n   基准参数：")
    print(f"      period=20, multiplier=2, n=1.5")
    print(f"\n   固定参数：")
    print(f"      trailing_stop_rate=80")
    print(f"      max_holding_period=100")
    print(f"      min_liqka=0.5, max_liqka=1.0")
    print(f"\n   搜索空间：")
    print(f"      period: [12, 28]")
    print(f"      multiplier: [1.5, 2.5]")
    print(f"      n: [1, 2]")

    # 创建优化研究
    study = optuna.create_study(
        direction='minimize',  # 最小化负夏普比率
        sampler=optuna.samplers.TPESampler(seed=42),  # TPE 采样器
        pruner=optuna.pruners.MedianPruner()  # 中位数剪枝器
    )

    # 加载多年度数据并合并
    print(f"\n加载数据...")
    all_data = []
    for year in years:
        df = load_data_multi_freq(freq=freq,
                                  start_date=f"{year}-01-01",
                                  end_date=f"{year}-12-31")
        if df is not None and len(df) > 0:
            all_data.append(df)
            print(f"   {year} 年: {len(df)} 行")

    if len(all_data) == 0:
        print(f"\n   ⚠️  没有有效数据")
        return None

    # 合并所有年度数据
    data = pd.concat(all_data)
    data = data.sort_index()
    print(f"\n   合并后总数据: {len(data)} 行")

    # 包装目标函数
    def wrapped_objective(trial):
        return objective(trial, data)

    # 运行优化
    print(f"\n开始优化...")
    print(f"   使用 TPE 算法进行贝叶斯搜索")
    print(f"   使用中位数剪枝器提前终止无效试验")

    study.optimize(wrapped_objective,
                  n_trials=n_trials,
                  timeout=timeout,
                  show_progress_bar=True)

    # 输出结果
    print(f"\n{'='*80}")
    print("优化完成！")
    print(f"{'='*80}")

    print(f"\n最优参数：")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    best_trial = study.best_trial
    print(f"\n最优回测结果：")
    print(f"   夏普比率: {-best_trial.value:.2f}")
    print(f"   累计收益: {best_trial.user_attrs['cumulative_return']:.2%}")
    print(f"   最大回撤: {best_trial.user_attrs['max_drawdown']:.2%}")
    print(f"   交易次数: {best_trial.user_attrs['total_trades']}")
    print(f"   胜率: {best_trial.user_attrs['win_rate']:.2f}%")

    print(f"\n优化统计：")
    print(f"   总试验次数: {len(study.trials)}")
    print(f"   完成试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"   剪枝试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"   失败试验: {len([t for t in study.trials if t.state == optique.trial.TrialState.FAIL])}")

    # 保存结果
    print(f"\n{'='*80}")
    print("保存优化结果...")
    print(f"{'='*80}")

    # 保存最优参数
    import json
    params_file = Path("/mnt/d/quant/qlib_backtest/supertrend_optuna_best_params.json")
    with open(params_file, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': -study.best_value,
            'best_trial_attrs': best_trial.user_attrs,
            'optimization_config': {
                'freq': freq,
                'years': years,
                'n_trials': n_trials,
                'objective_metric': objective_metric
            }
        }, f, indent=2)
    print(f"   最优参数已保存到: {params_file}")

    # 保存所有试验结果
    results_df = study.trials_dataframe()
    results_file = Path("/mnt/d/quant/qlib_backtest/supertrend_optuna_trials.csv")
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"   所有试验结果已保存到: {results_file}")

    # 生成可视化图表（如果需要）
    try:
        # 参数重要性图
        fig = vis.plot_param_importances(study)
        fig_file = Path("/mnt/d/quant/qlib_backtest/supertrend_optuna_importances.png")
        fig.write_image(str(fig_file))
        print(f"   参数重要性图已保存到: {fig_file}")

        # 优化历史图
        fig = vis.plot_optimization_history(study)
        fig_file = Path("/mnt/d/quant/qlib_backtest/supertrend_optuna_history.png")
        fig.write_image(str(fig_file))
        print(f"   优化历史图已保存到: {fig_file}")

        # 参数关系图
        fig = vis.plot_parallel_coordinate(study)
        fig_file = Path("/mnt/d/quant/qlib_backtest/supertrend_optuna_parallel.png")
        fig.write_image(str(fig_file))
        print(f"   参数关系图已保存到: {fig_file}")

    except Exception as e:
        print(f"   ⚠️  生成可视化图表失败: {e}")

    return study


def main():
    """主程序"""
    # 优化配置
    freq = "15min"
    years = [2023, 2024, 2025]
    n_trials = 100  # 增加试验次数以获得更好的结果

    # 运行优化
    study = optimize_with_optuna(freq=freq, years=years, n_trials=n_trials)

    if study is not None:
        print(f"\n{'='*80}")
        print("参数优化完成！")
        print(f"{'='*80}")

        # 可以使用最优参数运行详细回测
        print(f"\n💡 提示：可以使用最优参数运行详细回测：")
        print(f"   python backtest_enhanced_custom_params.py \\\n"
              f"       --period {study.best_params['period']} \\\n"
              f"       --multiplier {study.best_params['multiplier']:.2f} \\\n"
              f"       --n {study.best_params['n']} \\\n"
              f"       --trailing_stop_rate {study.best_params['trailing_stop_rate']}")

    return study


if __name__ == "__main__":
    study = main()
