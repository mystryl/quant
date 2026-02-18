#!/usr/bin/env python3
"""
V03混合策略 - 参数验证器

提供参数优化后的验证功能：
- 过拟合检测
- 参数稳定性测试
- Walk-Forward分析
- 蒙特卡洛模拟
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import random

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入策略模块
from strategies.v03_hybrid.strat_v03_hybrid import (
    HybridMSBOBStrategy,
    calculate_metrics,
    load_data
)

# 导入配置模块
from strategies.v03_hybrid.optimization.config import (
    OPTIMIZATION_CONFIG,
    check_success_criteria
)


# ============================================================================
# 参数验证器
# ============================================================================

class ParameterValidator:
    """参数验证器"""

    def __init__(
        self,
        best_params: Dict,
        train_years: List[int] = None,
        validate_year: int = 2025,
        freq: str = '60min',
        results_dir: Optional[Path] = None
    ):
        """
        初始化验证器

        Args:
            best_params: 最优参数字典
            train_years: 训练集年份
            validate_year: 验证集年份
            freq: 数据频率
            results_dir: 结果保存目录
        """
        self.best_params = best_params
        self.train_years = train_years or OPTIMIZATION_CONFIG['data_split']['train_years']
        self.validate_year = validate_year
        self.freq = freq
        self.results_dir = results_dir or current_dir / "results"

        # 加载数据
        self.train_data = self._load_training_data()
        self.validate_data = load_data(validate_year, freq)

        # 验证结果
        self.validation_results = {}

    def _load_training_data(self) -> pd.DataFrame:
        """加载训练数据"""
        df_list = []
        for year in self.train_years:
            try:
                df = load_data(year, self.freq)
                df_list.append(df)
            except FileNotFoundError:
                pass
        return pd.concat(df_list, ignore_index=True).sort_values('datetime').reset_index(drop=True)

    def run_all_validations(self) -> Dict:
        """运行所有验证测试"""
        print("\n" + "="*80)
        print("参数验证")
        print("="*80)

        # 1. 基础性能验证
        print("\n1. 基础性能验证...")
        self.validation_results['basic'] = self._validate_basic_performance()

        # 2. 过拟合检测
        print("\n2. 过拟合检测...")
        self.validation_results['overfitting'] = self._detect_overfitting()

        # 3. 参数稳定性测试
        print("\n3. 参数稳定性测试...")
        self.validation_results['stability'] = self._test_parameter_stability()

        # 4. Walk-Forward分析
        print("\n4. Walk-Forward分析...")
        self.validation_results['walk_forward'] = self._walk_forward_analysis()

        # 5. 蒙特卡洛模拟
        print("\n5. 蒙特卡洛模拟...")
        self.validation_results['monte_carlo'] = self._monte_carlo_simulation()

        # 生成验证报告
        self._generate_validation_report()

        return self.validation_results

    def _validate_basic_performance(self) -> Dict:
        """基础性能验证"""
        results = {}

        # 训练集
        strategy_train = HybridMSBOBStrategy(**self.best_params)
        df_train = strategy_train.run_strategy(self.train_data.copy())
        metrics_train = calculate_metrics(df_train, freq=self.freq)

        # 验证集
        strategy_val = HybridMSBOBStrategy(**self.best_params)
        df_val = strategy_val.run_strategy(self.validate_data.copy())
        metrics_val = calculate_metrics(df_val, freq=self.freq)

        results['train'] = metrics_train
        results['validation'] = metrics_val

        # 打印
        print(f"\n训练集:")
        print(f"  盈亏比: {metrics_train['profit_loss_ratio']:.2f}")
        print(f"  夏普比率: {metrics_train['sharpe_ratio']:.2f}")
        print(f"  累计收益: {metrics_train['cumulative_return']:.2%}")
        print(f"  胜率: {metrics_train['win_rate']:.2f}%")
        print(f"  最大回撤: {metrics_train['max_drawdown']:.2%}")

        print(f"\n验证集:")
        print(f"  盈亏比: {metrics_val['profit_loss_ratio']:.2f}")
        print(f"  夏普比率: {metrics_val['sharpe_ratio']:.2f}")
        print(f"  累计收益: {metrics_val['cumulative_return']:.2%}")
        print(f"  胜率: {metrics_val['win_rate']:.2f}%")
        print(f"  最大回撤: {metrics_val['max_drawdown']:.2%}")

        # 检查成功标准
        success_passed, failed_items = check_success_criteria(metrics_val, 'success')
        results['success_check'] = {
            'passed': success_passed,
            'failed_items': failed_items
        }

        return results

    def _detect_overfitting(self) -> Dict:
        """过拟合检测"""
        train_m = self.validation_results['basic']['train']
        val_m = self.validation_results['basic']['validation']

        # 计算性能差异
        metrics_to_check = ['profit_loss_ratio', 'sharpe_ratio', 'cumulative_return', 'win_rate']
        diff_threshold = OPTIMIZATION_CONFIG['validation']['overfitting_threshold']

        overfitting_results = {}

        for metric in metrics_to_check:
            train_val = train_m[metric]
            val_val = val_m[metric]

            # 计算相对差异
            if train_val != 0:
                rel_diff = abs(train_val - val_val) / abs(train_val) * 100
            else:
                rel_diff = 0 if val_val == 0 else 100

            is_acceptable = rel_diff <= diff_threshold * 100

            overfitting_results[metric] = {
                'train_value': train_val,
                'val_value': val_val,
                'diff_percentage': rel_diff,
                'is_acceptable': is_acceptable
            }

        # 整体判断
        all_acceptable = all(r['is_acceptable'] for r in overfitting_results.values())
        overfitting_results['overall'] = {
            'is_overfitting': not all_acceptable,
            'overfitting_threshold': diff_threshold * 100
        }

        # 打印
        for metric, result in overfitting_results.items():
            if metric == 'overall':
                continue
            status = '✓' if result['is_acceptable'] else '✗'
            print(f"  {status} {metric}: 训练={result['train_value']:.2f}, "
                  f"验证={result['val_value']:.2f}, 差异={result['diff_percentage']:.1f}%")

        if overfitting_results['overall']['is_overfitting']:
            print(f"\n  ⚠️ 警告: 检测到过拟合！")
        else:
            print(f"\n  ✓ 无过拟合现象")

        return overfitting_results

    def _test_parameter_stability(self) -> Dict:
        """参数稳定性测试"""
        print("\n  对最优参数进行±10%扰动测试...")

        stability_threshold = OPTIMIZATION_CONFIG['validation']['stability_threshold']
        stability_results = {}

        # 基准性能（验证集）
        base_strategy = HybridMSBOBStrategy(**self.best_params)
        df_base = base_strategy.run_strategy(self.validate_data.copy())
        base_metrics = calculate_metrics(df_base, freq=self.freq)
        base_score = (base_metrics['profit_loss_ratio'] * 0.4 +
                      base_metrics['sharpe_ratio'] * 0.3 +
                      base_metrics['cumulative_return'] * 100 * 0.2)

        # 对每个参数进行扰动测试
        for param_name, param_value in self.best_params.items():
            test_results = []

            # ±10%扰动（浮点数）或±1（整数）
            if isinstance(param_value, int):
                perturbations = [-1, 0, 1]
            else:
                perturbations = [-0.1, 0, 0.1]

            for perturbation in perturbations:
                if perturbation == 0:
                    continue

                # 创建扰动参数
                test_params = self.best_params.copy()

                if isinstance(param_value, int):
                    new_value = param_value + perturbation
                else:
                    new_value = param_value * (1 + perturbation)

                test_params[param_name] = new_value

                # 运行测试
                try:
                    test_strategy = HybridMSBOBStrategy(**test_params)
                    df_test = test_strategy.run_strategy(self.validate_data.copy())
                    test_metrics = calculate_metrics(df_test, freq=self.freq)
                    test_score = (test_metrics['profit_loss_ratio'] * 0.4 +
                                  test_metrics['sharpe_ratio'] * 0.3 +
                                  test_metrics['cumulative_return'] * 100 * 0.2)

                    # 计算性能下降百分比
                    score_decline = (base_score - test_score) / base_score * 100

                    test_results.append({
                        'perturbation': perturbation,
                        'new_value': new_value,
                        'score': test_score,
                        'score_decline': score_decline,
                        'metrics': test_metrics
                    })

                except Exception as e:
                    print(f"    警告: 参数 {param_name}={new_value} 测试失败: {e}")

            # 判断稳定性
            max_decline = max([r['score_decline'] for r in test_results]) if test_results else 0
            is_stable = max_decline <= stability_threshold * 100

            stability_results[param_name] = {
                'base_value': param_value,
                'max_decline': max_decline,
                'is_stable': is_stable,
                'tests': test_results
            }

            status = '✓' if is_stable else '✗'
            print(f"  {status} {param_name}={param_value}: 最大性能下降={max_decline:.1f}%")

        # 整体判断
        all_stable = all(r['is_stable'] for r in stability_results.values())
        stability_results['overall'] = {
            'is_stable': all_stable,
            'stability_threshold': stability_threshold * 100
        }

        if all_stable:
            print(f"\n  ✓ 参数稳定性良好")
        else:
            print(f"\n  ⚠️ 警告: 部分参数不稳定")

        return stability_results

    def _walk_forward_analysis(self) -> Dict:
        """Walk-Forward分析"""
        print("\n  进行Walk-Forward分析...")

        # 定义时间窗口
        windows = [
            {'train': [2023], 'test': 2024},
            {'train': [2024], 'test': 2025},
        ]

        wf_results = []

        for window in windows:
            train_years = window['train']
            test_year = window['test']

            # 加载训练数据
            df_train_list = []
            for year in train_years:
                try:
                    df = load_data(year, self.freq)
                    df_train_list.append(df)
                except FileNotFoundError:
                    continue

            if not df_train_list:
                continue

            df_train = pd.concat(df_train_list, ignore_index=True).sort_values('datetime').reset_index(drop=True)

            # 加载测试数据
            try:
                df_test = load_data(test_year, self.freq)
            except FileNotFoundError:
                continue

            # 运行策略
            try:
                strategy = HybridMSBOBStrategy(**self.best_params)
                df_test_result = strategy.run_strategy(df_test)
                test_metrics = calculate_metrics(df_test_result, freq=self.freq)

                wf_results.append({
                    'train_years': train_years,
                    'test_year': test_year,
                    'metrics': test_metrics
                })

                print(f"    {test_year}年: 盈亏比={test_metrics['profit_loss_ratio']:.2f}, "
                      f"收益={test_metrics['cumulative_return']:.2%}")

            except Exception as e:
                print(f"    警告: {test_year}年测试失败: {e}")

        # 计算一致性
        if len(wf_results) >= 2:
            pl_ratios = [r['metrics']['profit_loss_ratio'] for r in wf_results]
            pl_std = np.std(pl_ratios)
            pl_mean = np.mean(pl_ratios)
            pl_cv = pl_std / pl_mean * 100 if pl_mean != 0 else 0

            walk_forward_summary = {
                'num_windows': len(wf_results),
                'results': wf_results,
                'consistency': {
                    'profit_loss_ratio_mean': pl_mean,
                    'profit_loss_ratio_std': pl_std,
                    'coefficient_of_variation': pl_cv
                }
            }

            print(f"    盈亏比一致性: 均值={pl_mean:.2f}, 标准差={pl_std:.2f}, "
                  f"变异系数={pl_cv:.1f}%")

        else:
            walk_forward_summary = {
                'num_windows': len(wf_results),
                'results': wf_results,
                'consistency': None
            }

        return walk_forward_summary

    def _monte_carlo_simulation(self, n_runs: int = 100) -> Dict:
        """蒙特卡洛模拟"""
        print(f"\n  进行蒙特卡洛模拟 ({n_runs}次)...")

        # 运行策略获取交易序列
        strategy = HybridMSBOBStrategy(**self.best_params)
        df_result = strategy.run_strategy(self.validate_data.copy())

        # 识别交易
        trades = []
        in_position = False
        entry_idx = None
        entry_price = None
        pos_type = None

        for i in range(len(df_result)):
            if not in_position and df_result['position'].iloc[i] != 0:
                in_position = True
                entry_idx = i
                entry_price = df_result['close'].iloc[i]
                pos_type = 'long' if df_result['position'].iloc[i] > 0 else 'short'
            elif in_position and df_result['position'].iloc[i] == 0:
                exit_price = df_result['close'].iloc[i]
                if pos_type == 'long':
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price
                trades.append(pnl)
                in_position = False

        if not trades:
            return {'error': '无交易数据'}

        # 蒙特卡洛模拟
        simulated_returns = []

        for _ in range(n_runs):
            # 随机打乱交易顺序
            shuffled_trades = random.sample(trades, len(trades))

            # 计算累计收益
            cumulative_return = (1 + pd.Series(shuffled_trades)).prod() - 1
            simulated_returns.append(cumulative_return)

        # 统计分析
        original_return = (1 + pd.Series(trades)).prod() - 1
        simulated_mean = np.mean(simulated_returns)
        simulated_std = np.std(simulated_returns)

        # 计算p值
        p_value = sum(1 for r in simulated_returns if r >= original_return) / n_runs

        monte_carlo_results = {
            'num_trades': len(trades),
            'num_simulations': n_runs,
            'original_cumulative_return': original_return,
            'simulated_mean': simulated_mean,
            'simulated_std': simulated_std,
            'simulated_min': np.min(simulated_returns),
            'simulated_max': np.max(simulated_returns),
            'percentile_5': np.percentile(simulated_returns, 5),
            'percentile_95': np.percentile(simulated_returns, 95),
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }

        print(f"    原始收益: {original_return:.2%}")
        print(f"    模拟均值: {simulated_mean:.2%} ± {simulated_std:.2%}")
        print(f"    p值: {p_value:.3f} ({'显著' if p_value < 0.05 else '不显著'})")

        return monte_carlo_results

    def _generate_validation_report(self):
        """生成验证报告"""
        report_path = self.results_dir / "validation_report.json"

        # 转换numpy类型
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        report = convert_types(self.validation_results)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n验证报告已保存: {report_path}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    # 示例：从优化报告加载最优参数
    results_dir = current_dir / "results"
    json_files = list(results_dir.glob("*/optimization_report.json"))

    if not json_files:
        print("错误: 未找到优化报告")
        print("请先运行 optuna_optimizer.py")
        return

    # 加载最新的优化报告
    latest_report = max(json_files, key=lambda p: p.stat().st_mtime)
    with open(latest_report, 'r', encoding='utf-8') as f:
        report = json.load(f)

    best_params = report['best_params']

    print(f"加载最优参数: {latest_report}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # 创建验证器
    validator = ParameterValidator(
        best_params=best_params,
        train_years=[2023, 2024],
        validate_year=2025,
        freq='60min',
        results_dir=latest_report.parent
    )

    # 运行验证
    results = validator.run_all_validations()

    return results


if __name__ == '__main__':
    main()
