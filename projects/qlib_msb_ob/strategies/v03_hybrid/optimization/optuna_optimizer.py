#!/usr/bin/env python3
"""
V03混合策略 - Optuna贝叶斯优化器

使用Optuna进行智能参数优化，特点：
- TPE采样算法：基于树结构的Parzen估计器
- 剪枝（Pruning）：提前终止无效试验
- 并行计算：支持多进程加速
- 可视化：optuna-dashboard实时监控
"""

import sys
import os
import time
import json
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

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
    PARAM_SPACE,
    OPTIMIZATION_CONFIG,
    calculate_objective_score,
    check_success_criteria,
    get_param_space_for_optimization
)


# ============================================================================
# Optuna优化器
# ============================================================================

class V03OptunaOptimizer:
    """V03策略Optuna优化器"""

    def __init__(
        self,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_trials: int = 100,
        train_years: List[int] = None,
        validate_year: int = 2025,
        freq: str = '60min'
    ):
        """
        初始化优化器

        Args:
            storage: Optuna存储URL（如'sqlite:///db.sqlite3'）
            study_name: Study名称
            n_trials: 迭代次数
            train_years: 训练集年份列表
            validate_year: 验证集年份
            freq: 数据频率
        """
        self.storage = storage or f"sqlite:///{current_dir}/results/optuna_study.db"
        self.study_name = study_name or OPTIMIZATION_CONFIG['optuna']['study_name']
        self.n_trials = n_trials
        self.train_years = train_years or OPTIMIZATION_CONFIG['data_split']['train_years']
        self.validate_year = validate_year
        self.freq = freq

        # 创建结果目录
        self.results_dir = current_dir / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 加载训练数据
        print(f"加载训练数据: {train_years}年")
        self.train_data = self._load_training_data()

        # 加载验证数据
        print(f"加载验证数据: {validate_year}年")
        self.validate_data = load_data(validate_year, freq)

        # 优化历史
        self.optimization_history = []

        print(f"\n优化器初始化完成:")
        print(f"  训练集: {train_years} ({len(self.train_data)} 条记录)")
        print(f"  验证集: {validate_year} ({len(self.validate_data)} 条记录)")
        print(f"  迭代次数: {n_trials}")
        print(f"  结果目录: {self.results_dir}")

    def _load_training_data(self) -> pd.DataFrame:
        """加载并合并训练数据"""
        df_list = []
        for year in self.train_years:
            try:
                df = load_data(year, self.freq)
                df_list.append(df)
                print(f"  {year}年: {len(df)} 条记录")
            except FileNotFoundError as e:
                print(f"  警告: {e}")
        return pd.concat(df_list, ignore_index=True).sort_values('datetime').reset_index(drop=True)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna目标函数

        Args:
            trial: Optuna试验对象

        Returns:
            优化得分（越高越好）
        """
        # 1. 采样参数
        params = {
            'pivot_len': trial.suggest_int('pivot_len', 5, 10),
            'msb_zscore': trial.suggest_float('msb_zscore', 0.2, 0.8),
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            'atr_multiplier': trial.suggest_float('atr_multiplier', 0.8, 2.0),
            'tp1_multiplier': trial.suggest_float('tp1_multiplier', 0.3, 1.0),
            'tp2_multiplier': trial.suggest_float('tp2_multiplier', 0.8, 1.5),
            'tp3_multiplier': trial.suggest_float('tp3_multiplier', 1.2, 2.5),
        }

        # 2. 运行策略
        try:
            strategy = HybridMSBOBStrategy(**params)
            df_result = strategy.run_strategy(self.train_data.copy())
            metrics = calculate_metrics(df_result, freq=self.freq)

            # 3. 检查基本约束
            constraints = OPTIMIZATION_CONFIG['constraints']

            # 交易次数太少，返回低分
            if metrics['total_trades'] < constraints['min_total_trades']:
                return 0.0

            # 4. 计算组合得分
            score = calculate_objective_score(metrics, OPTIMIZATION_CONFIG)

            # 5. 记录中间结果
            trial.set_user_attr('win_rate', metrics['win_rate'])
            trial.set_user_attr('profit_loss_ratio', metrics['profit_loss_ratio'])
            trial.set_user_attr('cumulative_return', metrics['cumulative_return'])
            trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
            trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
            trial.set_user_attr('total_trades', metrics['total_trades'])

            # 6. 保存到历史
            self.optimization_history.append({
                'params': params.copy(),
                'metrics': metrics.copy(),
                'score': score,
                'trial_number': trial.number
            })

            return score

        except Exception as e:
            print(f"试验 #{trial.number} 出错: {e}")
            return 0.0

    def optimize(self, timeout: Optional[int] = None) -> Tuple[optuna.Study, Dict]:
        """
        运行优化

        Args:
            timeout: 超时时间（秒），None表示不限制

        Returns:
            (study, validation_metrics)
        """
        print("\n" + "="*80)
        print("开始Optuna贝叶斯优化")
        print("="*80)

        # 创建或加载study
        study = optuna.create_study(
            storage=self.storage,
            study_name=self.study_name,
            direction='maximize',
            load_if_exists=True
        )

        # 定义回调函数用于实时输出
        def callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                value = trial.value
                params = trial.params
                print(f"试验 #{trial.number:3d}: 得分={value:.4f}  "
                      f"msb_zscore={params['msb_zscore']:.2f}  "
                      f"atr_mult={params['atr_multiplier']:.2f}  "
                      f"tp1={params['tp1_multiplier']:.2f}")

        # 运行优化
        start_time = time.time()
        study.optimize(self.objective, n_trials=self.n_trials, callbacks=[callback])
        elapsed_time = time.time() - start_time

        # 输出结果
        print("\n" + "="*80)
        print("优化完成！")
        print("="*80)
        print(f"总用时: {elapsed_time:.1f}秒")
        print(f"总试验次数: {len(study.trials)}")
        print(f"最优得分: {study.best_value:.4f}")
        print(f"\n最优参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # 保存优化历史
        self._save_optimization_history(study)

        # 在验证集上测试
        print("\n" + "="*80)
        print(f"验证集测试 ({self.validate_year}年)")
        print("="*80)
        val_metrics = self._validate_on_test_set(study.best_params)

        # 生成优化报告
        self._generate_optimization_report(study, val_metrics)

        return study, val_metrics

    def _validate_on_test_set(self, best_params: Dict) -> Dict:
        """在验证集上测试最优参数"""
        strategy = HybridMSBOBStrategy(**best_params)
        df_val_result = strategy.run_strategy(self.validate_data.copy())
        val_metrics = calculate_metrics(df_val_result, freq=self.freq)

        print(f"\n验证集结果:")
        print(f"  盈亏比: {val_metrics['profit_loss_ratio']:.2f}")
        print(f"  夏普比率: {val_metrics['sharpe_ratio']:.2f}")
        print(f"  累计收益: {val_metrics['cumulative_return']:.2%}")
        print(f"  胜率: {val_metrics['win_rate']:.2f}%")
        print(f"  最大回撤: {val_metrics['max_drawdown']:.2%}")
        print(f"  交易次数: {val_metrics['total_trades']}")

        return val_metrics

    def _save_optimization_history(self, study: optuna.Study):
        """保存优化历史到CSV"""
        df_history = pd.DataFrame(self.optimization_history)

        # 展开params和metrics
        df_params = pd.json_normalize(df_history['params'])
        df_metrics = pd.json_normalize(df_history['metrics'])

        df_combined = pd.concat([
            df_history[['trial_number', 'score']],
            df_params,
            df_metrics
        ], axis=1)

        csv_path = self.results_dir / "optimization_history.csv"
        df_combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n优化历史已保存: {csv_path}")

    def _generate_optimization_report(self, study: optuna.Study, val_metrics: Dict):
        """生成优化报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'study_name': self.study_name,
            'n_trials': len(study.trials),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'train_metrics': {
                'win_rate': study.best_trial.user_attrs.get('win_rate', 0),
                'profit_loss_ratio': study.best_trial.user_attrs.get('profit_loss_ratio', 0),
                'cumulative_return': study.best_trial.user_attrs.get('cumulative_return', 0),
                'max_drawdown': study.best_trial.user_attrs.get('max_drawdown', 0),
                'sharpe_ratio': study.best_trial.user_attrs.get('sharpe_ratio', 0),
                'total_trades': study.best_trial.user_attrs.get('total_trades', 0),
            },
            'validation_metrics': val_metrics,
            'optimization_config': {
                'train_years': self.train_years,
                'validate_year': self.validate_year,
                'freq': self.freq,
                'param_space': PARAM_SPACE,
            }
        }

        # 检查成功标准
        success_passed, success_failed = check_success_criteria(val_metrics, 'success')
        excellent_passed, excellent_failed = check_success_criteria(val_metrics, 'excellent')

        report['success_check'] = {
            'success': success_passed,
            'excellent': excellent_passed,
            'failed_items': success_failed if not success_passed else []
        }

        # 过拟合检测
        train_pl_ratio = study.best_trial.user_attrs.get('profit_loss_ratio', 0)
        val_pl_ratio = val_metrics.get('profit_loss_ratio', 0)
        overfitting_diff = abs(train_pl_ratio - val_pl_ratio) / max(train_pl_ratio, 0.001) * 100

        report['overfitting_check'] = {
            'train_profit_loss_ratio': float(train_pl_ratio),
            'val_profit_loss_ratio': float(val_pl_ratio),
            'diff_percentage': float(overfitting_diff),
            'is_overfitting': bool(overfitting_diff > OPTIMIZATION_CONFIG['validation']['overfitting_threshold'] * 100)
        }

        # 保存JSON报告
        json_path = self.results_dir / "optimization_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n优化报告已保存: {json_path}")

        # 打印总结
        print("\n" + "="*80)
        print("优化总结")
        print("="*80)

        print(f"\n训练集最优结果:")
        train_m = report['train_metrics']
        print(f"  盈亏比: {train_m['profit_loss_ratio']:.2f}")
        print(f"  夏普比率: {train_m['sharpe_ratio']:.2f}")
        print(f"  累计收益: {train_m['cumulative_return']:.2%}")
        print(f"  胜率: {train_m['win_rate']:.2f}%")

        print(f"\n验证集结果:")
        val_m = report['validation_metrics']
        print(f"  盈亏比: {val_m['profit_loss_ratio']:.2f}")
        print(f"  夏普比率: {val_m['sharpe_ratio']:.2f}")
        print(f"  累计收益: {val_m['cumulative_return']:.2%}")
        print(f"  胜率: {val_m['win_rate']:.2f}%")

        print(f"\n过拟合检测:")
        of = report['overfitting_check']
        print(f"  训练集盈亏比: {of['train_profit_loss_ratio']:.2f}")
        print(f"  验证集盈亏比: {of['val_profit_loss_ratio']:.2f}")
        print(f"  差异: {of['diff_percentage']:.1f}%")
        print(f"  状态: {'[!] 过拟合' if of['is_overfitting'] else '[OK] 正常'}")

        print(f"\n达标检查:")
        sc = report['success_check']
        if sc['success']:
            print(f"  [OK] 达到成功标准")
        elif sc['excellent']:
            print(f"  [STAR] 达到优秀标准")
        else:
            print(f"  [X] 未达到成功标准")
            for item in sc['failed_items']:
                print(f"    - {item[0]}: {item[1]:.2f} (目标: {item[2]:.2f} {item[3]})")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    print("="*80)
    print("V03混合策略 - Optuna贝叶斯优化")
    print("="*80)

    # 创建优化器
    # 运行15分钟周期优化
    optimizer = V03OptunaOptimizer(
        n_trials=100,
        train_years=[2023, 2024],
        validate_year=2025,
        freq='15min',  # 使用15分钟周期
        study_name='V03_Hybrid_15min_Optimization'
    )

    # 运行优化
    study, val_metrics = optimizer.optimize()

    # 返回最优参数
    return study.best_params, val_metrics


if __name__ == '__main__':
    # 安装提示
    try:
        import optuna
    except ImportError:
        print("错误: 未安装Optuna")
        print("请运行: pip install optuna optuna-dashboard")
        sys.exit(1)

    best_params, val_metrics = main()
