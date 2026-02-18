#!/usr/bin/env python3
"""
V03混合策略参数优化配置

参数空间设计原则：
1. 针对盈亏比优化：放宽止损、放大止盈、提高MSB阈值
2. 合理范围：基于策略逻辑和经验设定
3. 优先级：MSB阈值、ATR倍数、止盈倍数为关键参数
"""

# ============================================================================
# 参数空间定义（针对盈亏比优化）
# ============================================================================

PARAM_SPACE = {
    # 枢轴点窗口（影响信号质量）
    'pivot_len': (5, 10),

    # MSB动量阈值（关键：影响信号频率和质量）
    # 原值0.3，提高可减少假信号
    'msb_zscore': (0.2, 0.8),

    # ATR周期
    'atr_period': (10, 20),

    # ATR止损倍数（关键：影响盈亏比）
    # 原值1.0，放大可提高盈亏比
    'atr_multiplier': (0.8, 2.0),

    # 止盈倍数（关键：影响平均盈利）
    # 原值0.5，放大可提高盈利幅度
    'tp1_multiplier': (0.3, 1.0),

    # 原值1.0
    'tp2_multiplier': (0.8, 1.5),

    # 原值1.5，可以更大
    'tp3_multiplier': (1.2, 2.5),
}

# ============================================================================
# 优化配置
# ============================================================================

OPTIMIZATION_CONFIG = {
    # 优化目标权重
    'objective_weights': {
        'profit_loss_ratio': 0.40,    # 盈亏比权重40%（最重要）
        'sharpe_ratio': 0.30,          # 夏普比率权重30%
        'cumulative_return': 0.20,     # 累计收益权重20%
        'max_drawdown_penalty': 0.10,  # 回撤惩罚权重10%
    },

    # 约束条件
    'constraints': {
        'min_win_rate': 70.0,          # 最低胜率70%
        'max_max_drawdown': 0.05,      # 最大回撤5%
        'min_total_trades': 50,        # 最少交易次数50
        'max_single_loss': 0.03,       # 单笔最大亏损3%
    },

    # 数据集划分
    'data_split': {
        'train_years': [2023, 2024],   # 训练集
        'validate_year': 2025,         # 验证集
    },

    # Optuna配置
    'optuna': {
        'n_trials': 100,               # 迭代次数
        'study_name': 'V03_Hybrid_60min_Optimization',
        'direction': 'maximize',       # 最大化目标
    },

    # 网格搜索配置
    'grid_search': {
        'enabled': True,               # 是否启用精细网格搜索
        'range_ratio': 0.2,            # 最优参数±20%范围
        'step_size': {
            'pivot_len': 1,            # 整数步长
            'msb_zscore': 0.05,        # 浮点数步长
            'atr_period': 1,
            'atr_multiplier': 0.1,
            'tp1_multiplier': 0.05,
            'tp2_multiplier': 0.05,
            'tp3_multiplier': 0.1,
        }
    },

    # 验证配置
    'validation': {
        'overfitting_threshold': 0.20,  # 训练集vs验证集性能差异阈值20%
        'stability_threshold': 0.10,    # 参数稳定性测试阈值10%
        'monte_carlo_runs': 100,        # 蒙特卡洛模拟次数
    },

    # 成功标准
    'success_criteria': {
        'target_profit_loss_ratio': 1.2,
        'target_sharpe_ratio': 2.0,
        'target_cumulative_return': 0.25,
        'acceptable_win_rate': 70.0,
        'acceptable_max_drawdown': 0.05,
    },

    # 优秀标准
    'excellent_criteria': {
        'target_profit_loss_ratio': 1.2,
        'target_sharpe_ratio': 2.0,
        'target_cumulative_return': 0.25,
        'overfitting_threshold': 0.15,  # 更严格的过拟合检测
    },

    # 失败标准
    'failure_criteria': {
        'min_profit_loss_ratio': 0.6,
        'max_overfitting_diff': 0.30,
        'max_stability_decline': 0.20,
    }
}


# ============================================================================
# 辅助函数
# ============================================================================

def get_param_space_for_optimization(optimizer_type='optuna'):
    """
    根据优化器类型返回合适的参数空间

    Args:
        optimizer_type: 'optuna' 或 'grid'

    Returns:
        参数空间字典
    """
    if optimizer_type == 'optuna':
        return PARAM_SPACE.copy()
    elif optimizer_type == 'grid':
        # 网格搜索使用更精细的离散空间
        return {
            'pivot_len': [5, 6, 7, 8, 9, 10],
            'msb_zscore': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'atr_period': [10, 12, 14, 16, 18, 20],
            'atr_multiplier': [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            'tp1_multiplier': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'tp2_multiplier': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            'tp3_multiplier': [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.5],
        }
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")


def calculate_objective_score(metrics, config=OPTIMIZATION_CONFIG):
    """
    计算组合优化得分

    Args:
        metrics: 策略指标字典
        config: 优化配置

    Returns:
        组合得分
    """
    weights = config['objective_weights']
    constraints = config['constraints']

    # 基础得分
    score = (
        weights['profit_loss_ratio'] * metrics.get('profit_loss_ratio', 0) +
        weights['sharpe_ratio'] * metrics.get('sharpe_ratio', 0) +
        weights['cumulative_return'] * (metrics.get('cumulative_return', 0) * 100)
    )

    # 约束条件惩罚
    penalty = 1.0

    # 胜率惩罚
    if metrics.get('win_rate', 0) < constraints['min_win_rate']:
        penalty *= 0.5

    # 回撤惩罚
    if metrics.get('max_drawdown', 0) > constraints['max_max_drawdown']:
        penalty *= 0.7

    # 交易次数惩罚
    if metrics.get('total_trades', 0) < constraints['min_total_trades']:
        penalty *= 0.8

    return score * penalty


def check_success_criteria(metrics, criteria_type='success'):
    """
    检查是否达到成功标准

    Args:
        metrics: 策略指标字典
        criteria_type: 'success', 'excellent', 或 'failure'

    Returns:
        (是否达标, 未达标项列表)
    """
    if criteria_type == 'success':
        criteria = OPTIMIZATION_CONFIG['success_criteria']
    elif criteria_type == 'excellent':
        criteria = OPTIMIZATION_CONFIG['excellent_criteria']
    elif criteria_type == 'failure':
        criteria = OPTIMIZATION_CONFIG['failure_criteria']
    else:
        raise ValueError(f"未知的标准类型: {criteria_type}")

    passed = True
    failed_items = []

    for key, target_value in criteria.items():
        if key == 'overfitting_threshold' or key == 'max_stability_decline':
            # 这些是验证指标，不在基础指标中
            continue

        current_value = metrics.get(key, 0)

        # 判断逻辑
        if 'ratio' in key or 'return' in key or 'rate' in key:
            # 越高越好
            if current_value < target_value:
                passed = False
                failed_items.append((key, current_value, target_value, '>='))
        else:
            # 越低越好
            if current_value > target_value:
                passed = False
                failed_items.append((key, current_value, target_value, '<='))

    return passed, failed_items


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print("V03混合策略参数优化配置")
    print("="*80)
    print("\n参数空间:")
    for key, value in PARAM_SPACE.items():
        print(f"  {key}: {value}")

    print("\n优化目标权重:")
    for key, value in OPTIMIZATION_CONFIG['objective_weights'].items():
        print(f"  {key}: {value:.2f}")

    print("\n约束条件:")
    for key, value in OPTIMIZATION_CONFIG['constraints'].items():
        print(f"  {key}: {value}")

    print("\n成功标准:")
    for key, value in OPTIMIZATION_CONFIG['success_criteria'].items():
        print(f"  {key}: {value}")
