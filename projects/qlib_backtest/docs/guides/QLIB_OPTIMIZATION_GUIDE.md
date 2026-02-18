# Qlib 策略参数优化指南

## 概述

Qlib 提供了多种参数优化方法，主要分为两大类：

1. **Tuner（超参数调优器）** - 使用 hyperopt 自动优化策略参数
2. **Optimizer（投资组合优化器）** - 优化投资组合权重分配

## 1. Tuner - 超参数调优器

### 1.1 功能介绍

`qlib.contrib.tuner` 模块提供了一个基于 **hyperopt** 的超参数调优框架，可以自动搜索最优的策略参数。

**核心特性：**
- 使用 TPE（Tree-structured Parzen Estimator）算法
- 支持优化模型参数、策略参数、数据标签参数
- 通过 subprocess 并行运行回测
- 支持多种优化目标（夏普比率、收益、信息比率等）

### 1.2 架构设计

```
Tuner (基类)
  └── QLibTuner (实现类)
       ├── objective() - 优化目标函数
       ├── setup_space() - 设置搜索空间
       └── tune() - 执行优化
```

### 1.3 核心代码位置

```python
# 主要文件
/home/mystryl/.local/lib/python3.12/site-packages/qlib/contrib/tuner/
  ├── tuner.py          # Tuner 基类和 QLibTuner 实现
  ├── space.py          # 预定义的搜索空间
  ├── config.py         # 配置文件
  └── launcher.py       # 启动器
```

### 1.4 使用方法

#### 1.4.1 定义搜索空间

在 `qlib/contrib/tuner/space.py` 中定义参数搜索空间：

```python
from hyperopt import hp

# TopkAmountStrategy 搜索空间
TopkAmountStrategySpace = {
    "topk": hp.choice("topk", [30, 35, 40]),
    "buffer_margin": hp.choice("buffer_margin", [200, 250, 300]),
}

# 数据标签搜索空间
QLibDataLabelSpace = {
    "labels": hp.choice(
        "labels",
        [["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["Ref($close, -5)/$close - 1"]],
    )
}
```

#### 1.4.2 配置 Tuner

```python
tuner_config = {
    "experiment": {
        "dir": "../experiments",
        "name": "my_experiment",
        "id": "tuner_001"
    },
    "model": {
        "class": "LSTM",
        "space": "MyModelSpace",  # 自定义搜索空间类名
        "args": {...}
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "space": "MyStrategySpace",  # 自定义搜索空间类名
        "args": {...}
    },
    "max_evals": 50  # 最多尝试 50 次参数组合
}

optim_config = {
    "report_type": "excess_return_with_cost",  # 优化目标类型
    "report_factor": "annualized_return",     # 优化目标因子
    "optim_type": "max"  # 最大化
}
```

#### 1.4.3 运行优化

```python
from qlib.contrib.tuner import QLibTuner

# 创建 Tuner
tuner = QLibTuner(tuner_config, optim_config)

# 执行优化（自动搜索最优参数）
tuner.tune()

# 获取最优参数
print(f"Best params: {tuner.best_params}")
print(f"Best result: {tuner.best_res}")
```

### 1.5 自定义 Tuner 示例

如果需要为自定义策略创建 Tuner，可以继承 `Tuner` 基类：

```python
from qlib.contrib.tuner import Tuner
from hyperopt import hp, STATUS_OK, STATUS_FAIL

class MyStrategyTuner(Tuner):
    def setup_space(self):
        """定义搜索空间"""
        space = {
            'period': hp.choice('period', [7, 10, 14, 20]),
            'multiplier': hp.uniform('multiplier', 1.5, 4.0),
            'n': hp.choice('n', [1, 2, 3]),
            'trailing_stop_rate': hp.uniform('trailing_stop_rate', 60, 100)
        }
        return space

    def objective(self, params):
        """优化目标函数"""
        try:
            # 使用当前参数运行回测
            strategy = MyStrategy(**params)
            results = run_backtest(strategy, data)

            # 返回优化目标（最小化负夏普比率 = 最大化夏普比率）
            return {
                'loss': -results['sharpe_ratio'],
                'status': STATUS_OK
            }
        except Exception as e:
            return {
                'loss': float('inf'),
                'status': STATUS_FAIL
            }

    def save_local_best_params(self):
        """保存最优参数"""
        import json
        with open('best_params.json', 'w') as f:
            json.dump(self.best_params, f)

# 使用
tuner = MyStrategyTuner(tuner_config, optim_config)
tuner.tune()
```

## 2. Optimizer - 投资组合优化器

### 2.1 功能介绍

`qlib.contrib.strategy.optimizer` 模块提供了投资组合权重优化功能，不是用于优化策略参数。

**支持的优化方法：**
- **GMV** (Global Minimum Variance) - 全局最小方差
- **MVO** (Mean Variance Optimization) - 均值方差优化
- **RP** (Risk Parity) - 风险平价
- **INV** (Inverse Volatility) - 反向波动率
- **Enhanced Indexing** - 增强指数化

### 2.2 使用示例

```python
from qlib.contrib.strategy.optimizer import PortfolioOptimizer
import numpy as np

# 创建优化器
optimizer = PortfolioOptimizer(
    method="mvo",      # 使用均值方差优化
    lamb=0.5,         # 风险厌恶参数
    delta=0.2,        # 换手率限制
    alpha=0.0         # L2 正则化
)

# 协方差矩阵
S = np.array([[0.01, 0.005], [0.005, 0.02]])

# 预期收益
r = np.array([0.1, 0.15])

# 初始权重
w0 = np.array([0.5, 0.5])

# 优化
w = optimizer(S, r, w0)
print(f"Optimized weights: {w}")
```

## 3. 强化学习模块

### 3.1 功能介绍

`qlib.rl` 模块提供了基于强化学习的策略优化功能，可以用于：

- 订单执行优化（`order_execution`）
- 策略学习（`strategy`）

### 3.2 目录结构

```
qlib/rl/
  ├── order_execution/    # 订单执行优化
  ├── strategy/          # 策略学习
  ├── trainer.py         # 训练器
  └── reward.py          # 奖励函数
```

## 4. 实际应用建议

### 4.1 对于技术指标策略（如 SuperTrend）

**推荐方法：** 使用自定义 Tuner + 网格搜索

理由：
- Qlib 的 Tuner 主要针对基于预测信号的 TopkDropout 策略
- 技术指标策略更适合直接网格搜索或使用第三方优化库（如 Optuna）

**示例代码：**

```python
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from qlib_supertrend_enhanced import SupertrendEnhancedStrategy, run_backtest

def grid_search_optimization(data, param_grid, objective='sharpe_ratio'):
    """网格搜索优化"""
    all_results = []

    # 生成所有参数组合
    param_combinations = list(itertools.product(
        param_grid['period'],
        param_grid['multiplier'],
        param_grid['n'],
        param_grid['trailing_stop_rate']
    ))

    for period, multiplier, n, ts_rate in param_combinations:
        # 创建策略
        strategy = SupertrendEnhancedStrategy(
            period=period,
            multiplier=multiplier,
            n=n,
            trailing_stop_rate=ts_rate
        )

        # 运行回测
        df_strategy = strategy.generate_signal(data.copy())
        results = run_backtest(df_strategy, strategy.name)

        # 保存结果
        results.update({
            'period': period,
            'multiplier': multiplier,
            'n': n,
            'trailing_stop_rate': ts_rate
        })
        all_results.append(results)

    # 按目标排序
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(objective, ascending=False)

    return results_df

# 参数网格
param_grid = {
    'period': [7, 10, 14, 20],
    'multiplier': [2.0, 2.5, 3.0, 3.5],
    'n': [1, 2, 3],
    'trailing_stop_rate': [60, 70, 80, 90]
}

# 运行优化
results = grid_search_optimization(data, param_grid)
print(results.head(10))
```

### 4.2 使用 Optuna 进行贝叶斯优化

```python
import optuna
from qlib_supertrend_enhanced import SupertrendEnhancedStrategy, run_backtest

def objective(trial):
    # 定义参数搜索空间
    period = trial.suggest_int('period', 5, 30)
    multiplier = trial.suggest_float('multiplier', 1.0, 5.0)
    n = trial.suggest_int('n', 1, 5)
    trailing_stop_rate = trial.suggest_int('trailing_stop_rate', 50, 100)

    # 创建策略并运行回测
    strategy = SupertrendEnhancedStrategy(
        period=period,
        multiplier=multiplier,
        n=n,
        trailing_stop_rate=trailing_stop_rate
    )

    df_strategy = strategy.generate_signal(data.copy())
    results = run_backtest(df_strategy, strategy.name)

    # 返回优化目标（Optuna 默认最小化）
    return -results['sharpe_ratio']

# 创建优化研究
study = optuna.create_study(direction='minimize')

# 运行优化（100 次试验）
study.optimize(objective, n_trials=100)

# 获取最优参数
print(f"Best params: {study.best_params}")
print(f"Best sharpe ratio: {-study.best_value}")
```

### 4.3 对于基于预测信号的投资组合策略

**推荐方法：** 使用 Qlib Tuner

理由：
- Qlib Tuner 专为预测信号策略设计
- 支持 TopkDropout、Enhanced Indexing 等策略
- 自动集成工作流和回测

## 5. 总结

| 优化需求 | 推荐方法 | 备注 |
|---------|---------|------|
| 技术指标策略参数优化 | 网格搜索 / Optuna | 更直接、灵活 |
| 基于预测信号的策略参数优化 | Qlib Tuner | 原生支持 |
| 投资组合权重优化 | Qlib Optimizer | 专门用于权重分配 |
| 订单执行优化 | Qlib RL | 强化学习方法 |

## 6. 相关文件

- Qlib 源代码位置：`/home/mystryl/.local/lib/python3.12/site-packages/qlib/`
- Tuner 模块：`qlib/contrib/tuner/tuner.py`
- Optimizer 模块：`qlib/contrib/strategy/optimizer/optimizer.py`
- RL 模块：`qlib/rl/`

## 7. 参考资料

- Qlib 官方文档：https://qlib.readthedocs.io/
- Hyperopt 文档：http://hyperopt.github.io/hyperopt/
- Optuna 文档：https://optuna.readthedocs.io/
