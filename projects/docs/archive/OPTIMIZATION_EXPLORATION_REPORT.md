# Qlib 参数优化探索报告

**日期：** 2026-02-15
**目标：** 探索 qlib 是否有优化策略参数的方法或函数

---

## 探索过程

### 1. 初始搜索

首先搜索了 qlib 官方文档中关于参数优化的内容，发现了：

- Qlib 的策略文档（`https://qlib.readthedocs.io/en/latest/component/strategy.html`）
- 主要介绍了投资组合策略的使用，但没有找到专门的参数优化函数

### 2. GitHub Issue 调查

发现了一个关键的 GitHub Issue #1458：
- **问题：** qlib 论文中提到了 Hyperparameters Tuning Engine (HTE)，但在文档和源代码中找不到如何使用
- **链接：** https://github.com/microsoft/qlib/issues/1458
- **结论：** HTE 功能可能尚未完全实现或文档化

### 3. 源代码探索

通过检查 qlib 安装路径，发现了以下重要模块：

#### 3.1 Tuner 模块
**位置：** `/home/mystryl/.local/lib/python3.12/site-packages/qlib/contrib/tuner/`

**核心文件：**
- `tuner.py` - Tuner 基类和 QLibTuner 实现
- `space.py` - 预定义的搜索空间
- `config.py` - 配置文件
- `launcher.py` - 启动器

**功能：**
- 使用 **hyperopt** 库进行超参数优化
- 采用 TPE（Tree-structured Parzen Estimator）算法
- 支持优化模型参数、策略参数、数据标签参数
- 通过 subprocess 并行运行回测

**主要类：**
```python
class Tuner:
    - objective()  # 优化目标函数（抽象方法）
    - setup_space()  # 设置搜索空间（抽象方法）
    - tune()  # 执行优化

class QLibTuner(Tuner):
    - 针对 qlib 工作流的实现
    - 使用 estimator 进程运行回测
    - 支持多种优化目标（夏普比率、收益等）
```

**预定义搜索空间：**
```python
TopkAmountStrategySpace = {
    "topk": hp.choice("topk", [30, 35, 40]),
    "buffer_margin": hp.choice("buffer_margin", [200, 250, 300]),
}

QLibDataLabelSpace = {
    "labels": hp.choice("labels", [...])
}
```

#### 3.2 Optimizer 模块
**位置：** `/home/mystryl/.local/lib/python3.12/site-packages/qlib/contrib/strategy/optimizer/`

**核心文件：**
- `optimizer.py` - 投资组合优化器
- `enhanced_indexing.py` - 增强指数化优化器
- `base.py` - 基类

**功能：**
- **不是**用于优化策略参数，而是用于优化投资组合权重分配
- 支持多种优化方法：
  - GMV (Global Minimum Variance) - 全局最小方差
  - MVO (Mean Variance Optimization) - 均值方差优化
  - RP (Risk Parity) - 风险平价
  - INV (Inverse Volatility) - 反向波动率
  - Enhanced Indexing - 增强指数化

**使用场景：**
- 给定协方差矩阵和预期收益，计算最优权重
- 不是用于优化技术指标参数（如 SuperTrend 的 period、multiplier）

#### 3.3 强化学习模块
**位置：** `/home/mystryl/.local/lib/python3.12/site-packages/qlib/rl/`

**功能：**
- 订单执行优化（`order_execution`）
- 策略学习（`strategy`）

**适用场景：**
- 高频交易策略
- 智能订单路由

---

## 发现总结

### Qlib 提供的参数优化方法

| 方法 | 适用场景 | 实现库 | 说明 |
|-----|---------|--------|------|
| **Tuner** | 基于预测信号的策略参数优化 | hyperopt | 使用 TPE 算法，支持 TopkDropout、Enhanced Indexing 等策略 |
| **Optimizer** | 投资组合权重优化 | scipy/cvxpy | 优化资产配置权重，不是策略参数 |
| **RL** | 订单执行和策略学习 | 自定义 | 强化学习方法 |

### 对于技术指标策略（如 SuperTrend）

**结论：** Qlib 的 Tuner 模块主要针对基于预测信号的策略（如 TopkDropout），对于技术指标策略来说：

1. **Qlib Tuner 不是最优选择**
   - 需要配置完整的 qlib 工作流
   - 通过 subprocess 运行，效率较低
   - 学习曲线较陡

2. **推荐使用第三方优化库**
   - **Optuna** - 贝叶斯优化，效率高，可视化好
   - **Hyperopt** - Qlib 使用的库，可以单独使用
   - **网格搜索** - 简单直接，适合参数较少的情况

---

## 推荐方案

### 方案 1：使用 Optuna（推荐）

**优点：**
- 贝叶斯优化，比网格搜索更高效
- 优秀的可视化工具
- 支持剪枝（提前终止无效试验）
- 灵活易用

**示例代码：**
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

    # 返回优化目标
    return -results['sharpe_ratio']

# 创建优化研究
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 获取最优参数
print(f"Best params: {study.best_params}")
```

**已创建文件：**
- `/mnt/d/quant/qlib_backtest/optimize_supertrend_optuna.py` - 完整的 Optuna 优化脚本

### 方案 2：使用网格搜索

**优点：**
- 简单直观
- 不需要额外依赖
- 适合参数较少的情况

**已实现：**
- `/mnt/d/quant/qlib_backtest/optimize_enhanced_supertrend.py` - 网格搜索优化脚本
- `/mnt/d/quant/qlib_backtest/optimize_enhanced_params.py` - 多频率参数优化

### 方案 3：使用 Qlib Tuner（不推荐）

**适用场景：**
- 基于 qlib 预测信号的策略
- 已经配置了完整的 qlib 工作流

**缺点：**
- 配置复杂
- 学习曲线陡
- 对于技术指标策略效率低

---

## 文件清单

### 创建的文档
1. **QLIB_OPTIMIZATION_GUIDE.md** - Qlib 参数优化完整指南
   - Tuner 模块详细介绍
   - Optimizer 模块使用说明
   - 实际应用建议和示例代码

2. **OPTIMIZATION_EXPLORATION_REPORT.md** - 本探索报告
   - 探索过程记录
   - 发现总结
   - 推荐方案

3. **optimize_supertrend_optuna.py** - Optuna 优化脚本
   - 使用贝叶斯优化
   - 支持多年度数据
   - 自动生成可视化图表

### 已有的优化脚本
1. **optimize_enhanced_supertrend.py** - 网格搜索优化（7个参数）
2. **optimize_enhanced_params.py** - 多频率参数优化

---

## 使用建议

### 对于 SuperTrend 策略

1. **快速探索：** 使用网格搜索（`optimize_enhanced_supertrend.py`）
2. **精细优化：** 使用 Optuna（`optimize_supertrend_optuna.py`）
3. **多频率：** 使用 `optimize_enhanced_params.py`

### 对于基于预测信号的投资组合策略

1. **使用 Qlib Tuner** - 原生支持，集成完整

### 对于投资组合权重优化

1. **使用 Qlib Optimizer** - 专门的权重优化工具

---

## 下一步行动

### 1. 尝试 Optuna 优化
```bash
cd /mnt/d/quant/qlib_backtest
python optimize_supertrend_optuna.py
```

### 2. 比较不同优化方法
- 网格搜索 vs 贝叶斯优化
- 优化时间 vs 结果质量

### 3. 探索其他优化目标
- 最大化收益率
- 最小化最大回撤
- 多目标优化

---

## 参考资料

### Qlib 官方资源
- 官方文档：https://qlib.readthedocs.io/
- GitHub 仓库：https://github.com/microsoft/qlib
- 策略文档：https://qlib.readthedocs.io/en/latest/component/strategy.html

### 优化库文档
- Optuna：https://optuna.readthedocs.io/
- Hyperopt：http://hyperopt.github.io/hyperopt/

### 相关文章
- Hyperparameter optimization - Wikipedia
- Qlib 论文中提到的 HTE (Hyperparameters Tuning Engine)

---

## 结论

**Qlib 确实提供了参数优化功能，但主要针对特定类型的策略：**

1. **Tuner** - 适用于基于预测信号的投资组合策略
2. **Optimizer** - 适用于投资组合权重优化
3. **RL** - 适用于订单执行和策略学习

**对于技术指标策略（如 SuperTrend），推荐使用 Optuna 等第三方优化库：**
- 更灵活、更高效
- 学习曲线更低
- 可视化工具更完善

**已提供的工具：**
- 完整的 Qlib 优化指南（`QLIB_OPTIMIZATION_GUIDE.md`）
- Optuna 优化脚本（`optimize_supertrend_optuna.py`）
- 网格搜索脚本（已有）

可以根据具体需求选择合适的优化方法。
