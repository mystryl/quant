# V03混合策略参数优化

## 目录结构

```
optimization/
├── __init__.py              # 模块初始化
├── config.py                # 参数空间和优化配置
├── optuna_optimizer.py      # Optuna贝叶斯优化器
├── validator.py             # 参数验证器（过拟合检测、稳定性测试）
└── results/                 # 优化结果输出目录
    ├── 20260218_HHMMSS/     # 时间戳目录
    │   ├── optuna_study.db  # Optuna数据库
    │   ├── optimization_history.csv  # 优化历史
    │   ├── optimization_report.json  # 优化报告
    │   └── validation_report.json    # 验证报告
```

## 使用说明

### 1. 安装依赖

```bash
pip install optuna optuna-dashboard pandas numpy
```

### 2. 运行参数优化

```bash
cd D:/quant/projects/qlib_msb_ob/strategies/v03_hybrid/optimization
python optuna_optimizer.py
```

优化时间：约45-60分钟（100次迭代）

### 3. 监控优化进度（可选）

在另一个终端运行Optuna Dashboard：

```bash
optuna-dashboard sqlite:///D:/quant/projects/qlib_msb_ob/strategies/v03_hybrid/optimization/results/optuna_study.db
```

### 4. 运行参数验证

优化完成后，运行验证测试：

```bash
python validator.py
```

### 5. 快速测试（减少迭代次数）

编辑 `optuna_optimizer.py` 中的 `n_trials` 参数：

```python
optimizer = V03OptunaOptimizer(
    n_trials=20,  # 从100减少到20
    train_years=[2023, 2024],
    validate_year=2025,
    freq='60min'
)
```

## 参数空间

| 参数名 | 范围 | 原值 | 作用 |
|--------|------|------|------|
| pivot_len | 5-10 | 7 | 枢轴点检测窗口 |
| msb_zscore | 0.2-0.8 | 0.3 | MSB动量阈值 |
| atr_period | 10-20 | 14 | ATR计算周期 |
| atr_multiplier | 0.8-2.0 | 1.0 | 止损ATR倍数 |
| tp1_multiplier | 0.3-1.0 | 0.5 | 第一止盈倍数 |
| tp2_multiplier | 0.8-1.5 | 1.0 | 第二止盈倍数 |
| tp3_multiplier | 1.2-2.5 | 1.5 | 第三止盈倍数 |

## 优化目标

组合得分 = 盈亏比×40% + 夏普比率×30% + 累计收益×20% - 回撤惩罚×10%

### 约束条件
- 最低胜率：70%
- 最大回撤：5%
- 最少交易次数：50次

### 目标指标
- 盈亏比：0.46 → 1.2+
- 夏普比率：1.17 → 2.0+
- 累计收益：16.31% → 25%+

## 优化结果解读

### 优化报告 (optimization_report.json)

```json
{
  "best_value": 1.935,          // 最优得分
  "best_params": {...},          // 最优参数
  "train_metrics": {...},        // 训练集性能
  "validation_metrics": {...},   // 验证集性能
  "overfitting_check": {...}     // 过拟合检测
}
```

### 验证报告 (validation_report.json)

包含：
- 基础性能验证
- 过拟合检测
- 参数稳定性测试
- Walk-Forward分析
- 蒙特卡洛模拟

## 中期优化结果

当前进度（25/100次试验）：

| 指标 | 原参数 | 最优参数（试验#15） |
|------|--------|---------------------|
| pivot_len | 7 | 6 |
| msb_zscore | 0.3 | 0.22 |
| atr_period | 14 | 12 |
| atr_multiplier | 1.0 | 1.11 |
| tp1_multiplier | 0.5 | 0.74 |
| tp2_multiplier | 1.0 | 1.01 |
| tp3_multiplier | 1.5 | 1.91 |
| 得分 | ~0.8 | 1.94 |

初步观察：
- MSB阈值降低（0.3→0.22）：产生更多信号
- ATR倍数增大（1.0→1.11）：放宽止损
- TP1增大（0.5→0.74）：提高止盈目标
- TP3增大（1.5→1.91）：提高最大盈利空间

## 下一步

1. 等待优化完成（约还需30-40分钟）
2. 查看最终优化报告
3. 运行验证测试
4. 根据验证结果决定是否进行精细网格搜索
