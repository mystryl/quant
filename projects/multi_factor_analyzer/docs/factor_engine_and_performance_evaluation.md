# 因子管理器和性能评估引擎使用指南

本指南介绍如何使用因子管理器（FactorManager）和性能评估引擎（PerformanceEvaluator）来计算和评估因子性能。

## 目录

1. [快速开始](#快速开始)
2. [因子表达式解析器](#因子表达式解析器)
3. [因子管理器](#因子管理器)
4. [性能评估引擎](#性能评估引擎)
5. [完整示例](#完整示例)
6. [API 参考](#api-参考)

---

## 快速开始

### 安装依赖

```bash
pip install qlib pandas numpy
```

### 基本使用

```python
from src.core import FactorManager, PerformanceEvaluator
from qlib_backtest.scripts.data import SmartDataProvider

# 1. 初始化数据提供者
provider = SmartDataProvider("/path/to/data")

# 2. 创建因子管理器
manager = FactorManager(provider)

# 3. 注册因子
manager.register_factor(
    "MA20",
    "Ref($close, 20) / $close - 1",
    metadata={"description": "20日均线偏离度"}
)

# 4. 计算因子
factor_data = manager.calculate_factor(
    "MA20",
    instruments=["SH600000"],
    start_date="2020-01-01",
    end_date="2020-12-31"
)

# 5. 评估性能
evaluator = PerformanceEvaluator()
metrics = evaluator.calculate_all(factor_data, return_data)

print(f"IC 均值: {metrics['ic_mean']:.4f}")
print(f"ICIR: {metrics['icir']:.4f}")
```

---

## 因子表达式解析器

### 概述

因子表达式解析器（FactorExpressionParser）用于检测因子表达式中的未来函数，防止回测时的数据泄露问题。

### 未来函数检测

未来函数是指在计算因子时使用了未来的数据，这会导致回测结果失真。

```python
from src.core import FactorExpressionParser

parser = FactorExpressionParser()

# 安全的表达式
safe_expr = "Ref($close, 20) / $close - 1"
parser.validate_no_future_functions(safe_expr)  # ✓ 通过

# 危险的表达式
dangerous_expr = "Ref($close, -1)"  # 引用未来数据
parser.validate_no_future_functions(dangerous_expr)  # ✗ 抛出异常
```

### 主要功能

#### 1. 验证表达式安全性

```python
from src.core import validate_factor_expression

# 便捷函数
is_valid = validate_factor_expression("Mean($close, 20)")
```

#### 2. 提取表达式字段

```python
expr = "($close - Ref($low, 5)) / Ref($volume, 10)"
fields = parser.extract_fields(expr)
# 返回: ['close', 'low', 'volume']
```

#### 3. 复杂度分析

```python
score = parser.get_complexity_score("Ref($close, 5) / Ref($close, 20) - 1")
# 返回: 0-100 的复杂度分数
```

#### 4. 温和的安全检查

```python
is_safe, warnings = parser.check_expression_safety(expr)
# 不抛出异常，返回警告列表
```

### 支持的未来函数模式

解析器会检测以下危险模式：

- `Ref($close, -N)` - 正向偏移量（N>0）
- `$close[-N]` - 负索引
- `Roll(..., N)` - 不安全的滚动操作

---

## 因子管理器

### 概述

因子管理器（FactorManager）负责因子的注册、计算和缓存管理。

### 初始化

```python
from src.core import FactorManager
from qlib_backtest.scripts.data import SmartDataProvider

provider = SmartDataProvider("/path/to/data")
manager = FactorManager(
    data_provider=provider,
    cache_enabled=True,
    cache_dir="./factor_cache"
)
```

### 注册因子

#### 方式 1: 表达式字符串

```python
manager.register_factor(
    name="MA20",
    factor_def="Ref($close, 20) / $close - 1",
    metadata={
        "description": "20日均线偏离度",
        "category": "技术指标",
        "author": "Your Name"
    }
)
```

#### 方式 2: Python 函数

```python
def custom_momentum(provider, instruments, start_date, end_date):
    """自定义动量因子"""
    data = provider.get_data(
        instruments,
        ["$close", "$volume"],
        start_date,
        end_date
    )

    # 计算动量
    momentum = data["$close"].pct_change(20)

    return momentum

manager.register_factor("momentum", custom_momentum)
```

### 计算因子

#### 单个因子

```python
factor_data = manager.calculate_factor(
    name="MA20",
    instruments=["SH600000", "SH600001"],
    start_date="2020-01-01",
    end_date="2020-12-31",
    use_cache=True  # 使用缓存
)
```

#### 批量因子

```python
factor_list = ["MA20", "MA60", "VOL20"]
factors = manager.calculate_batch_factors(
    names=factor_list,
    instruments=["SH600000"],
    start_date="2020-01-01",
    end_date="2020-12-31"
)
# 返回: {"MA20": df1, "MA60": df2, "VOL20": df3}
```

### 因子管理

```python
# 列出所有因子
factors = manager.list_factors()
for factor in factors:
    print(f"{factor['name']}: {factor['type']}")

# 获取因子信息
info = manager.get_factor_info("MA20")
print(info['metadata']['description'])

# 删除因子
manager.unregister_factor("MA20")

# 清除缓存
manager.clear_cache("MA20")  # 清除特定因子
manager.clear_cache()        # 清除所有缓存
```

### 缓存机制

因子管理器支持自动缓存已计算的因子：

- **内存缓存**: 快速访问，进程重启后丢失
- **磁盘缓存**: 持久化存储，可跨进程使用

```python
# 禁用缓存
manager = FactorManager(provider, cache_enabled=False)

# 自定义缓存目录
manager = FactorManager(provider, cache_dir="./my_cache")
```

---

## 性能评估引擎

### 概述

性能评估引擎（PerformanceEvaluator）计算因子的预测能力和实际交易效果。

### 核心指标

| 指标 | 说明 | 优秀标准 |
|------|------|----------|
| IC 均值 | 因子预测能力 | > 0.03 |
| ICIR | IC 稳定性 | > 0.5 |
| Rank IC | 非线性预测能力 | > 0.04 |
| 年化收益率 | 实际交易效果 | > 5% |
| 夏普比率 | 风险调整收益 | > 1.5 |
| 最大回撤 | 风险控制 | < 15% |
| 胜率 | 交易成功率 | > 55% |

### 初始化

```python
from src.core import PerformanceEvaluator

# 使用简单收益率（默认）
evaluator = PerformanceEvaluator(use_log_return=False)

# 使用对数收益率
evaluator = PerformanceEvaluator(use_log_return=True)
```

### 计算性能指标

#### 一次性计算所有指标

```python
metrics = evaluator.calculate_all(
    pred=factor_df,      # 因子值
    label=return_df,     # 未来收益率
    quantile=0.2         # 多空分组分位数
)

# 访问指标
print(f"IC 均值: {metrics['ic_mean']:.4f}")
print(f"ICIR: {metrics['icir']:.4f}")
print(f"年化收益率: {metrics['annual_return']:.2%}")
```

#### 单独计算指标

```python
# IC 和 Rank IC
ic, rank_ic = evaluator.calculate_ic(factor_df, return_df)

# ICIR
icir = evaluator.calculate_icir(ic)

# 多空收益
long_short, avg_return = evaluator.calculate_long_short_return(
    factor_df, return_df, quantile=0.2
)

# 年化收益率
annual_return = evaluator.calculate_annual_return(daily_returns)

# 夏普比率
sharpe = evaluator.calculate_sharpe_ratio(daily_returns)

# 最大回撤
max_dd = evaluator.calculate_max_drawdown(daily_returns)

# 胜率
win_rate = evaluator.calculate_win_rate(daily_returns)
```

### 收益率计算

#### 简单收益率

```python
# 适合短期策略，更直观
simple_return = evaluator.calculate_simple_return(
    price_df,
    shift=2  # T+1 到 T+2
)
```

#### 对数收益率

```python
# 适合长期策略，可加性好
log_return = evaluator.calculate_log_return(
    price_df,
    shift=2
)
```

### 生成报告

```python
report = evaluator.generate_report(
    metrics=metrics,
    factor_name="MA20"
)
print(report)
```

输出示例：

```
============================================================
因子性能评估报告: MA20
============================================================

1. 预测能力指标
------------------------------------------------------------
   IC 均值:         0.0479
   IC 标准差:       0.3138
   ICIR:            0.1527
   Rank IC 均值:     0.0387
   Rank IC 标准差:     0.3173
   Rank ICIR:        0.1220

2. 交易效果指标
------------------------------------------------------------
   年化收益率:      210.92%
   夏普比率:          1.87
   最大回撤:        45.04%
   胜率:           53.83%

3. 综合评估
------------------------------------------------------------
   可靠性等级: D (较差)
   评价: 因子表现较差，不建议使用，建议重新设计。
============================================================
```

---

## 完整示例

### 示例 1: 定义和评估单个因子

```python
from src.core import FactorManager, PerformanceEvaluator
from qlib_backtest.scripts.data import SmartDataProvider

# 1. 初始化
provider = SmartDataProvider("/path/to/data")
manager = FactorManager(provider)

# 2. 注册因子
manager.register_factor(
    "MA20",
    "Ref($close, 20) / $close - 1"
)

# 3. 计算因子
factor_data = manager.calculate_factor(
    "MA20",
    instruments=["SH600000"],
    start_date="2020-01-01",
    end_date="2020-12-31"
)

# 4. 评估性能
evaluator = PerformanceEvaluator()
metrics = evaluator.calculate_all(
    factor_data['factor'],
    return_data['return']
)

# 5. 查看结果
print(f"IC: {metrics['ic_mean']:.4f}")
print(f"ICIR: {metrics['icir']:.4f}")
print(evaluator.generate_report(metrics, "MA20"))
```

### 示例 2: 比较多个因子

```python
factors = {
    "MA5": "Ref($close, 5) / $close - 1",
    "MA10": "Ref($close, 10) / $close - 1",
    "MA20": "Ref($close, 20) / $close - 1",
    "MA60": "Ref($close, 60) / $close - 1",
}

# 注册所有因子
for name, expr in factors.items():
    manager.register_factor(name, expr)

# 计算所有因子
results = {}
for name in factors.keys():
    factor_data = manager.calculate_factor(
        name,
        instruments=["SH600000"],
        start_date="2020-01-01",
        end_date="2020-12-31"
    )

    metrics = evaluator.calculate_all(
        factor_data['factor'],
        return_data['return']
    )

    results[name] = metrics

# 找出最佳因子
best_factor = max(results.keys(), key=lambda k: results[k]['icir'])
print(f"最佳因子: {best_factor} (ICIR: {results[best_factor]['icir']:.4f})")
```

### 示例 3: 自定义因子函数

```python
def volatility_factor(provider, instruments, start_date, end_date):
    """波动率因子"""
    data = provider.get_data(
        instruments,
        ["$close"],
        start_date,
        end_date
    )

    # 计算20日滚动标准差
    volatility = data["$close"].pct_change().rolling(20).std()

    return volatility

# 注册自定义因子
manager.register_factor(
    "VOL20",
    volatility_factor,
    metadata={"description": "20日波动率"}
)

# 计算和评估
factor_data = manager.calculate_factor(
    "VOL20",
    instruments=["SH600000"],
    start_date="2020-01-01",
    end_date="2020-12-31"
)

metrics = evaluator.calculate_all(
    factor_data['VOL20'],
    return_data['return']
)
```

---

## API 参考

### FactorExpressionParser

```python
class FactorExpressionParser:
    def __init__(self, custom_patterns=None)
    def validate_no_future_functions(self, expression) -> bool
    def check_expression_safety(self, expression) -> Tuple[bool, List[str]]
    def extract_fields(self, expression) -> List[str]
    def get_complexity_score(self, expression) -> float
```

### FactorManager

```python
class FactorManager:
    def __init__(self, data_provider, cache_enabled=True, cache_dir=None)
    def register_factor(self, name, factor_def, metadata=None)
    def unregister_factor(self, name)
    def calculate_factor(self, name, instruments, start_date, end_date, use_cache=True)
    def calculate_batch_factors(self, names, instruments, start_date, end_date)
    def list_factors(self) -> List[Dict]
    def get_factor_info(self, name) -> Dict
    def clear_cache(self, name=None)
```

### PerformanceEvaluator

```python
class PerformanceEvaluator:
    def __init__(self, use_log_return=False, date_col='datetime')
    def calculate_all(self, pred, label, quantile=0.2) -> Dict
    def calculate_ic(self, pred, label) -> Tuple[pd.Series, pd.Series]
    def calculate_icir(self, ic) -> float
    def calculate_long_short_return(self, pred, label, quantile=0.2)
    def calculate_annual_return(self, returns, periods_per_year=252) -> float
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0) -> float
    def calculate_max_drawdown(self, returns) -> float
    def calculate_win_rate(self, returns) -> float
    def calculate_log_return(self, price_df, shift=2) -> pd.DataFrame
    def calculate_simple_return(self, price_df, shift=2) -> pd.DataFrame
    def generate_report(self, metrics, factor_name="Factor") -> str
```

---

## 最佳实践

### 1. 因子设计原则

- **避免未来函数**: 始终使用历史数据计算因子
- **考虑交易成本**: 因子应该在实际交易中有可操作性
- **简单有效**: 优先选择逻辑简单、解释性强的因子
- **稳健性**: 因子在不同市场环境下都应该有效

### 2. 性能评估建议

- **多维度评估**: 综合考虑 IC、IR、多空收益等指标
- **样本外测试**: 分段训练和测试，避免过拟合
- **参数敏感性**: 测试不同参数下的因子表现
- **交易成本**: 考虑手续费和滑点的影响

### 3. 因子组合策略

- **低相关性**: 选择相关性低的因子进行组合
- **互补性**: 选择不同类型、不同逻辑的因子
- **动态权重**: 根据因子表现动态调整权重

---

## 常见问题

### Q1: 如何判断因子是否有效？

A: 主要看以下指标：
- IC 均值 > 0.03，且 t 检验显著
- ICIR > 0.5，稳定性好
- 多空收益 > 5% 年化
- 最大回撤 < 15%

### Q2: 对数收益率 vs 简单收益率如何选择？

A:
- **简单收益率**: 更直观，适合短期策略（日内、日线）
- **对数收益率**: 可加性好，适合长期策略（周线、月线）

### Q3: 如何避免未来函数？

A:
- 使用 `FactorExpressionParser` 自动检测
- 检查所有 `Ref` 函数的第二个参数
- 避免使用负索引
- 在代码审查时重点关注

### Q4: 因子缓存应该开启吗？

A:
- **开发阶段**: 建议开启，提高迭代速度
- **生产环境**: 根据数据量和更新频率决定
- **注意**: 缓存会占用磁盘空间，定期清理

---

## 参考资源

- [系统设计文档](../SYSTEM_DESIGN.md)
- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [因子研究论文集](https://papers.ssrn.com/)

---

## 更新日志

- **2024-03-19**: 初始版本，实现因子管理器和性能评估引擎
- **待定**: 计划添加因子相关性分析、因子组合优化等功能
