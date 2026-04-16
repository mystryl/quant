# 可靠性评估模块文档

## 概述

可靠性评估模块提供了综合评估因子质量的功能，包括：

1. **可靠性评估器 (ReliabilityEvaluator)** - 综合评估因子可靠性
2. **因子相关性分析器 (FactorCorrelationAnalyzer)** - 分析因子之间的相关性
3. **配置系统** - 可配置的权重和阈值

## 安装

模块已包含在 `multi_factor_analyzer` 项目中，无需额外安装。

```python
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from core.reliability import ReliabilityEvaluator, evaluate_factor_reliability
from core.correlation_analyzer import FactorCorrelationAnalyzer, analyze_factor_correlation
from core.config import get_weights, get_thresholds
```

## 快速开始

### 1. 因子可靠性评估

```python
from core.reliability import ReliabilityEvaluator
import pandas as pd

# 准备性能指标
metrics = {
    'ic_mean': 0.05,           # IC 均值
    'icir': 0.8,               # IC 信息比率
    'rank_ic_mean': 0.06,      # Rank IC 均值
    'rank_icir': 0.9,          # Rank IC 信息比率
    'long_short_return': pd.Series([...]),  # 多空收益序列
    'annual_return': 0.12,      # 年化收益率
    'sharpe_ratio': 2.5,       # 夏普比率
    'max_drawdown': -0.08,     # 最大回撤
    'win_rate': 0.65,          # 胜率
}

# 创建评估器
evaluator = ReliabilityEvaluator()

# 评估因子
result = evaluator.evaluate(metrics, factor_name="MA20")

# 查看结果
print(f"可靠性等级: {result['reliability']}")  # 'A+', 'A', 'B', 'C', 'D', 'F'
print(f"综合评分: {result['total_score']:.2f}")
print(result['recommendation'])

# 生成详细报告
report = evaluator.generate_report(result, "MA20")
print(report)
```

### 2. 因子相关性分析

```python
from core.correlation_analyzer import FactorCorrelationAnalyzer

# 准备因子数据
factor_dict = {
    'MA20': ma20_df,   # pd.Series with MultiIndex (datetime, instrument)
    'MA60': ma60_df,
    'RSI': rsi_df,
}

# 创建分析器
analyzer = FactorCorrelationAnalyzer(method='spearman', threshold=0.7)

# 分析相关性
result = analyzer.analyze_correlation(factor_dict)

# 查看相关性矩阵
print(result['correlation_matrix'])

# 查看高度相关的因子对
for pair in result['high_correlation_pairs']:
    print(f"{pair['factor1']} - {pair['factor2']}: {pair['correlation']:.3f}")

# 查看建议
print(result['recommendation'])
```

## 配置选项

### 权重配置

系统提供了三种预定义权重配置：

#### 1. 默认权重 (DEFAULT_WEIGHTS)

```python
from core.config import DEFAULT_WEIGHTS

# 默认权重
DEFAULT_WEIGHTS = {
    'ic_stability': 0.40,      # IC 稳定性 (ICIR)
    'ic_absolute': 0.20,        # IC 绝对值
    'ir': 0.20,                 # 信息比率
    'long_short_return': 0.10,  # 多空收益
    'win_rate': 0.10,           # 胜率
}
```

**适用场景**：
- 一般量化投资策略
- 平衡风险和收益
- 适合大多数情况

#### 2. 保守型权重 (CONSERVATIVE_WEIGHTS)

```python
from core.config import CONSERVATIVE_WEIGHTS

# 保守型权重
CONSERVATIVE_WEIGHTS = {
    'ic_stability': 0.50,      # IC 稳定性权重更高
    'ic_absolute': 0.15,
    'ir': 0.20,
    'long_short_return': 0.10,
    'win_rate': 0.05,          # 胜率权重降低
}
```

**适用场景**：
- 养老金、保险资金等风险厌恶型投资者
- 需要长期稳定收益的策略
- 对回撤敏感的资金

#### 3. 激进型权重 (AGGRESSIVE_WEIGHTS)

```python
from core.config import AGGRESSIVE_WEIGHTS

# 激进型权重
AGGRESSIVE_WEIGHTS = {
    'ic_stability': 0.30,      # IC 稳定性权重降低
    'ic_absolute': 0.20,
    'ir': 0.15,
    'long_short_return': 0.25,  # 多空收益权重更高
    'win_rate': 0.10,
}
```

**适用场景**：
- 对冲基金、私募基金等追求高收益的策略
- 能够承受较大波动的资金
- 短期交易策略

### 使用预定义权重

```python
# 使用保守型权重
evaluator = ReliabilityEvaluator(strategy_type='conservative')
result = evaluator.evaluate(metrics)

# 使用激进型权重
evaluator = ReliabilityEvaluator(strategy_type='aggressive')
result = evaluator.evaluate(metrics)
```

### 自定义权重

```python
# 定义自定义权重
custom_weights = {
    'ic_stability': 0.25,
    'ic_absolute': 0.15,
    'ir': 0.15,
    'long_short_return': 0.35,  # 更注重收益
    'win_rate': 0.10,
}

# 使用自定义权重
evaluator = ReliabilityEvaluator(weights=custom_weights)
result = evaluator.evaluate(metrics)
```

### 阈值配置

系统提供了三种预定义阈值：

```python
from core.config import get_thresholds

# 默认阈值
thresholds = get_thresholds('default')

# 严格阈值
thresholds = get_thresholds('strict')

# 宽松阈值
thresholds = get_thresholds('relaxed')

# 使用自定义阈值
evaluator = ReliabilityEvaluator(strictness='strict')
result = evaluator.evaluate(metrics)
```

## 可靠性等级

系统使用六级评分系统：

| 等级 | 评分范围 | 描述 | 建议 |
|------|----------|------|------|
| A+ | 90-100 | 优秀 | 建议重点使用 |
| A | 80-90 | 良好 | 建议使用 |
| B | 70-80 | 中等 | 可以与其他因子组合使用 |
| C | 60-70 | 一般 | 建议谨慎使用或优化 |
| D | 50-60 | 较差 | 不建议单独使用 |
| F | 0-50 | 失败 | 不建议使用 |

## 评估维度

### 1. IC 稳定性 (ICIR)

- **权重**: 40% (默认)
- **指标**: ICIR = IC 均值 / IC 标准差
- **优秀标准**: ICIR ≥ 0.7
- **理论依据**: Grinold & Kahn, "Active Portfolio Management"

### 2. IC 预测能力 (IC 绝对值)

- **权重**: 20% (默认)
- **指标**: |IC 均值|
- **优秀标准**: |IC| ≥ 0.05
- **理论依据**: Grinold & Kahn, "Active Portfolio Management"

### 3. 信息比率 (IR)

- **权重**: 20% (默认)
- **指标**: 夏普比率
- **优秀标准**: Sharpe ≥ 2.0
- **理论依据**: Sharpe, "Information Ratio and Performance"

### 4. 多空收益

- **权重**: 10% (默认)
- **指标**: 年化收益率
- **优秀标准**: 年化收益 ≥ 10%
- **理论依据**: Jacobs & Levy, "Long-Short Equity Strategies"

### 5. 胜率

- **权重**: 10% (默认)
- **指标**: 正收益天数占比
- **优秀标准**: 胜率 ≥ 60%
- **理论依据**: Kaufman, "Trading Systems and Methods"

## 高级功能

### 1. 批量评估因子

```python
from core.reliability import ReliabilityEvaluator
from core.performance_eval import PerformanceEvaluator

evaluator = ReliabilityEvaluator()
perf_evaluator = PerformanceEvaluator()

results = {}

for factor_name, factor_df in factors.items():
    # 计算性能指标
    metrics = perf_evaluator.calculate_all(
        pred=factor_df,
        label=returns_df
    )

    # 评估可靠性
    result = evaluator.evaluate(metrics, factor_name=factor_name)
    results[factor_name] = result

# 排序
sorted_factors = sorted(
    results.items(),
    key=lambda x: x[1]['total_score'],
    reverse=True
)

for rank, (name, result) in enumerate(sorted_factors, 1):
    print(f"{rank}. {name}: {result['reliability']} ({result['total_score']:.2f})")
```

### 2. 因子去重建议

```python
from core.correlation_analyzer import FactorCorrelationAnalyzer

analyzer = FactorCorrelationAnalyzer()
result = analyzer.analyze_correlation(factor_dict)

# 查看高度相关的因子对
if result['high_correlation_pairs']:
    print("发现高度相关的因子对，建议去重：")
    for pair in result['high_correlation_pairs']:
        print(f"  - {pair['factor1']} 和 {pair['factor2']}")

# 查看处理建议
print(result['recommendation'])
```

### 3. 可视化相关性矩阵

```python
import matplotlib.pyplot as plt

analyzer = FactorCorrelationAnalyzer()
result = analyzer.analyze_correlation(factor_dict)

# 绘制相关性矩阵
fig = analyzer.plot_correlation_matrix(
    result['correlation_matrix'],
    save_path='correlation_matrix.png'
)

plt.show()
```

## 完整示例

参见 `examples/reliability_simple_demo.py` 和 `examples/reliability_evaluation_example.py`。

```bash
# 运行简化示例
python examples/reliability_simple_demo.py

# 运行完整示例
python examples/reliability_evaluation_example.py
```

## API 参考

### ReliabilityEvaluator

```python
class ReliabilityEvaluator:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        strategy_type: str = 'default',
        strictness: str = 'default'
    ):
        """初始化可靠性评估器"""

    def evaluate(
        self,
        metrics: Dict[str, Union[float, pd.Series]],
        scenario_results: Optional[Dict[str, any]] = None,
        factor_name: str = "Factor"
    ) -> Dict[str, any]:
        """评估因子可靠性"""

    def generate_report(
        self,
        evaluation_result: Dict[str, any],
        factor_name: str = "Factor"
    ) -> str:
        """生成评估报告"""
```

### FactorCorrelationAnalyzer

```python
class FactorCorrelationAnalyzer:
    def __init__(
        self,
        method: str = 'spearman',
        threshold: Optional[float] = None,
        min_periods: int = 10
    ):
        """初始化相关性分析器"""

    def analyze_correlation(
        self,
        factor_dict: Dict[str, pd.DataFrame],
        threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """分析因子相关性"""

    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Axes:
        """绘制相关性矩阵"""
```

## 测试

运行单元测试：

```bash
# 测试配置文件
pytest tests/test_reliability_config.py -v

# 测试相关性分析器
pytest tests/test_correlation_analyzer.py -v

# 测试可靠性评估器
pytest tests/test_reliability_evaluator.py -v

# 运行所有测试
pytest tests/test_reliability_*.py -v
```

## 参考文献

1. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management*. McGraw-Hill.
2. Sharpe, W. F. (1994). "The Sharpe Ratio". *Journal of Portfolio Management*.
3. Jacobs, B. I., & Levy, K. N. (1995). "Long-Short Equity Strategies". *Journal of Portfolio Management*.
4. Kaufman, P. J. (2013). *Trading Systems and Methods*. Wiley.

## 常见问题

### Q: 如何选择合适的权重配置？

A:
- **默认权重**: 适合大多数情况，平衡各项指标
- **保守型权重**: 适合风险厌恶型投资者，更注重稳定性
- **激进型权重**: 适合追求高收益的投资者，更注重收益
- **自定义权重**: 根据自己的投资理念和风险偏好调整

### Q: 如何处理高度相关的因子？

A: 有以下几种方法：
1. **因子筛选**: 保留表现更好的因子（IC、IR 更高）
2. **正交化处理**: 使用回归方法剔除相关性
3. **主成分分析**: 提取综合因子
4. **分组使用**: 在不同市场环境下分别使用

### Q: 可靠性等级为 C 或 D 的因子还能用吗？

A: 可以考虑：
1. **优化因子**: 重新设计因子逻辑
2. **组合使用**: 与其他因子组合，分散风险
3. **辅助因子**: 不作为主要因子，而是辅助判断
4. **放弃使用**: 如果优化后仍表现不佳，建议放弃

## 更新日志

### v1.0.0 (2024-03-19)
- 初始版本发布
- 实现可靠性评估器
- 实现因子相关性分析器
- 支持可配置的权重系统
- 提供三种预定义权重配置
- 完整的单元测试和文档
