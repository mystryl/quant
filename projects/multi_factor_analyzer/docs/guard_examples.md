# 因子表达式解析器和未来函数检测 - 使用示例

本文档展示了如何使用因子表达式解析器和未来函数检测机制。

## 目录

1. [快速开始](#快速开始)
2. [基本用法](#基本用法)
3. [检测模式](#检测模式)
4. [真实场景示例](#真实场景示例)
5. [辅助函数使用](#辅助函数使用)
6. [最佳实践](#最佳实践)

## 快速开始

### 安装

```python
from src.utils.guard import FactorExpressionParser, validate_expression
from src.utils.helpers import calculate_ic, calculate_returns
```

### 基本验证

```python
# 创建解析器
parser = FactorExpressionParser()

# 验证安全的表达式
try:
    parser.validate_no_future_functions("Ref($close, 1)")
    print("✅ 表达式安全")
except FutureFunctionError as e:
    print(f"❌ {e}")

# 验证不安全的表达式
try:
    parser.validate_no_future_functions("Ref($close, -1)")
except FutureFunctionError as e:
    print(f"❌ 检测到未来函数: {e}")
```

## 基本用法

### 1. 验证因子表达式

```python
from src.utils.guard import FactorExpressionParser, FutureFunctionError

parser = FactorExpressionParser()

# 安全的因子表达式
safe_factors = [
    "Ref($close, 1) / Ref($close, 5) - 1",  # 动量因子
    "Std($close, 20) / Mean($close, 20)",   # 波动率因子
    "Mean($close, 20) / $close - 1",         # 反转因子
]

for factor in safe_factors:
    try:
        parser.validate_no_future_functions(factor)
        print(f"✅ {factor}")
    except FutureFunctionError as e:
        print(f"❌ {factor}\n{e}")

# 不安全的因子表达式（会抛出异常）
unsafe_factors = [
    "Ref($close, -1)",       # 未来引用
    "$close[-1]",            # 负索引
    "Roll($close, 5)",       # 未来偏移
]

for factor in unsafe_factors:
    try:
        parser.validate_no_future_functions(factor)
        print(f"✅ {factor}")
    except FutureFunctionError as e:
        print(f"❌ {factor}")
```

### 2. 分析表达式

```python
# 获取详细的分析结果
expr = "Ref($close, 1) + $volume"
result = parser.analyze_expression(expr)

print(f"表达式: {expr}")
print(f"安全: {result['safe']}")
print(f"变量: {parser.extract_variables(expr)}")

# 不安全的表达式
unsafe_expr = "Ref($close, -1) + Ref($open, -2)"
result = parser.analyze_expression(unsafe_expr)

print(f"\n表达式: {unsafe_expr}")
print(f"安全: {result['safe']}")
print(f"检测到的未来函数数量: {len(result['future_functions'])}")

for func in result['future_functions']:
    print(f"  - {func['pattern']}: {func['match']}")
    print(f"    建议: {func['suggestion']}")
```

### 3. 获取表达式信息

```python
# 获取完整的表达式信息
expr = "Mean($close, 20) / Std($close, 30)"
info = parser.get_expression_info(expr)

print(f"表达式: {info['expression']}")
print(f"变量: {info['variables']}")
print(f"安全: {info['safe']}")
print(f"未来函数数量: {info['future_function_count']}")
print(f"可疑模式数量: {info['suspicious_pattern_count']}")
```

## 检测模式

### 支持的未来函数模式

解析器能够检测以下未来函数模式：

1. **Ref() 使用负数**
   ```python
   "Ref($close, -1)"  # ❌ 错误：引用未来数据
   "Ref($close, 1)"   # ✅ 正确：引用历史数据
   ```

2. **负索引**
   ```python
   "$close[-1]"  # ❌ 错误：未来引用
   "$close[1]"   # ✅ 正确：历史引用
   "$close[0]"   # ✅ 正确：当前值
   ```

3. **Roll() 使用正数**
   ```python
   "Roll($close, 1)"   # ❌ 错误：向前滚动
   "Roll($close, -1)"  # ✅ 正确：向后滚动
   ```

4. **Shift() 使用负数**
   ```python
   "Shift($close, -1)"  # ❌ 错误：未来偏移
   "Shift($close, 1)"   # ✅ 正确：历史偏移
   ```

### 严格模式

```python
# 启用严格模式（会检测可疑模式）
parser_strict = FactorExpressionParser(strict_mode=True)

# 可能包含可疑模式的表达式
expr = "Mean($close, -1)"  # -1 可能是未来引用

import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    parser_strict.validate_no_future_functions(expr)

    if w:
        print(f"⚠️  警告: {w[0].message}")
```

## 真实场景示例

### 1. 动量因子

```python
# 正确的动量因子实现
momentum_factors = [
    # 5日动量
    "Ref($close, 1) / Ref($close, 5) - 1",

    # 20日动量
    "Ref($close, 1) / Ref($close, 20) - 1",

    # 60日动量
    "Ref($close, 1) / Ref($close, 60) - 1",
]

parser = FactorExpressionParser()
for factor in momentum_factors:
    if parser.validate_no_future_functions(factor):
        print(f"✅ {factor}")
```

### 2. 波动率因子

```python
# 波动率因子
volatility_factors = [
    # 标准差 / 均值
    "Std($close, 20) / Mean($close, 20)",

    # ATR (Average True Range)
    "Mean($high - $low, 20) / $close",

    # 收益率波动率
    "Std(Ref($close, 1) / Ref($close, 2) - 1, 20)",
]

for factor in volatility_factors:
    if parser.validate_no_future_functions(factor):
        print(f"✅ {factor}")
```

### 3. 反转因子

```python
# 价格反转因子
reversal_factors = [
    # 短期反转
    "Ref($close, 1) - Ref($close, 5)",

    # 标准化反转
    "(Ref($close, 1) - Ref($close, 5)) / Ref($close, 5)",

    # 相对强弱
    "($close - Mean($close, 20)) / Std($close, 20)",
]

for factor in reversal_factors:
    if parser.validate_no_future_functions(factor):
        print(f"✅ {factor}")
```

### 4. 成交量因子

```python
# 成交量相关因子
volume_factors = [
    # 换手率
    "$volume / Mean($volume, 20)",

    # 量比
    "$volume / Ref($volume, 1)",

    # 成交量波动率
    "Std($volume, 20) / Mean($volume, 20)",
]

for factor in volume_factors:
    if parser.validate_no_future_functions(factor):
        print(f"✅ {factor}")
```

### 5. 复合因子

```python
# 组合多个因子
composite_factors = [
    # 动量 + 波动率
    "(Ref($close, 1) / Ref($close, 20) - 1) / (Std($close, 20) / Mean($close, 20))",

    # 价格 + 成交量
    "($close - Ref($close, 5)) / Ref($close, 5) * ($volume / Mean($volume, 20))",

    # 多因子组合
    "Mean($close, 20) / $close + $volume / Mean($volume, 20)",
]

for factor in composite_factors:
    if parser.validate_no_future_functions(factor):
        print(f"✅ {factor}")
```

## 辅助函数使用

### 1. 数据标准化

```python
from src.utils.helpers import normalize, remove_outliers
import pandas as pd

# 准备数据
data = pd.Series([1, 2, 3, 4, 5, 100])

# 移除异常值
clean_data = remove_outliers(data, method='quantile')
print(f"原始数据: {data.tolist()}")
print(f"处理后: {clean_data.tolist()}")

# 标准化
normalized = normalize(clean_data, method='zscore')
print(f"Z-score 标准化: {normalized.tolist()}")
```

### 2. 计算收益率

```python
from src.utils.helpers import calculate_returns, calculate_forward_returns
import pandas as pd

# 创建价格数据
prices = pd.DataFrame({
    'A': [100, 102, 101, 103, 105],
    'B': [50, 51, 52, 51, 53]
})

# 计算简单收益率
returns = calculate_returns(prices, method='simple')
print("简单收益率:")
print(returns)

# 计算未来收益率（用于标签）
forward_returns = calculate_forward_returns(prices, forward_period=2)
print("\n未来收益率 (T+2):")
print(forward_returns)
```

### 3. 计算IC和多空收益

```python
from src.utils.helpers import calculate_ic, calculate_long_short_return
import pandas as pd
import numpy as np

# 创建测试数据
dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'] * 3)
instruments = ['A', 'B', 'C'] * 3

# 因子数据
factor = pd.DataFrame({
    'factor': [1.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 2.0]
})
factor.index = pd.MultiIndex.from_arrays(
    [dates, instruments],
    names=['datetime', 'instrument']
)

# 收益率数据
returns = pd.DataFrame({
    'return': [0.01, 0.02, 0.03, 0.02, 0.03, 0.01, 0.03, 0.01, 0.02]
})
returns.index = pd.MultiIndex.from_arrays(
    [dates, instruments],
    names=['datetime', 'instrument']
)

# 计算IC
ic, rank_ic = calculate_ic(factor, returns)
print("IC:")
print(ic)
print("\nRank IC:")
print(rank_ic)
print(f"IC 均值: {ic.mean():.4f}")
print(f"IC 标准差: {ic.std():.4f}")

# 计算多空收益
ls_return, avg_return = calculate_long_short_return(factor, returns)
print("\n多空收益:")
print(ls_return)
print(f"年化多空收益: {ls_return.mean() * 252:.2%}")
```

### 4. 因子中性化

```python
from src.utils.helpers import neutralize
import pandas as pd

# 创建测试数据
dates = pd.to_datetime(['2020-01-01'] * 4 + ['2020-01-02'] * 4)
instruments = ['A', 'B', 'C', 'D'] * 2

# 因子数据
factor = pd.DataFrame({
    'factor': [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0]
})
factor.index = pd.MultiIndex.from_arrays(
    [dates, instruments],
    names=['datetime', 'instrument']
)

# 行业数据
industry = pd.DataFrame({
    'industry': ['Tech', 'Tech', 'Finance', 'Finance',
                 'Tech', 'Tech', 'Finance', 'Finance']
})
industry.index = pd.MultiIndex.from_arrays(
    [dates, instruments],
    names=['datetime', 'instrument']
)

# 执行行业中性化
neutralized_factor = neutralize(factor, industry=industry, method='orthogonal')
print("原始因子:")
print(factor)
print("\n中性化后因子:")
print(neutralized_factor)
```

### 5. 数据验证

```python
from src.utils.helpers import validate_data_format, check_missing_values, check_infinite_values
import pandas as pd
import numpy as np

# 创建测试数据
dates = pd.to_datetime(['2020-01-01', '2020-01-02'] * 3)
instruments = ['A', 'B', 'C'] * 2

data = pd.DataFrame({
    'factor': [1.0, np.nan, 3.0, np.inf, 5.0, 6.0]
})
data.index = pd.MultiIndex.from_arrays(
    [dates, instruments],
    names=['datetime', 'instrument']
)

# 验证数据格式
try:
    validate_data_format(data, "测试数据")
    print("✅ 数据格式正确")
except ValueError as e:
    print(f"❌ 数据格式错误: {e}")

# 检查缺失值
missing_stats = check_missing_values(data)
print(f"\n缺失值统计:")
print(f"  因子列: {missing_stats['missing_count']['factor']} 个")

# 检查无穷值
infinite_stats = check_infinite_values(data)
print(f"\n无穷值统计:")
print(f"  正无穷: {infinite_stats['positive_infinite']['factor']} 个")
print(f"  负无穷: {infinite_stats['negative_infinite']['factor']} 个")
```

## 最佳实践

### 1. 因子开发流程

```python
from src.utils.guard import FactorExpressionParser
from src.utils.helpers import (
    calculate_ic, calculate_long_short_return,
    normalize, remove_outliers
)

def develop_factor(expression, prices, volumes):
    """
    因子开发完整流程
    """
    # 1. 验证表达式无未来函数
    parser = FactorExpressionParser(strict_mode=True)
    try:
        parser.validate_no_future_functions(expression)
        print(f"✅ 表达式验证通过: {expression}")
    except FutureFunctionError as e:
        print(f"❌ 表达式验证失败: {e}")
        return None

    # 2. 计算因子值（这里需要实际实现因子计算逻辑）
    # factor_values = calculate_factor(expression, prices, volumes)

    # 3. 数据清洗
    # factor_clean = remove_outliers(factor_values)
    # factor_normalized = normalize(factor_clean, method='zscore')

    # 4. 计算未来收益率（标签）
    # forward_returns = calculate_forward_returns(prices, forward_period=2)

    # 5. 计算性能指标
    # ic, rank_ic = calculate_ic(factor_normalized, forward_returns)
    # ls_return, _ = calculate_long_short_return(factor_normalized, forward_returns)

    # 6. 评估因子
    # print(f"IC 均值: {ic.mean():.4f}")
    # print(f"ICIR: {ic.mean() / ic.std():.4f}")
    # print(f"年化多空收益: {ls_return.mean() * 252:.2%}")

    return True

# 示例使用
expression = "Ref($close, 1) / Ref($close, 20) - 1"
develop_factor(expression, None, None)
```

### 2. 批量验证因子

```python
from src.utils.guard import FactorExpressionParser

def batch_validate_factors(factor_list):
    """
    批量验证因子表达式
    """
    parser = FactorExpressionParser()
    results = {
        'safe': [],
        'unsafe': []
    }

    for name, expression in factor_list.items():
        try:
            parser.validate_no_future_functions(expression)
            results['safe'].append(name)
            print(f"✅ {name}: {expression}")
        except FutureFunctionError as e:
            results['unsafe'].append(name)
            print(f"❌ {name}: {expression}")
            print(f"   错误: {e}")

    return results

# 示例使用
factors = {
    'momentum_5d': "Ref($close, 1) / Ref($close, 5) - 1",
    'momentum_20d': "Ref($close, 1) / Ref($close, 20) - 1",
    'volatility': "Std($close, 20) / Mean($close, 20)",
    'bad_factor': "Ref($close, -1)",  # 错误的因子
}

results = batch_validate_factors(factors)
print(f"\n安全因子数量: {len(results['safe'])}")
print(f"不安全因子数量: {len(results['unsafe'])}")
```

### 3. 便捷函数使用

```python
from src.utils.guard import validate_expression, analyze_expression

# 快速验证
if validate_expression("Ref($close, 1)"):
    print("表达式安全")

# 快速分析
result = analyze_expression("Ref($close, 1) + $volume")
print(f"安全: {result['safe']}")
print(f"变量: {result.get('variables', [])}")
```

### 4. 错误处理

```python
from src.utils.guard import FactorExpressionParser, FutureFunctionError

parser = FactorExpressionParser()

def safe_validate(expression):
    """
    安全的验证函数，带有友好的错误处理
    """
    try:
        parser.validate_no_future_functions(expression)
        return True, "表达式安全"
    except FutureFunctionError as e:
        # 提取关键错误信息
        error_msg = str(e)
        if "未来函数" in error_msg:
            return False, "表达式包含未来函数，请检查"
        elif "负索引" in error_msg:
            return False, "表达式包含负索引，请使用正数"
        else:
            return False, f"验证失败: {error_msg}"

# 使用示例
is_safe, message = safe_validate("Ref($close, 1)")
print(f"验证结果: {is_safe}, 消息: {message}")

is_safe, message = safe_validate("Ref($close, -1)")
print(f"验证结果: {is_safe}, 消息: {message}")
```

## 总结

### 关键要点

1. **始终验证因子表达式**：在计算因子之前，先验证表达式无未来函数
2. **使用严格模式**：在开发阶段使用严格模式，可以检测到潜在问题
3. **理解检测模式**：熟悉支持的未来函数检测模式，避免误报
4. **完整的测试**：为每个因子编写单元测试，确保表达式正确
5. **使用辅助函数**：利用辅助函数简化数据处理和性能计算

### 常见陷阱

1. **负数参数**：`Ref($close, -1)` 是未来引用，应该使用正数
2. **负索引**：`$close[-1]` 是未来引用，应该使用正数或零
3. **混淆函数**：`Roll()` 和 `Shift()` 的参数含义不同
4. **复杂表达式**：在复杂表达式中容易忽略未来函数

### 参考资源

- 系统设计文档: `/Users/mystryl/Documents/Quant/projects/multi_factor_analyzer/SYSTEM_DESIGN.md`
- 代码实现: `src/utils/guard.py`
- 测试用例: `tests/test_guard.py`
- 辅助函数: `src/utils/helpers.py`
