# Code Review: KDJ 和 RSI 因子测试脚本

**文件**: `examples/test_kdj_rsi_factors.py`
**日期**: 2026-03-19
**审查重点**: 已遇到的实际 bug 和潜在问题

---

## 🔴 已遇到的严重 Bug

### Bug #1: 导入不存在的模块 (已修复)

**位置**: Line 21
**问题**:
```python
from src.utils.helpers import calculate_returns, calculate_rolling_oscillator
```

**错误**: `ImportError: cannot import name 'calculate_rolling_oscillator'`

**根本原因**: `calculate_rolling_oscillator` 函数在 `helpers.py` 中不存在

**修复方案**:
```python
from src.utils.helpers import calculate_returns  # 移除不存在的导入
```

**严重程度**: 🔴 严重 - 导致脚本无法运行

---

### Bug #2: 数据格式不匹配 qlib API (已修复)

**位置**: Line 313-314
**问题**:
```python
pred = factor_final.unstack()
label = returns_final.unstack()
metrics = evaluator.calculate_all(pred, label)
```

**错误**: `ValueError: If using all scalar values, you must pass an index`

**根本原因**:
- qlib 的 `calc_ic` 期望 MultiIndex Series，而不是 unstack 后的 DataFrame
- 数据转换逻辑错误，导致格式不匹配

**修复方案**:
```python
# 直接使用 Series（如果是 DataFrame，提取第一列）
if isinstance(factor_final, pd.DataFrame):
    pred = factor_final.iloc[:, 0]
else:
    pred = factor_final

if isinstance(returns_final, pd.DataFrame):
    label = returns_final.iloc[:, 0]
else:
    label = returns_final

metrics = evaluator.calculate_all(pred, label)
```

**严重程度**: 🔴 严重 - 核心功能完全失败

---

### Bug #3: 调用了不存在的参数 (已修复)

**位置**: Line 284-287
**问题**:
```python
summary = aligner.get_alignment_summary(factor_aligned, returns_aligned)
```

**错误**: `TypeError: CycleAligner.get_alignment_summary() missing 1 required positional argument: 'shift'`

**根本原因**: 方法签名不匹配
```python
# 实际签名
def get_alignment_summary(self, factor_df, price_df, shift: int)
```

**修复方案**:
```python
# 方案 1: 不使用 get_alignment_summary，直接打印
print(f"   - 因子数据点数: {len(factor_aligned)}")
print(f"   - 收益率数据点数: {len(returns_aligned)}")
print(f"   - 对齐方式: default (T+1 to T+2)")

# 方案 2: 正确调用
summary = aligner.get_alignment_summary(factor_aligned, returns_aligned, shift=2)
```

**严重程度**: 🔴 严重 - 方法调用错误

---

### Bug #4: 使用了错误的参数名 (已修复)

**位置**: Line 374
**问题**:
```python
reports = report_gen.generate_full_report(
    factor_name=factor_name,
    metrics=metrics,
    scenario_results=scenario_results,
    evaluation=evaluation  # ❌ 错误的参数名
)
```

**错误**: `TypeError: ReportGenerator.generate_full_report() got an unexpected keyword argument 'evaluation'`

**根本原因**: 参数名应该是 `reliability_result` 而不是 `evaluation`

**修复方案**:
```python
reports = report_gen.generate_full_report(
    factor_name=factor_name,
    metrics=metrics,
    scenario_results=scenario_results,
    reliability_result=evaluation  # ✅ 正确的参数名
)
```

**严重程度**: 🟡 中等 - API 使用错误

---

### Bug #5: 保存路径重复 (已修复)

**位置**: Line 377-383
**问题**:
```python
md_path = os.path.join(output_dir, f'{factor_name}_report.md')
report_gen.save_markdown(reports['markdown'], md_path)
```

**错误**: `FileNotFoundError: [Errno 2] No such file or directory: 'output/kdj/output/kdj/KDJ_Cross_report.md'`

**根本原因**:
- `ReportGenerator` 内部已经处理了 `output_dir`
- 再次添加导致路径重复

**修复方案**:
```python
# 直接使用文件名，不添加 output_dir
md_filename = f'{factor_name}_report.md'
report_gen.save_markdown(reports['markdown'], md_filename)
```

**严重程度**: 🟡 中等 - 路径错误

---

### Bug #6: 不支持的参数 (已修复)

**位置**: Line 393
**问题**:
```python
charts = visualizer.save_all_charts(
    metrics, scenario_results, factor_name, format='png'  # ❌ 不支持 format 参数
)
```

**错误**: `TypeError: Visualizer.save_all_charts() got an unexpected keyword argument 'format'`

**根本原因**: `save_all_charts` 方法不接受 `format` 参数

**修复方案**:
```python
charts = visualizer.save_all_charts(
    metrics, scenario_results, factor_name  # ✅ 移除 format 参数
)
```

**严重程度**: 🟡 中等 - API 使用错误

---

## 🟡 潜在问题和改进建议

### Issue #1: 数据准备函数效率低

**位置**: Line 230-273
**问题**: 使用循环逐个处理股票，效率不高

**当前实现**:
```python
for instrument, df in data_dict.items():
    # 计算因子
    factor = factor_func(df)

    # 创建临时 DataFrame
    temp_factor = pd.DataFrame({
        'datetime': factor.index,
        'instrument': instrument,
        'factor': factor.values
    })
    factor_data.append(temp_factor)

# 合并所有股票的数据
factor_df = pd.concat(factor_data, ignore_index=True)
```

**改进建议**:
```python
# 使用向量化操作
data_frames = []
for instrument, df in data_dict.items():
    factor = factor_func(df)
    data_frames.append(pd.DataFrame({
        'factor': factor,
    }, index=pd.MultiIndex.from_product(
        [factor.index, [instrument]],
        names=['datetime', 'instrument']
    ))

factor_df = pd.concat(data_frames)
```

**影响**: 性能提升，特别是处理大量股票时

---

### Issue #2: 错误处理不足

**位置**: 整个脚本
**问题**: 缺少异常处理，一旦出错就无法继续

**当前**: 没有错误处理
**建议**: 添加 try-except 块
```python
try:
    metrics = evaluator.calculate_all(pred, label)
except Exception as e:
    print(f"❌ 性能评估失败: {e}")
    return None, None, None
```

---

### Issue #3: 硬编码的参数

**位置**: Line 435-440
**问题**: 参数硬编码在函数调用中
```python
kdj_metrics, kdj_scenarios, kdj_eval = analyze_factor(
    'KDJ_Cross',
    factor_kdj_df,
    price_df,
    output_dir='./output/kdj'  # 硬编码
)
```

**建议**: 使用配置对象
```python
config = {
    'kdj': {'output_dir': './output/kdj', 'n': 9, 'm1': 3, 'm2': 3},
    'rsi': {'output_dir': './output/rsi', 'fast_n': 6, 'slow_n': 14}
}
```

---

### Issue #4: 内存使用优化

**位置**: Line 272-273
**问题**: 一次性删除 NaN 可能导致内存浪费
```python
# 删除 NaN
factor_df = factor_df.dropna()
price_df = price_df.dropna()
```

**建议**: 在合并前就处理
```python
# 在计算因子时就处理
factor = factor_func(df).dropna()
```

---

### Issue #5: 缺少数据验证

**位置**: 整个脚本
**问题**: 没有验证输入数据的有效性

**建议**: 添加数据验证
```python
def validate_data(data_dict):
    """验证输入数据"""
    required_columns = ['open', 'high', 'low', 'close', 'volume']

    for instrument, df in data_dict.items():
        # 检查必需列
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{instrument} 缺少必需列: {missing_cols}")

        # 检查数据范围
        if (df['high'] < df['low']).any():
            raise ValueError(f"{instrument} 存在 high < low 的异常数据")

        # 检查缺失值
        if df[required_columns].isnull().all().any():
            print(f"⚠️ {instrument} 存在全为 NaN 的列")
```

---

### Issue #6: KDJ 计算可能不准确

**位置**: Line 60-88
**问题**: KDJ 的 RSV 计算可能有除零错误

**当前实现**:
```python
rsv = (close - low_n) / (high_n - low_n) * 100
```

**问题**: 当 `high_n == low_n` 时会产生除零错误或 inf

**修复建议**:
```python
# 避免除零
denominator = high_n - low_n
rsv = np.where(
    denominator != 0,
    (close - low_n) / denominator * 100,
    50  # 当 high == low 时，设为中性值
)
```

---

## 🟢 优点和良好实践

### ✅ 做得好的地方

1. **清晰的文档** - 每个函数都有详细的 docstring
2. **模块化设计** - 功能分解清晰
3. **完整的测试流程** - 包含数据准备、因子计算、性能评估、报告生成
4. **详细的结果输出** - 打印了所有关键指标
5. **可视化支持** - 生成了多种专业图表

---

## 📋 优先修复建议

### 🔴 高优先级（必须修复）

1. ✅ **Bug #1: 导入错误** - 已修复
2. ✅ **Bug #2: 数据格式错误** - 已修复
3. ✅ **Bug #3: API 调用错误** - 已修复

### 🟡 中优先级（建议修复）

4. ✅ **Bug #4-6: 参数名错误** - 已修复
5. **Issue #2: 错误处理** - 添加异常处理
6. **Issue #6: KDJ 除零错误** - 添加除零保护

### 🟢 低优先级（可选优化）

7. **Issue #1: 性能优化** - 向量化操作
8. **Issue #3: 参数配置** - 使用配置文件
9. **Issue #5: 数据验证** - 添加输入验证

---

## 🛠️ 修复后的代码框架

```python
def analyze_factor(factor_name, factor_df, price_df, output_dir='./output'):
    """
    分析单个因子（增强版，包含错误处理）
    """
    try:
        # 1. 周期对齐
        print(f"1. 周期对齐...")
        aligner = CycleAligner()
        factor_aligned, returns_aligned = aligner.align(
            factor_df, price_df, method='default'
        )

        # 2. 准备收益率数据
        returns_final = returns_aligned
        common_index = factor_aligned.index.intersection(returns_final.index)
        factor_final = factor_aligned.loc[common_index]
        returns_final = returns_final.loc[common_index]

        # 3. 性能评估（修复数据格式）
        evaluator = PerformanceEvaluator()

        # 确保 Series 格式
        pred = factor_final.iloc[:, 0] if isinstance(factor_final, pd.DataFrame) else factor_final
        label = returns_final.iloc[:, 0] if isinstance(returns_final, pd.DataFrame) else returns_final

        metrics = evaluator.calculate_all(pred, label)

        # ... 其他处理

    except Exception as e:
        print(f"❌ 因子分析失败 ({factor_name}): {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
```

---

## 📊 测试建议

### 单元测试
```python
def test_calculate_kdj():
    """测试 KDJ 计算"""
    # 测试正常情况
    # 测试边界情况（high == low）
    # 测试缺失值处理

def test_data_format():
    """测试数据格式转换"""
    # 验证 MultiIndex 格式正确
    # 验证 qlib API 兼容性
```

### 集成测试
```python
def test_full_pipeline():
    """测试完整流程"""
    # 使用小数据集测试端到端
    # 验证所有步骤都能正常工作
```

---

## 🎯 总结

### 当前状态
- ✅ 所有已知的 bug 都已修复
- ✅ 脚本可以成功运行并生成报告
- ⚠️ 仍有一些潜在问题需要优化

### 建议
1. **立即修复**: 添加错误处理（Issue #2）
2. **重要**: 修复 KDJ 除零错误（Issue #6）
3. **优化**: 性能和内存优化（Issue #1, #4）
4. **增强**: 添加数据验证（Issue #5）

### 风险评估
- **当前风险**: 🟡 中等（可以运行，但缺少错误处理）
- **生产就绪**: ❌ 否（需要增强错误处理和边界检查）
- **测试就绪**: ✅ 是（可以用于测试和演示）

---

**审查人**: Claude Code
**审查日期**: 2026-03-19
**文件版本**: 最终版本（所有已知 bug 已修复）
