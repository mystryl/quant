"""
快速开始指南

5分钟上手因子管理和性能评估
"""

from src.core import (
    FactorExpressionParser,
    PerformanceEvaluator,
    validate_factor_expression
)
import numpy as np
import pandas as pd

print("=" * 70)
print("因子管理和性能评估 - 快速开始")
print("=" * 70)

# ============================================================================
# 步骤 1: 验证因子表达式（防止未来函数）
# ============================================================================
print("\n【步骤 1】验证因子表达式")
print("-" * 70)

# 创建解析器
parser = FactorExpressionParser()

# 示例表达式
expressions = [
    ("安全的表达式", "Ref($close, 20) / $close - 1"),
    ("危险的表达式", "Ref($close, -1)"),  # 包含未来函数
]

for name, expr in expressions:
    try:
        is_valid = parser.validate_no_future_functions(expr)
        print(f"✓ {name}: {expr}")
    except ValueError as e:
        print(f"✗ {name}: {expr}")
        print(f"  原因: 包含未来函数")

# 提取字段
expr = "($close - Ref($low, 5)) / Ref($volume, 10)"
fields = parser.extract_fields(expr)
print(f"\n表达式: {expr}")
print(f"使用的字段: {fields}")

# ============================================================================
# 步骤 2: 准备数据
# ============================================================================
print("\n【步骤 2】准备模拟数据")
print("-" * 70)

np.random.seed(42)

# 创建日期和股票
dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
instruments = ['SH600000', 'SH600001', 'SH600002', 'SH600003', 'SH600004']

# 创建多索引
index = pd.MultiIndex.from_product(
    [dates, instruments],
    names=['datetime', 'instrument']
)

# 生成因子值（动量因子）
n = len(index)
factor_values = np.random.randn(n).cumsum()
factor_df = pd.DataFrame({'factor': factor_values}, index=index)

# 生成未来收益率（与因子有一定相关性）
return_values = factor_values * 0.05 + np.random.randn(n) * 0.1
return_df = pd.DataFrame({'return': return_values}, index=index)

print(f"✓ 数据创建完成")
print(f"  时间范围: {dates[0].date()} 到 {dates[-1].date()}")
print(f"  股票数量: {len(instruments)}")
print(f"  观测值数量: {len(index):,}")

# ============================================================================
# 步骤 3: 评估因子性能
# ============================================================================
print("\n【步骤 3】评估因子性能")
print("-" * 70)

# 创建评估器
evaluator = PerformanceEvaluator(use_log_return=False)

# 计算所有性能指标
print("\n计算性能指标...")
metrics = evaluator.calculate_all(
    pred=factor_df['factor'],
    label=return_df['return'],
    quantile=0.2
)

# ============================================================================
# 步骤 4: 查看结果
# ============================================================================
print("\n【步骤 4】查看评估结果")
print("-" * 70)

# 打印关键指标
print("\n📊 预测能力指标:")
print(f"  IC 均值:     {metrics['ic_mean']:>10.4f}")
print(f"  IC 标准差:   {metrics['ic_std']:>10.4f}")
print(f"  ICIR:        {metrics['icir']:>10.4f}")
print(f"  Rank IC 均值: {metrics['rank_ic_mean']:>10.4f}")

print("\n💰 交易效果指标:")
print(f"  年化收益率:   {metrics['annual_return']:>10.2%}")
print(f"  夏普比率:    {metrics['sharpe_ratio']:>10.2f}")
print(f"  最大回撤:    {metrics['max_drawdown']:>10.2%}")
print(f"  胜率:       {metrics['win_rate']:>10.2%}")

# ============================================================================
# 步骤 5: 生成报告
# ============================================================================
print("\n【步骤 5】生成性能报告")
print("-" * 70)

report = evaluator.generate_report(metrics, "动量因子")
print(report)

# ============================================================================
# 进阶：比较多个因子
# ============================================================================
print("\n【进阶】比较多个因子")
print("-" * 70)

# 创建三个不同质量的因子
factors = {
    '高质量': np.random.randn(n) * 0.1 + np.random.randn(n) * 0.02,
    '中等质量': np.random.randn(n) * 0.05 + np.random.randn(n) * 0.1,
    '低质量': np.random.randn(n) * 0.01 + np.random.randn(n) * 0.15,
}

# 评估每个因子
results = {}
for name, values in factors.items():
    factor_series = pd.Series(values, index=index)
    metrics = evaluator.calculate_all(factor_series, return_df['return'])
    results[name] = metrics

# 打印对比表
print(f"\n{'因子名称':<12} {'IC均值':<10} {'ICIR':<10} {'年化收益':<12} {'夏普比率':<10}")
print("-" * 70)

for name, metrics in results.items():
    print(f"{name:<12} {metrics['ic_mean']:<10.4f} "
          f"{metrics['icir']:<10.4f} {metrics['annual_return']:<12.2%} "
          f"{metrics['sharpe_ratio']:<10.2f}")

# 找出最佳因子
best_factor = max(results.keys(), key=lambda k: results[k]['icir'])
print(f"\n🏆 最佳因子: {best_factor} (ICIR: {results[best_factor]['icir']:.4f})")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("恭喜！您已经掌握了因子管理和性能评估的基本用法")
print("=" * 70)

print("\n📚 下一步学习:")
print("  1. 查看完整示例: python examples/factor_and_evaluation_example.py")
print("  2. 阅读使用指南: docs/factor_engine_and_performance_evaluation.md")
print("  3. 探索更多功能: src/core/")

print("\n💡 常用功能:")
print("  • FactorExpressionParser: 验证因子表达式")
print("  • FactorManager: 管理和计算因子")
print("  • PerformanceEvaluator: 评估因子性能")

print("\n" + "=" * 70)
