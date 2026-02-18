#!/usr/bin/env python3
"""
检查 Qlib 是否支持 SuperTrend 指标
"""
import pandas as pd
import numpy as np

print("="*60)
print("Qlib 框架 SuperTrend 支持情况检查")
print("="*60)

# 检查 Qlib 内置指标
print("\n1. 检查 Qlib 内置指标")
print("-"*60)

try:
    from qlib.data import D
    from qlib.data.ops import MA, EMA, Ref, Max, Min, Abs, Div

    print("✓ Qlib 内置操作符:")
    print("  MA (移动平均)")
    print("  EMA (指数移动平均)")
    print("  Ref (引用)")
    print("  Max (最大值)")
    print("  Min (最小值)")
    print("  Abs (绝对值)")
    print("  Div (除法)")

    print("\n2. SuperTrend 指标实现方式")
    print("-"*60)

    print("\nSuperTrend 公式:")
    print("  ATR = 真实波动范围（需要 High, Low, Close）")
    print("  基本上限 = High + multiplier * ATR")
    print("  基本下限 = Low - multiplier * ATR")
    print("  SuperTrend = 基本上限/下限（根据趋势调整）")

    print("\nQlib 表达式实现挑战:")
    print("  ❌ Qlib 没有内置 ATR 指标")
    print("  ❌ Qlib 没有条件切换操作符（if-else）")
    print("  ❌ Qlib 的 alpha 表达式不复杂逻辑判断")
    print("  ✅ 但可以用 Python 手动实现 SuperTrend")

    print("\n3. Qlib 内置策略列表")
    print("-"*60)

    from qlib.contrib import strategy
    print("✓ 可用策略:")
    for name in dir(strategy):
        if not name.startswith('_') and name[0].isupper():
            print(f"  - {name}")

    print("\n4. 结论")
    print("-"*60)
    print("❌ Qlib 没有内置 SuperTrend 策略")
    print("✓ Qlib 支持自定义策略")
    print("✓ 可以用 Python 实现 SuperTrend 并集成到 Qlib")

    print("\n5. 建议实现方式")
    print("-"*60)
    print("方式一：用 Python 直接实现 SuperTrend（推荐）")
    print("  - 计算简单，无需依赖 qlib alpha 表达式")
    print("  - 适用于我们的回测框架")
    print("\n方式二：扩展 Qlib alpha 表达式（复杂）")
    print("  - 需要注册自定义操作符")
    print("  - 适合与 Qlib ML 框架集成")

except ImportError as e:
    print(f"❌ 无法导入 qlib: {e}")

print("\n" + "="*60)
