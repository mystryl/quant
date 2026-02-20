#!/usr/bin/env python3
"""
趋势标签单元测试

验证标签生成的正确性和无未来函数污染
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from labels.trend_label_final import (
    generate_trend_labels,
    validate_no_lookahead,
    TREND_LABEL_CONFIG
)


def test_label_generation():
    """测试标签生成功能"""
    print("\n" + "="*60)
    print("测试1: 标签生成功能")
    print("="*60)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='60min')
    df = pd.DataFrame({
        'datetime': dates,
        'close': np.linspace(100, 110, 100) + np.random.randn(100) * 0.5
    })

    # 生成标签
    df = generate_trend_labels(df, price_col='close')

    # 验证
    assert 'future_return' in df.columns, "缺少future_return列"
    assert 'trend_label' in df.columns, "缺少trend_label列"
    assert df['trend_label'].notna().sum() == 80, f"有效标签数量应为80，实际{df['trend_label'].notna().sum()}"
    assert df['trend_label'].iloc[-20:].isna().all(), "最后20个标签应为NaN"

    print("✓ 标签生成功能测试通过")
    return True


def test_no_lookahead():
    """测试无未来函数污染"""
    print("\n" + "="*60)
    print("测试2: 无未来函数污染")
    print("="*60)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='60min')
    df = pd.DataFrame({
        'datetime': dates,
        'close': np.linspace(100, 110, 100)
    })

    # 生成标签
    df = generate_trend_labels(df, price_col='close')

    # 验证最后window个样本无标签
    window = TREND_LABEL_CONFIG['window']
    last_labels = df['trend_label'].iloc[-window:]
    assert last_labels.isna().all(), f"最后{window}个样本不应有标签"

    # 验证shift(-window)的正确性
    sample_idx = 50
    current_price = df['close'].iloc[sample_idx]
    future_price = df['close'].iloc[sample_idx + window]
    expected_return = (future_price - current_price) / current_price
    actual_return = df['future_return'].iloc[sample_idx]

    assert abs(expected_return - actual_return) < 1e-10, \
        f"未来收益率计算错误: 预期{expected_return}, 实际{actual_return}"

    print("✓ 无未来函数污染测试通过")
    return True


def test_label_distribution():
    """测试标签分布合理性"""
    print("\n" + "="*60)
    print("测试3: 标签分布合理性")
    print("="*60)

    # 创建测试数据（有明确上涨趋势）
    dates = pd.date_range('2024-01-01', periods=200, freq='60min')
    prices = np.linspace(100, 120, 200) + np.random.randn(200) * 0.5
    df = pd.DataFrame({
        'datetime': dates,
        'close': prices
    })

    # 生成标签
    df = generate_trend_labels(df, price_col='close')

    # 统计标签
    valid_labels = df['trend_label'].dropna()
    label_counts = valid_labels.value_counts()

    print(f"\n标签分布:")
    print(f"  上涨 (1): {label_counts.get(1, 0)}")
    print(f"  震荡 (0): {label_counts.get(0, 0)}")
    print(f"  下跌 (-1): {label_counts.get(-1, 0)}")

    # 验证标签值域
    assert set(valid_labels.unique()).issubset({-1, 0, 1}), \
        f"标签值域错误: {set(valid_labels.unique())}"

    print("✓ 标签分布合理性测试通过")
    return True


def test_threshold_logic():
    """测试阈值逻辑正确性"""
    print("\n" + "="*60)
    print("测试4: 阈值逻辑正确性")
    print("="*60)

    # 创建固定价格数据
    dates = pd.date_range('2024-01-01', periods=50, freq='60min')
    base_price = 100.0

    # 创建已知收益率的数据
    df = pd.DataFrame({
        'datetime': dates,
        'close': [base_price] * 50
    })

    # 手动设置未来价格以测试阈值
    # 上涨案例：收益率 > 0.3%
    df.loc[0, 'close'] = 100.0
    df.loc[20, 'close'] = 100.5  # 未来20根涨0.5%

    # 下跌案例：收益率 < -0.3%
    df.loc[1, 'close'] = 100.0
    df.loc[21, 'close'] = 99.5  # 未来20根跌-0.5%

    # 震荡案例：-0.3% <= 收益率 <= 0.3%
    df.loc[2, 'close'] = 100.0
    df.loc[22, 'close'] = 100.1  # 未来20根涨0.1%

    # 生成标签
    df = generate_trend_labels(df, price_col='close')

    # 验证标签
    assert df['trend_label'].iloc[0] == 1, "收益率0.5%应标记为上涨(1)"
    assert df['trend_label'].iloc[1] == -1, "收益率-0.5%应标记为下跌(-1)"
    assert df['trend_label'].iloc[2] == 0, "收益率0.1%应标记为震荡(0)"

    print("✓ 阈值逻辑正确性测试通过")
    return True


def test_data_integrity():
    """测试数据完整性"""
    print("\n" + "="*60)
    print("测试5: 数据完整性")
    print("="*60)

    # 读取实际生成的标签数据
    label_file = project_root / 'data/labels/final_labels_20bars.csv'

    if not label_file.exists():
        print(f"⚠ 标签文件不存在: {label_file}")
        print("跳过数据完整性测试")
        return True

    df = pd.read_csv(label_file)

    # 验证列
    required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'trend_label']
    for col in required_cols:
        assert col in df.columns, f"缺少必需列: {col}"

    # 验证数据类型
    assert df['trend_label'].dtype in [np.float64, np.int64, object], \
        f"trend_label类型错误: {df['trend_label'].dtype}"

    # 验证值域
    valid_labels = df['trend_label'].dropna()
    unique_labels = set(valid_labels.unique())
    assert unique_labels.issubset({-1, 0, 1}), \
        f"标签值域错误: {unique_labels}"

    # 验证时间顺序
    assert df['datetime'].is_monotonic_increasing, "时间序列不是递增的"

    print("✓ 数据完整性测试通过")
    print(f"  数据行数: {len(df)}")
    print(f"  有效标签: {len(valid_labels)}")
    print(f"  标签分布: 上涨{valid_labels.eq(1).sum()}, 震荡{valid_labels.eq(0).sum()}, 下跌{valid_labels.eq(-1).sum()}")

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始运行趋势标签单元测试")
    print("="*60)

    tests = [
        ("标签生成功能", test_label_generation),
        ("无未来函数污染", test_no_lookahead),
        ("标签分布合理性", test_label_distribution),
        ("阈值逻辑正确性", test_threshold_logic),
        ("数据完整性", test_data_integrity)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}失败: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}出错: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"测试结果: {passed}通过, {failed}失败")
    print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
