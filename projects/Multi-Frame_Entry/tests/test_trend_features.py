#!/usr/bin/env python3
"""
趋势特征单元测试

验证特征计算的正确性和无未来函数污染
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.trend_features import TrendFeatures


def test_feature_calculation():
    """测试特征计算功能"""
    print("\n" + "="*60)
    print("测试1: 特征计算功能")
    print("="*60)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='60min')
    df = pd.DataFrame({
        'datetime': dates,
        'open': np.linspace(100, 110, 100) + np.random.randn(100) * 0.5,
        'high': np.linspace(101, 111, 100) + np.random.randn(100) * 0.5,
        'low': np.linspace(99, 109, 100) + np.random.randn(100) * 0.5,
        'close': np.linspace(100, 110, 100) + np.random.randn(100) * 0.5,
        'volume': np.random.randint(1000, 10000, 100)
    })

    # 确保OHLC逻辑正确
    df['high'] = df[['open', 'close']].max(axis=1) + 0.5
    df['low'] = df[['open', 'close']].min(axis=1) - 0.5

    # 计算特征
    calculator = TrendFeatures()
    df_result = calculator.compute_all_features(df, shift=False)

    # 验证特征数量
    original_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df_result.columns if col not in original_cols]

    assert len(feature_cols) == 57, f"应有57个特征（35原始+22技术指标），实际{len(feature_cols)}"

    # 验证关键特征存在
    required_features = [
        'ema60_slope', 'ema20_slope', 'adx', 'atr',
        'golden_cross', 'ma_alignment', 'rolling_std_20'
    ]
    for feat in required_features:
        assert feat in df_result.columns, f"缺少特征: {feat}"

    print("✓ 特征计算功能测试通过")
    print(f"  生成特征数: {len(feature_cols)}")
    return True


def test_no_lookahead():
    """测试无未来函数污染"""
    print("\n" + "="*60)
    print("测试2: 无未来函数污染（shift(1)）")
    print("="*60)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='60min')
    df = pd.DataFrame({
        'datetime': dates,
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'close': np.linspace(100, 110, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })

    # 计算特征（不shift）
    calculator = TrendFeatures()
    df_no_shift = calculator.compute_all_features(df.copy(), shift=False)

    # 计算特征（shift）
    df_shifted = calculator.compute_all_features(df.copy(), shift=True)

    # 验证shift效果
    original_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df_shifted.columns if col not in original_cols]

    # 第一行所有特征应为NaN
    first_row_features = df_shifted[feature_cols].iloc[0]
    assert first_row_features.isna().all(), "第一行特征应全部为NaN"

    # 验证shift正确性
    for feat in feature_cols[:5]:  # 检查前5个特征
        # shift后的第i行应该等于shift前的第i-1行
        for i in range(1, min(10, len(df))):
            original_val = df_no_shift[feat].iloc[i-1]
            shifted_val = df_shifted[feat].iloc[i]

            if pd.notna(original_val) and pd.notna(shifted_val):
                assert abs(original_val - shifted_val) < 1e-10, \
                    f"特征{feat} shift不正确: 行{i}"

    print("✓ 无未来函数污染测试通过")
    print(f"  第一行特征: 全部为NaN ✓")
    print(f"  shift(1)验证: 正确 ✓")
    return True


def test_feature_values_validity():
    """测试特征值的有效性"""
    print("\n" + "="*60)
    print("测试3: 特征值有效性")
    print("="*60)

    # 创建测试数据（上升趋势）
    dates = pd.date_range('2024-01-01', periods=200, freq='60min')
    prices = 100 + np.linspace(0, 20, 200) + np.random.randn(200) * 0.5

    df = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    })

    # 计算特征
    calculator = TrendFeatures()
    df_result = calculator.compute_all_features(df, shift=False)

    # 验证斜率特征（上升趋势应为正）
    ema60_slope_valid = df_result['ema60_slope'].dropna()
    assert ema60_slope_valid.mean() > 0, "上升趋势中EMA60斜率均值应为正"

    # 验证ADX范围（0-100）
    adx_valid = df_result['adx'].dropna()
    assert (adx_valid >= 0).all() and (adx_valid <= 100).all(), \
        f"ADX应在0-100范围，发现异常值: min={adx_valid.min()}, max={adx_valid.max()}"

    # 验证ATR为正
    atr_valid = df_result['atr'].dropna()
    assert (atr_valid >= 0).all(), "ATR应非负"

    # 验证金叉死叉为0/1
    golden_cross_valid = df_result['golden_cross'].dropna()
    assert set(golden_cross_valid.unique()).issubset({0, 1}), \
        "金叉死叉应为0或1"

    # 验证均线排列为-1/0/1
    ma_alignment_valid = df_result['ma_alignment'].dropna()
    assert set(ma_alignment_valid.unique()).issubset({-1, 0, 1}), \
        "均线排列应为-1、0或1"

    print("✓ 特征值有效性测试通过")
    print(f"  EMA60斜率均值: {ema60_slope_valid.mean():.6f} (应为正)")
    print(f"  ADX范围: {adx_valid.min():.2f} - {adx_valid.max():.2f}")
    print(f"  ATR均值: {atr_valid.mean():.2f}")
    return True


def test_feature_missing_values():
    """测试特征缺失值处理"""
    print("\n" + "="*60)
    print("测试4: 特征缺失值处理")
    print("="*60)

    # 读取实际生成的特征数据
    feature_file = project_root / 'data/features/trend_features.csv'

    if not feature_file.exists():
        print(f"⚠ 特征文件不存在: {feature_file}")
        print("跳过缺失值测试")
        return True

    df = pd.read_csv(feature_file)

    # 验证必需列存在
    required_cols = [
        'datetime', 'close', 'trend_label',
        'ema60_slope', 'adx', 'atr', 'golden_cross'
    ]
    for col in required_cols:
        assert col in df.columns, f"缺少必需列: {col}"

    # 统计缺失值
    original_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'trend_label']
    feature_cols = [col for col in df.columns if col not in original_cols]

    missing_stats = df[feature_cols].isna().sum()
    missing_pct = (missing_stats / len(df) * 100).round(2)

    print(f"\n特征缺失值统计:")
    print(f"  总行数: {len(df)}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  有效样本（所有特征都有值）: {df[feature_cols].notna().all(axis=1).sum()}")
    print(f"\n前10个特征的缺失值:")
    for feat in feature_cols[:10]:
        print(f"    {feat}: {missing_stats[feat]} ({missing_pct[feat]}%)")

    # 验证：第一行应该所有特征都是NaN（因为shift(1)）
    first_row_features = df[feature_cols].iloc[0]
    assert first_row_features.isna().all(), "第一行所有特征应为NaN"

    print("\n✓ 特征缺失值处理测试通过")
    print(f"  第一行特征全部为NaN (shift(1)正确)")
    return True


def test_data_integrity():
    """测试数据完整性"""
    print("\n" + "="*60)
    print("测试5: 数据完整性")
    print("="*60)

    # 读取实际数据
    feature_file = project_root / 'data/features/trend_features.csv'

    if not feature_file.exists():
        print(f"⚠ 特征文件不存在: {feature_file}")
        print("跳过数据完整性测试")
        return True

    df = pd.read_csv(feature_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 验证时间连续性
    assert df['datetime'].is_monotonic_increasing, "时间序列不是递增的"

    # 验证标签列存在
    assert 'trend_label' in df.columns, "缺少trend_label列"

    # 验证特征列存在
    original_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'trend_label']
    feature_cols = [col for col in df.columns if col not in original_cols]

    assert len(feature_cols) == 57, f"应有57个特征（35原始+22技术指标），实际{len(feature_cols)}"

    # 验证特征和标签对齐
    valid_mask = df[feature_cols].notna().all(axis=1)
    valid_labels = df.loc[valid_mask, 'trend_label'].dropna()

    print(f"\n✓ 数据完整性测试通过")
    print(f"  数据行数: {len(df)}")
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  有效样本（特征+标签都有值）: {len(valid_labels)}")
    print(f"  标签分布: 上涨{valid_labels.eq(1).sum()}, 震荡{valid_labels.eq(0).sum()}, 下跌{valid_labels.eq(-1).sum()}")

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始运行趋势特征单元测试")
    print("="*60)

    tests = [
        ("特征计算功能", test_feature_calculation),
        ("无未来函数污染", test_no_lookahead),
        ("特征值有效性", test_feature_values_validity),
        ("特征缺失值处理", test_feature_missing_values),
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
