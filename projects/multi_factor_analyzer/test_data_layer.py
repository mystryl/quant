"""
数据访问层测试脚本

测试 data 模块的基本功能
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))

# 添加 qlib_backtest 的 scripts 目录到路径
qlib_backtest_path = Path(__file__).resolve().parent.parent / "qlib_backtest"
if qlib_backtest_path.exists():
    scripts_path = qlib_backtest_path / "scripts"
    sys.path.insert(0, str(scripts_path))

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_provider():
    """测试 FactorDataProvider"""
    print("\n" + "=" * 60)
    print("测试 FactorDataProvider".center(60))
    print("=" * 60)

    try:
        # 使用绝对导入避免模块冲突
        from src.data.provider import FactorDataProvider

        provider = FactorDataProvider()

        # 列出可用合约
        print("\n1. 列出可用合约:")
        instruments = provider.list_instruments(pattern="HC*")
        print(f"   找到 {len(instruments)} 个 HC 开头的合约")
        if instruments:
            print(f"   示例: {instruments[:3]}")

        if not instruments:
            print("   ⚠️  没有找到合约，跳过后续测试")
            return

        # 获取单个合约数据
        print("\n2. 获取单个合约数据:")
        test_instrument = instruments[0]
        data = provider.get_factor_data(
            instruments=test_instrument,
            start_date="2024-01-01",
            end_date="2024-01-31",
            fields=["close", "volume"]
        )
        print(f"   ✓ 读取 {len(data)} 条数据")
        print(f"   字段: {list(data.columns)}")
        print(f"   时间范围: {data.index.min()} - {data.index.max()}")

        # 获取多个合约数据
        if len(instruments) >= 2:
            print("\n3. 获取多个合约数据:")
            data = provider.get_factor_data(
                instruments=instruments[:2],
                start_date="2024-01-01",
                end_date="2024-01-31",
                fields=["close", "volume"]
            )
            print(f"   ✓ 读取 {len(data)} 条数据")
            print(f"   索引层级: {data.index.nlevels}")
            print(f"   合约数: {data.index.get_level_values('instrument').nunique()}")

        # 获取交易日历
        print("\n4. 获取交易日历:")
        calendar = provider.get_calendar(test_instrument)
        print(f"   ✓ 交易日历长度: {len(calendar)}")
        print(f"   前5个交易日: {calendar[:5]}")

        # 缓存统计
        print("\n5. 缓存统计:")
        stats = provider.get_cache_stats()
        print(f"   缓存统计: {stats}")

        print("\n✅ FactorDataProvider 测试通过")

    except Exception as e:
        print(f"\n❌ FactorDataProvider 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_loader():
    """测试 DataLoader"""
    print("\n" + "=" * 60)
    print("测试 DataLoader".center(60))
    print("=" * 60)

    try:
        from src.data.loader import DataLoader
        from src.data.provider import FactorDataProvider

        # 先获取可用的合约
        provider = FactorDataProvider()
        instruments = provider.list_instruments(pattern="HC*")

        if not instruments:
            print("   ⚠️  没有找到合约，跳过测试")
            return

        loader = DataLoader(fill_method='ffill')

        # 基本加载
        print("\n1. 基本加载:")
        data = loader.load_data(
            instruments=instruments[0],
            start_date="2024-01-01",
            end_date="2024-01-31",
            fields=["close", "volume"]
        )
        print(f"   ✓ 读取 {len(data)} 条数据")

        # 批量加载
        if len(instruments) >= 2:
            print("\n2. 批量加载:")
            data_dict = loader.load_batch(
                instruments=instruments[:3],
                start_date="2024-01-01",
                end_date="2024-01-31",
                fields=["close", "volume"],
                show_progress=True
            )
            print(f"   ✓ 成功加载 {len(data_dict)} 个合约")
            for instrument, data in data_dict.items():
                print(f"      {instrument}: {len(data)} 条记录")

        # 缺失值处理
        print("\n3. 测试缺失值处理:")
        import pandas as pd
        import numpy as np

        test_data = pd.DataFrame({
            'close': [100, 101, np.nan, 103, np.nan, 105],
            'volume': [1000, np.nan, 1200, 1300, 1400, np.nan]
        })

        print("   原始数据:")
        print(f"      缺失值: {test_data.isnull().sum().sum()}")

        filled = loader._fill_missing_values(test_data.copy(), 'ffill')
        print(f"   前向填充后缺失值: {filled.isnull().sum().sum()}")

        print("\n✅ DataLoader 测试通过")

    except Exception as e:
        print(f"\n❌ DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_validator():
    """测试 DataValidator"""
    print("\n" + "=" * 60)
    print("测试 DataValidator".center(60))
    print("=" * 60)

    try:
        from src.data.validator import DataValidator
        import pandas as pd
        import numpy as np

        validator = DataValidator()

        # 数据质量检查
        print("\n1. 数据质量检查:")
        test_data = pd.DataFrame({
            'close': [100, 101, 102, np.nan, 104, 105, 1000],  # NaN 和异常值
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600],
            'high': [102, 103, 104, 105, 106, 107, 1005],
            'low': [99, 100, 101, 102, 103, 104, 995]
        })

        report = validator.check_data_quality(test_data)
        print(f"   ✓ 检查完成")
        print(f"   警告数: {len(report['warnings'])}")
        print(f"   错误数: {len(report['errors'])}")

        # 因子标准化
        print("\n2. 因子标准化:")
        factor = pd.Series([1, 2, 3, 4, 5, 100])

        print("   z-score 标准化:")
        zscore = validator.standardize_factor(factor, method='zscore')
        print(f"   ✓ 均值: {zscore.mean():.6f}, 标准差: {zscore.std():.6f}")

        print("   min-max 标准化:")
        minmax = validator.standardize_factor(factor, method='minmax')
        print(f"   ✓ 范围: [{minmax.min():.2f}, {minmax.max():.2f}]")

        print("   秩标准化:")
        rank = validator.standardize_factor(factor, method='rank')
        print(f"   ✓ 范围: [{rank.min():.2f}, {rank.max():.2f}]")

        # 因子中性化
        print("\n3. 因子中性化:")
        factor = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        market_cap = pd.Series([100, 200, 300, 400, 500])

        neutralized = validator.neutralize_factor(
            factor=factor,
            market_cap=market_cap
        )
        print(f"   ✓ 原始因子与市值相关系数: {factor.corr(market_cap):.3f}")
        print(f"   ✓ 中性化后相关系数: {neutralized.corr(market_cap):.3f}")

        # 去极值
        print("\n4. 去极值:")
        factor = pd.Series([1, 2, 3, 4, 5, 100, -50])
        print(f"   原始范围: [{factor.min():.2f}, {factor.max():.2f}]")

        winsorized = validator.winsorize_factor(factor, limits=(0.05, 0.05))
        print(f"   ✓ 去极值后范围: [{winsorized.min():.2f}, {winsorized.max():.2f}]")

        print("\n✅ DataValidator 测试通过")

    except Exception as e:
        print(f"\n❌ DataValidator 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_integration():
    """集成测试"""
    print("\n" + "=" * 60)
    print("集成测试".center(60))
    print("=" * 60)

    try:
        from src.data.provider import FactorDataProvider
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator

        # 初始化
        provider = FactorDataProvider()
        loader = DataLoader(fill_method='ffill')
        validator = DataValidator()

        # 获取合约
        instruments = provider.list_instruments(pattern="HC*")
        if not instruments:
            print("   ⚠️  没有找到合约，跳过测试")
            return

        test_instrument = instruments[0]

        # 1. 加载数据
        print("\n1. 加载数据:")
        data = loader.load_data(
            instruments=test_instrument,
            start_date="2024-01-01",
            end_date="2024-01-31",
            fields=["open", "high", "low", "close", "volume"]
        )
        print(f"   ✓ 加载 {len(data)} 条数据")

        # 2. 数据质量检查
        print("\n2. 数据质量检查:")
        validator.check_data_quality(data, return_report=True)
        print("   ✓ 检查完成")

        # 3. 计算简单因子（动量因子）
        print("\n3. 计算动量因子:")
        data['momentum'] = data['close'].pct_change(5)
        print(f"   ✓ 动量因子统计:")
        print(f"      均值: {data['momentum'].mean():.6f}")
        print(f"      标准差: {data['momentum'].std():.6f}")
        print(f"      范围: [{data['momentum'].min():.6f}, {data['momentum'].max():.6f}]")

        # 4. 标准化因子
        print("\n4. 标准化因子:")
        data['momentum_zscore'] = validator.standardize_factor(
            data['momentum'],
            method='zscore'
        )
        print(f"   ✓ 标准化后统计:")
        print(f"      均值: {data['momentum_zscore'].mean():.6f}")
        print(f"      标准差: {data['momentum_zscore'].std():.6f}")

        # 5. 去极值
        print("\n5. 去极值:")
        data['momentum_winsorized'] = validator.winsorize_factor(
            data['momentum_zscore'],
            limits=(0.05, 0.05)
        )
        print(f"   ✓ 去极值后范围: [{data['momentum_winsorized'].min():.2f}, {data['momentum_winsorized'].max():.2f}]")

        print("\n✅ 集成测试通过")

    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("数据访问层测试".center(60))
    print("=" * 60)

    # 运行测试
    test_provider()
    test_loader()
    test_validator()
    test_integration()

    print("\n" + "=" * 60)
    print("所有测试完成".center(60))
    print("=" * 60)
