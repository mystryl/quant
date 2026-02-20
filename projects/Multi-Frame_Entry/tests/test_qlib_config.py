#!/usr/bin/env python3
"""
测试 Qlib 配置和数据加载

注意：需要在 qlib 环境中运行
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from data.qlib_config import (
        init_qlib,
        get_instruments,
        load_data,
        load_multi_freq_data
    )

    print("="*60)
    print("测试 Qlib 配置")
    print("="*60)

    # 初始化 Qlib
    print("\n1. 初始化 Qlib...")
    init_qlib()

    # 获取合约列表
    print("\n2. 获取合约列表...")
    instruments = get_instruments()
    print(f"   可用合约: {instruments}")

    # 加载 60min 数据
    print("\n3. 加载 60min 数据...")
    df_60min = load_data(
        instruments=['HC8888.XSGE'],
        fields=['$open', '$high', '$low', '$close', '$volume'],
        start_time='2024-01-01',
        end_time='2024-01-31',
        freq='60min'
    )
    print(f"   60min 数据形状: {df_60min.shape}")
    print(f"   前5行:")
    print(df_60min.head())

    # 加载多周期数据
    print("\n4. 加载多周期数据...")
    multi_freq_data = load_multi_freq_data(
        instruments=['HC8888.XSGE'],
        fields=['$close'],
        start_time='2024-01-01',
        end_time='2024-01-31',
        freqs=['5min', '15min', '60min']
    )
    print(f"   多周期数据加载完成:")
    for freq, df in multi_freq_data.items():
        print(f"     {freq}: {df.shape}")

    print("\n" + "="*60)
    print("✓ 所有测试通过")
    print("="*60)

except ImportError as e:
    print(f"错误: {e}")
    print("\n请确保在 Qlib 环境中运行此脚本")
    print("或激活正确的 Python 环境")
    sys.exit(1)
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
