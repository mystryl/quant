"""
统一数据管理系统 - 使用示例

演示如何使用 SmartDataProvider 进行数据管理和分析。
"""

import matplotlib.pyplot as plt
from projects.qlib_backtest.scripts.data import ParquetDataProvider, get_data


def example_1_basic_usage():
    """示例1：基本使用"""
    print("=" * 60)
    print("示例1: 基本使用 - 读取和分析数据")
    print("=" * 60)

    # 创建数据提供者
    provider = ParquetDataProvider()

    # 读取数据
    df = provider.get_data(
        "HC8888.XSGE",
        "2024-01-01",
        "2024-01-31",
        fields=["open", "high", "low", "close", "volume"],
        format='parquet'  # 使用 Parquet 快速读取
    )

    print(f"\n读取 {len(df)} 条数据")
    print("\n数据预览:")
    print(df.head())

    print("\n数据统计:")
    print(df.describe())


def example_2_convenience_function():
    """示例2：使用便捷函数"""
    print("\n" + "=" * 60)
    print("示例2: 便捷函数 - 快速读取数据")
    print("=" * 60)

    # 使用便捷函数
    df = get_data(
        "HC8888.XSGE",
        "2024-01-01",
        "2024-01-05",
        fields=["close"]
    )

    print(f"\n读取 {len(df)} 条数据")
    print(df.head())


def example_3_list_instruments():
    """示例3：列出可用合约"""
    print("\n" + "=" * 60)
    print("示例3: 列出可用合约")
    print("=" * 60)

    provider = ParquetDataProvider()

    # 列出所有合约
    all_instruments = provider.list_instruments()
    print(f"\n总共 {len(all_instruments)} 个合约")

    # 使用模式匹配
    hc_instruments = provider.list_instruments(pattern="HC*")
    print(f"HC 开头的合约: {hc_instruments}")


def example_4_qlib_conversion():
    """示例4：转换为 Qlib 格式"""
    print("\n" + "=" * 60)
    print("示例4: 转换为 Qlib 格式")
    print("=" * 60)

    provider = ParquetDataProvider()

    # 转换单个合约
    print("\n转换合约: HC8888.XSGE")
    provider.convert_to_qlib("HC8888.XSGE", force=False, progress=True)
    print("转换完成！")

    # 读取转换后的数据
    df = provider.get_data(
        "HC8888.XSGE",
        "2024-01-01",
        "2024-01-05",
        fields=["open", "close"],
        format='qlib'
    )

    print(f"\n从 Qlib 格式读取 {len(df)} 条数据")
    print(df.head())


def example_5_calendar():
    """示例5：获取交易日历"""
    print("\n" + "=" * 60)
    print("示例5: 获取交易日历")
    print("=" * 60)

    provider = ParquetDataProvider()

    # 获取交易日历
    calendar = provider.get_calendar("HC8888.XSGE")

    print(f"\n交易日历共 {len(calendar)} 个交易日")
    print(f"范围: {calendar[0]} 到 {calendar[-1]}")

    # 显示最近10个交易日
    print("\n最近10个交易日:")
    for date in calendar[-10:]:
        print(f"  {date}")


def example_6_cache_stats():
    """示例6：缓存统计"""
    print("\n" + "=" * 60)
    print("示例6: 缓存统计")
    print("=" * 60)

    provider = ParquetDataProvider()

    # 多次读取相同数据（测试缓存）
    print("\n第一次读取（从磁盘）:")
    df1 = provider.get_data("HC8888.XSGE", "2024-01-01", "2024-01-05", fields=["close"])

    print("第二次读取（从缓存）:")
    df2 = provider.get_data("HC8888.XSGE", "2024-01-01", "2024-01-05", fields=["close"])

    # 查看缓存统计
    stats = provider.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  启用: {stats['enabled']}")
    print(f"  大小: {stats['size']} 条")
    print(f"  最大: {stats['max_size']} 条")


def example_7_visualization():
    """示例7：数据可视化"""
    print("\n" + "=" * 60)
    print("示例7: 数据可视化")
    print("=" * 60)

    provider = ParquetDataProvider()

    # 读取数据
    df = provider.get_data(
        "HC8888.XSGE",
        "2024-01-01",
        "2024-01-31",
        fields=["open", "high", "low", "close", "volume"]
    )

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 价格走势
    df[['close']].plot(ax=ax1)
    ax1.set_title('HC8888.XSGE 价格走势 (2024年1月)')
    ax1.set_ylabel('价格')
    ax1.grid(True)

    # 成交量
    df[['volume']].plot(ax=ax2)
    ax2.set_title('HC8888.XSGE 成交量 (2024年1月)')
    ax2.set_ylabel('成交量')
    ax2.set_xlabel('时间')
    ax2.grid(True)

    plt.tight_layout()

    # 保存图表
    output_file = "/tmp/hc8888_price_volume.png"
    plt.savefig(output_file, dpi=100)
    print(f"\n图表已保存到: {output_file}")

    # 在交互环境中可以显示图表
    # plt.show()


def example_8_batch_processing():
    """示例8：批量处理"""
    print("\n" + "=" * 60)
    print("示例8: 批量处理多个合约")
    print("=" * 60)

    provider = ParquetDataProvider()

    # 获取所有合约
    instruments = provider.list_instruments()

    print(f"\n找到 {len(instruments)} 个合约")

    # 批量处理
    results = {}
    for instrument in instruments[:3]:  # 只处理前3个作为示例
        try:
            df = provider.get_data(
                instrument,
                "2024-01-01",
                "2024-01-31",
                fields=["close", "volume"],
                format='parquet'
            )

            # 计算统计信息
            results[instrument] = {
                "records": len(df),
                "avg_price": df['close'].mean(),
                "total_volume": df['volume'].sum()
            }

        except Exception as e:
            print(f"处理失败 {instrument}: {e}")

    # 显示结果
    print("\n处理结果:")
    for instrument, stats in results.items():
        print(f"\n{instrument}:")
        print(f"  记录数: {stats['records']}")
        print(f"  均价: {stats['avg_price']:.2f}")
        print(f"  总成交量: {stats['total_volume']:.0f}")


def main():
    """运行所有示例"""
    examples = [
        example_1_basic_usage,
        example_2_convenience_function,
        example_3_list_instruments,
        example_4_qlib_conversion,
        example_5_calendar,
        example_6_cache_stats,
        example_7_visualization,
        example_8_batch_processing
    ]

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\n示例 {i} 执行失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
