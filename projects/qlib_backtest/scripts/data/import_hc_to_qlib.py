#!/usr/bin/env python3
"""
将 HC8888 合约数据从源目录导入到 Qlib 格式

适配 macOS 环境，支持批量导入多个 CSV 文件。
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil
from datetime import datetime

# ==================== 配置路径 ====================
# 源数据目录（macOS 路径）
SOURCE_DIR = Path("/Users/mystryl/Documents/K线数据库/期货商品指数1min最新")

# Qlib 数据输出目录（统一数据存储位置）
QLIB_DIR = Path("/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq")

# 合约代码
INSTRUMENT = "HC8888.XSGE"

# 频率列表
FREQUENCIES = ["1min"]

# 字段映射：源CSV字段 -> Qlib特征
FIELD_MAPPING = {
    'open': '$open',
    'high': '$high',
    'low': '$low',
    'close': '$close',
    'volume': '$volume',
    'money': '$amount',      # money 映射为 amount
    'avg': '$vwap',          # avg 映射为 vwap
    'open_interest': '$open_interest'
}

def find_hc_files(source_dir: Path) -> list[Path]:
    """查找所有 HC 合约 CSV 文件"""
    print(f"正在扫描目录: {source_dir}")
    hc_files = list(source_dir.rglob("HC*.csv"))
    print(f"找到 {len(hc_files)} 个 HC 文件")
    return sorted(hc_files)

def load_and_merge_csvs(files: list[Path]) -> pd.DataFrame:
    """加载并合并所有 CSV 文件"""
    all_data = []

    for file_path in files:
        try:
            print(f"  读取: {file_path.name}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # 检查必需列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"    警告: 缺少列 {missing_cols}，跳过此文件")
                continue

            # 重命名列以匹配 Qlib 格式
            df_renamed = df.rename(columns=FIELD_MAPPING)
            all_data.append(df_renamed)
            print(f"    成功: {len(df)} 行数据")

        except Exception as e:
            print(f"    错误: {e}")
            continue

    if not all_data:
        raise ValueError("没有成功加载任何数据！")

    # 合并所有数据
    print(f"\n合并 {len(all_data)} 个文件...")
    df_merged = pd.concat(all_data, axis=0)

    # 去除重复索引（保留最后出现的值）
    df_merged = df_merged[~df_merged.index.duplicated(keep='last')]

    # 按时间排序
    df_merged = df_merged.sort_index()

    print(f"合并后总计: {len(df_merged)} 行数据")
    print(f"时间范围: {df_merged.index.min()} 到 {df_merged.index.max()}")

    return df_merged

def save_to_qlib_format(df: pd.DataFrame, qlib_dir: Path, instrument: str, frequencies: list[str]):
    """保存为 Qlib 格式"""
    print(f"\n保存到 Qlib 目录: {qlib_dir}")

    # 为每个频率创建目录
    for freq in frequencies:
        print(f"\n处理频率: {freq}")

        # 为每个特征创建目录
        for qlib_field in FIELD_MAPPING.values():
            feature_dir = qlib_dir / "instruments" / freq / qlib_field
            feature_dir.mkdir(parents=True, exist_ok=True)

        # 保存每个特征
        for src_field, qlib_field in FIELD_MAPPING.items():
            # 检查原始列名或重命名后的列名
            column_to_use = qlib_field if qlib_field in df.columns else src_field
            if column_to_use in df.columns:
                # 提取单列数据
                series = df[column_to_use].copy()

                # 如果是日线频率，进行重采样
                if freq == "day":
                    series_daily = series.resample('D').last()
                    series_daily = series_daily.dropna()
                    series_to_save = series_daily
                    print(f"  {qlib_field}: {len(series_daily)} 行 (日线)")
                else:
                    series_to_save = series
                    print(f"  {qlib_field}: {len(series)} 行 ({freq})")

                # 保存文件
                output_file = qlib_dir / "instruments" / freq / qlib_field / f"{instrument}.csv"
                series_to_save.to_csv(output_file, header=[qlib_field])

    print(f"\n数据已保存到: {qlib_dir}")

def create_calendars(df: pd.DataFrame, qlib_dir: Path):
    """创建交易日历文件"""
    calendars_dir = qlib_dir / "calendars"
    calendars_dir.mkdir(parents=True, exist_ok=True)

    # 1分钟日历（所有有数据的时间点）
    calendar_1min = df.index.unique().strftime('%Y-%m-%d %H:%M:%S').tolist()
    with open(calendars_dir / "1min.txt", 'w') as f:
        f.write('\n'.join(calendar_1min))
    print(f"创建 1min 日历: {len(calendar_1min)} 个时间点")

    # 日线日历
    df_daily = df.resample('D').last().dropna()
    calendar_day = df_daily.index.strftime('%Y-%m-%d').tolist()
    with open(calendars_dir / "day.txt", 'w') as f:
        f.write('\n'.join(calendar_day))
    print(f"创建 day 日历: {len(calendar_day)} 个交易日")

def show_directory_structure(qlib_dir: Path):
    """显示目录结构"""
    print(f"\n=== Qlib 数据目录结构 ===")
    for root, dirs, files in os.walk(qlib_dir):
        level = root.replace(str(qlib_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:
            print(f'{subindent}{file}')
        if len(files) > 3:
            print(f'{subindent}... and {len(files) - 3} more files')

def main():
    print("=" * 60)
    print("HC8888 合约数据导入到 Qlib 格式")
    print("=" * 60)

    # 1. 查找所有 HC 文件
    hc_files = find_hc_files(SOURCE_DIR)

    if not hc_files:
        print("错误: 未找到任何 HC 文件！")
        return

    # 2. 加载并合并数据
    df = load_and_merge_csvs(hc_files)

    # 显示数据统计
    print(f"\n数据统计:")
    print(f"  总行数: {len(df)}")
    print(f"  时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"  列: {list(df.columns)}")

    # 3. 保存为 Qlib 格式
    # 清理旧数据（可选）
    instrument_dir = QLIB_DIR / "instruments" / "1min" / "$close" / INSTRUMENT
    if instrument_dir.exists():
        print(f"\n清理旧数据...")
        # 注意：这里不删除整个目录，只覆盖文件

    save_to_qlib_format(df, QLIB_DIR, INSTRUMENT, FREQUENCIES)

    # 4. 创建日历文件
    create_calendars(df, QLIB_DIR)

    # 5. 显示目录结构
    show_directory_structure(QLIB_DIR)

    print("\n" + "=" * 60)
    print("导入完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
