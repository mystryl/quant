"""
RB9999 数据纯净度分析和去重分析

分别执行：
- 方案一：检查 RB9999 纯净数据（只包含 RB9999 相关文件）
- 方案二：去除重复数据
"""

import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 80)
print("RB9999 数据分析")
print("=" * 80)

# ============ 数据加载 ============
data_dir = Path('/mnt/d/quant/RB9999_qlib_data')
parquet_file = data_dir / 'RB9999_XSGE_all.parquet'

print(f"\n【步骤 1】加载数据")
print(f"文件：{parquet_file}")
df_all = pd.read_parquet(parquet_file)

print(f"✅ 数据加载成功")
print(f"   总行数：{len(df_all):,}")
print(f"   总列数：{len(df_all.columns)}")

# ============ 方案一：检查 RB9999 纯净数据 ============
print("\n" + "=" * 80)
print("【方案一】RB9999 纯净数据检查")
print("=" * 80)

# 筛选 RB9999 相关文件
rb_files = [
    'RB9999.XSGE_2023_1min.csv',
    'RB9999.XSGE_2024_1min.csv',
    'RB9999.XSGE_2025_12月.csv',
    'RB9999.XSGE_2025_1_1_2025_4_30_1min.csv',
    'RB9999.XSGE_2025_10_01_2025_10_31.csv',
    'RB9999/期货主力连续3月_8月1/RB9999.XSGE_20250301_20250801_1min.csv',
    'RB9999/期货主力连续8月1_9月1/RB9999.XSGE_1min_20250801_20250901.csv',
    'RB9999/期货主力连续9月1_9月30/RB9999.XSGE_1m_20250901_20250930.csv',
    'RB9999/期货主力连续指数_2025年11月/RB9999.XSGE_2025_11_01_2025_11_30.csv'
]

print(f"\n筛选 RB9999 相关文件：{len(rb_files)} 个")

# 筛选只包含 RB9999 数据的行（来自根目录和指数目录）
df_rb_pure = df_all[df_all['source_file'].isin(rb_files)]

print(f"✅ 筛选完成")
print(f"   RB9999 纯净数据行数：{len(df_rb_pure):,}")
print(f"   占比：{len(df_rb_pure)/len(df_all)*100:.2f}%")

# 分析纯净数据
df_rb_pure['date'] = pd.to_datetime(df_rb_pure['date'])

# 按年份统计
print(f"\n【纯净数据年份统计】")
for year in sorted(df_rb_pure['date'].dt.year.unique()):
    df_year = df_rb_pure[df_rb_pure['date'].dt.year == year]
    count = len(df_year)
    print(f"   {year}年：{count:,} 行 ({count/len(df_rb_pure)*100:.2f}%)")

# 按月份统计（2025年）
df_2025_pure = df_rb_pure[df_rb_pure['date'].dt.year == 2025]
print(f"\n【2025年各月份数据统计】")
for month in range(1, 13):
    df_month = df_2025_pure[df_2025_pure['date'].dt.month == month]
    count = len(df_month)
    print(f"   {month:2d}月：{count:,} 行")

# 保存纯净数据
rb_pure_file = data_dir / 'RB9999_XSGE_pure.parquet'
df_rb_pure.to_parquet(rb_pure_file, index=False)
print(f"\n✅ 纯净数据已保存：{rb_pure_file}")
print(f"   文件大小：{rb_pure_file.stat().st_size / 1024 / 1024:.2f} MB")

# ============ 方案二：去除重复数据 ============
print("\n" + "=" * 80)
print("【方案二】去除重复数据")
print("=" * 80)

print(f"\n原始数据：{len(df_all):,} 行")

# 方法一：完全重复（所有列相同）
df_full_dedup = df_all.drop_duplicates()
print(f"\n【去重方法一：完全重复】")
print(f"   去重后行数：{len(df_full_dedup):,}")
print(f"   重复行数：{len(df_all) - len(df_full_dedup):,}")
print(f"   去重率：{(len(df_all) - len(df_full_dedup))/len(df_all)*100:.2f}%")

# 方法二：基于关键字段去重（date + open + close + high + low）
df_key_dedup = df_all.drop_duplicates(subset=['date', 'open', 'close', 'high', 'low'])
print(f"\n【去重方法二：关键字段去重】")
print(f"   去重后行数：{len(df_key_dedup):,}")
print(f"   重复行数：{len(df_all) - len(df_key_dedup):,}")
print(f"   去重率：{(len(df_all) - len(df_key_dedup))/len(df_all)*100:.2f}%")

# 检查关键字段重复的分布
print(f"\n【关键字段重复分析】")
duplicates = df_all[df_all.duplicated(subset=['date', 'open', 'close', 'high', 'low'], keep=False)]
print(f"   重复行数：{len(duplicates):,}")
print(f"   占比：{len(duplicates)/len(df_all)*100:.2f}%")

# 按数据源统计重复
print(f"\n【按数据源统计重复】")
dup_by_source = duplicates.groupby('source_file').size().reset_index(name='count')
dup_by_source_sorted = dup_by_source.sort_values('count', ascending=False)
print(f"   重复最多的文件：")
for idx, row in dup_by_source_sorted.head(5).iterrows():
    print(f"     {row['source_file']:>70}: {row['count']:>10,} 行")

# 保存去重后的数据（方法一）
df_full_dedup_file = data_dir / 'RB9999_XSGE_dedup_full.parquet'
df_full_dedup.to_parquet(df_full_dedup_file, index=False)
print(f"\n✅ 完全去重数据已保存：{df_full_dedup_file}")

# 保存去重后的数据（方法二）
df_key_dedup_file = data_dir / 'RB9999_XSGE_dedup_key.parquet'
df_key_dedup.to_parquet(df_key_dedup_file, index=False)
print(f"✅ 关键字段去重数据已保存：{df_key_dedup_file}")

# ============ 对比分析 ============
print("\n" + "=" * 80)
print("【数据集对比】")
print("=" * 80)

print(f"\n{'数据集':>15} {'行数':>15} {'文件大小':>15}")
print(f"{'-' * 48}")
print(f"{'原始数据':>15} {len(df_all):>15,} {parquet_file.stat().st_size / 1024 / 1024:.2f:>15.2f} MB")
print(f"{'RB9999纯净':>15} {len(df_rb_pure):>15,} {rb_pure_file.stat().st_size / 1024 / 1024:.2f:>15.2f} MB")
print(f"{'完全去重':>15} {len(df_full_dedup):>15,} {df_full_dedup_file.stat().st_size / 1024 / 1024:.2f:>15.2f} MB")
print(f"{'关键字去重':>15} {len(df_key_dedup):>15,} {df_key_dedup_file.stat().st_size / 1024 / 1024:.2f:>15.2f} MB")
print(f"{'-' * 48}")

# 计算节省空间
space_saved_full = parquet_file.stat().st_size - df_full_dedup_file.stat().st_size
space_saved_key = parquet_file.stat().st_size - df_key_dedup_file.stat().st_size

print(f"\n【空间节省】")
print(f"   完全去重节省：{space_saved_full / 1024 / 1024:.2f} MB ({space_saved_full/parquet_file.stat().st_size*100:.1f}%)")
print(f"   关键字段去重节省：{space_saved_key / 1024 / 1024:.2f} MB ({space_saved_key/parquet_file.stat().st_size*100:.1f}%)")

# ============ 结论 ============
print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)

print("\n【1. RB9999 纯净数据】")
print(f"   ✅ 已提取 RB9999 相关文件的纯数据：{len(df_rb_pure):,} 行")
print(f"   ✅ 保存到：{rb_pure_file.name}")

print("\n【2. 重复数据分析】")
print(f"   ⚠️  发现重复数据：{len(duplicates):,} 行 ({len(duplicates)/len(df_all)*100:.2f}%)")
print(f"   💡 建议使用关键字段去重方法（保留同一时间的不同数据源）")
print(f"   ✅ 完全去重后：{len(df_full_dedup):,} 行")
print(f"   ✅ 关键字段去重后：{len(df_key_dedup):,} 行")

print("\n【3. 数据质量建议】")
print("   1. RB9999 纯净数据（方案一）适合：")
print("      - RB9999 专用分析")
print("      - 回测（避免其他品种干扰）")
print("   2. 关键字段去重数据（方案二）适合：")
print("      - 综合分析（包含所有品种的对比）")
print("      - 多品种套利研究")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

# 保存分析报告
report_file = data_dir / 'CLEANYSIS_REPORT.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("RB9999 数据纯净度和去重分析报告\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("方案一：RB9999 纯净数据\n")
    f.write("-" * 40 + "\n")
    f.write(f"原始数据：{len(df_all):,} 行\n")
    f.write(f"RB9999纯净：{len(df_rb_pure):,} 行\n")
    f.write(f"占比：{len(df_rb_pure)/len(df_all)*100:.2f}%\n")
    
    f.write("\n各年份数据量：\n")
    for year in sorted(df_rb_pure['date'].dt.year.unique()):
        df_year = df_rb_pure[df_rb_pure['date'].dt.year == year]
        f.write(f"  {year}：{len(df_year):,} 行\n")
    
    f.write(f"\n保存文件：{rb_pure_file.name}\n")
    f.write(f"文件大小：{rb_pure_file.stat().st_size / 1024 / 1024:.2f} MB\n")
    
    f.write("\n" + "=" * 60 + "\n\n")
    
    f.write("方案二：去重分析\n")
    f.write("-" * 40 + "\n")
    
    f.write(f"原始数据：{len(df_all):,} 行\n")
    
    f.write("去重方法一：完全重复\n")
    f.write(f"  去重后：{len(df_full_dedup):,} 行\n")
    f.write(f"  重复行数：{len(df_all) - len(df_full_dedup):,}\n")
    f.write(f"  去重率：{(len(df_all) - len(df_full_dedup))/len(df_all)*100:.2f}%\n")
    f.write(f"  保存文件：{df_full_dedup_file.name}\n")
    
    f.write("\n去重方法二：关键字段去重\n")
    f.write(f"  去重后：{len(df_key_dedup):,} 行\n")
    f.write(f"  重复行数：{len(df_all) - len(df_key_dedup):,}\n")
    f.write(f"  去重率：{(len(df_all) - len(df_key_dedup))/len(df_all)*100:.2f}%\n")
    f.write(f"   保存文件：{df_key_dedup_file.name}\n")
    
    f.write("\n重复数据统计\n")
    f.write(f"总重复行数（完全）：{len(df_all) - len(df_full_dedup):,}\n")
    f.write(f"总重复行数（关键字段）：{len(df_all) - len(df_key_dedup):,}\n")
    
    f.write("\n空间节省\n")
    f.write(f"完全去重节省：{space_saved_full / 1024 / 1024:.2f} MB\n")
    f.write(f"关键字去重节省：{space_saved_key / 1024 / 1024:.2f} MB\n")
    
    f.write("\n" + "=" * 60 + "\n")
    
    f.write("结论\n")
    f.write("-" * 40 + "\n")
    f.write("1. RB9999 纯净数据已提取，适合专用分析\n")
    f.write("2. 发现重复数据，建议根据使用场景选择去重方法\n")
    f.write("3. 关键字段去重保留更多数据，适合综合分析\n")
    f.write("4. 所有数据已保存为 Parquet 格式，便于访问\n")

print(f"\n✅ 分析报告已保存：{report_file}")
