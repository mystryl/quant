#!/usr/bin/env python3
"""
验证 HC8888.XSGE 数据的完整性和正确性
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ==================== 配置 ====================
QLIB_DIR = Path("/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq")
INSTRUMENT = "HC8888.XSGE"
FREQ = "1min"

# 要验证的字段
FIELDS = ['$open', '$high', '$low', '$close', '$volume', '$amount', '$vwap', '$open_interest']

print("=" * 80)
print(f"验证 {INSTRUMENT} 数据完整性")
print("=" * 80)

# ==================== 1. 加载数据 ====================
print("\n【步骤 1】加载数据...")
data = {}

for field in FIELDS:
    file_path = QLIB_DIR / "instruments" / FREQ / field / f"{INSTRUMENT}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data[field] = df[field]
        print(f"  ✓ {field:15s}: {len(df):>10} 行")
    else:
        print(f"  ✗ {field:15s}: 文件不存在")

# ==================== 2. 检查数据一致性 ====================
print("\n【步骤 2】检查数据一致性...")

# 所有字段应该有相同的索引
first_index = data[FIELDS[0]].index
all_consistent = True

for field in FIELDS:
    if not data[field].index.equals(first_index):
        print(f"  ✗ {field} 的索引与其他字段不一致")
        all_consistent = False

if all_consistent:
    print(f"  ✓ 所有 {len(FIELDS)} 个字段的时间戳完全一致")

# ==================== 3. 时间范围检查 ====================
print("\n【步骤 3】时间范围检查...")

start_time = first_index.min()
end_time = first_index.max()
time_span = end_time - start_time

print(f"  开始时间: {start_time}")
print(f"  结束时间: {end_time}")
print(f"  时间跨度: {time_span.days} 天")

# 检查是否有时间间隔过大（缺失数据）
time_diffs = first_index.to_series().diff()
max_gap = time_diffs.max()
print(f"  最大时间间隔: {max_gap}")

if max_gap > pd.Timedelta(minutes=1):
    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=1)]
    print(f"  ⚠ 发现 {len(gaps)} 个时间间隔 > 1分钟")
    if len(gaps) > 0 and len(gaps) <= 10:
        print("\n  最大的几个间隔:")
        for idx, gap in gaps.nlargest(5).items():
            print(f"    {idx}: {gap}")

# ==================== 4. 缺失值检查 ====================
print("\n【步骤 4】缺失值检查...")

total_records = len(first_index)
has_missing = False

for field in FIELDS:
    missing_count = data[field].isna().sum()
    missing_pct = (missing_count / total_records) * 100

    if missing_count > 0:
        has_missing = True
        print(f"  ✗ {field:15s}: {missing_count:>10} 个缺失值 ({missing_pct:.2f}%)")
    else:
        print(f"  ✓ {field:15s}: 无缺失值")

if not has_missing:
    print(f"\n  ✓ 所有字段均无缺失值")

# ==================== 5. 数据范围和合理性检查 ====================
print("\n【步骤 5】数据范围和合理性检查...")

def check_field_range(field, series):
    """检查字段数据的合理性"""
    min_val = series.min()
    max_val = series.max()
    mean_val = series.mean()
    std_val = series.std()

    print(f"\n  {field}:")
    print(f"    最小值: {min_val:10.2f}")
    print(f"    最大值: {max_val:10.2f}")
    print(f"    平均值: {mean_val:10.2f}")
    print(f"    标准差: {std_val:10.2f}")

    # 检查负值或零值
    if field in ['$open', '$high', '$low', '$close']:
        if (series <= 0).any():
            neg_count = (series <= 0).sum()
            print(f"    ⚠ 警告: 发现 {neg_count} 个非正数值")
        else:
            print(f"    ✓ 无非正数值")
    elif field == '$volume':
        if (series < 0).any():
            neg_count = (series < 0).sum()
            print(f"    ⚠ 警告: 发现 {neg_count} 个负值")
        else:
            print(f"    ✓ 无负值")

    # 检查价格逻辑关系
    if field == '$close':
        open_vals = data['$open']
        high_vals = data['$high']
        low_vals = data['$low']

        # close 应该在 low 和 high 之间
        invalid_close = (series > high_vals) | (series < low_vals)
        if invalid_close.any():
            print(f"    ⚠ 警告: {invalid_close.sum()} 条记录的收盘价超出最高/最低价范围")
        else:
            print(f"    ✓ 收盘价均在最高/最低价范围内")

for field in FIELDS:
    check_field_range(field, data[field])

# ==================== 6. OHLC 逻辑检查 ====================
print("\n【步骤 6】价格逻辑关系检查...")

df_ohlc = pd.DataFrame({
    'open': data['$open'],
    'high': data['$high'],
    'low': data['$low'],
    'close': data['$close']
})

# 检查 high >= open,close >= low
invalid_high = df_ohlc['high'] < df_ohlc[['open', 'close']].max(axis=1)
invalid_low = df_ohlc['low'] > df_ohlc[['open', 'close']].min(axis=1)

if invalid_high.any():
    print(f"  ✗ 发现 {invalid_high.sum()} 条记录：最高价 < 开盘价或收盘价")
else:
    print(f"  ✓ 所有记录：最高价 >= 开盘价且最高价 >= 收盘价")

if invalid_low.any():
    print(f"  ✗ 发现 {invalid_low.sum()} 条记录：最低价 > 开盘价或收盘价")
else:
    print(f"  ✓ 所有记录：最低价 <= 开盘价且最低价 <= 收盘价")

# ==================== 7. 交易日历检查 ====================
print("\n【步骤 7】交易日历检查...")

calendar_file = QLIB_DIR / "calendars" / "day.txt"
if calendar_file.exists():
    with open(calendar_file, 'r') as f:
        trading_days = [line.strip() for line in f if line.strip()]

    print(f"  交易日历记录数: {len(trading_days)}")

    # 检查数据是否覆盖所有交易日
    data_days = first_index.normalize().unique()
    calendar_days = pd.to_datetime(trading_days)

    missing_in_data = set(calendar_days) - set(data_days)
    if missing_in_data:
        print(f"  ⚠ 交易日历中有 {len(missing_in_data)} 天在数据中缺失")
    else:
        print(f"  ✓ 所有交易日都有数据")

else:
    print(f"  ✗ 交易日历文件不存在")

# ==================== 8. 统计摘要 ====================
print("\n【步骤 8】统计摘要...")
print(f"\n  总记录数: {total_records:,}")
print(f"  数据字段数: {len(FIELDS)}")
print(f"  数据完整性: {(1 - sum([data[f].isna().sum() for f in FIELDS]) / (total_records * len(FIELDS))) * 100:.2f}%")

# ==================== 9. 数据质量评分 ====================
print("\n【步骤 9】数据质量评分...")

issues = []

if has_missing:
    issues.append("存在缺失值")

if not all_consistent:
    issues.append("字段时间戳不一致")

if invalid_high.any() or invalid_low.any():
    issues.append("OHLC 逻辑错误")

if (data['$close'] <= 0).any():
    issues.append("存在非正价格")

if len(issues) == 0:
    print("  ✓✓✓ 数据质量: 优秀")
    print("  ✓ 所有检查通过，数据完整且正确")
elif len(issues) <= 2:
    print(f"  ⚠ 数据质量: 良好")
    print(f"  ⚠ 发现 {len(issues)} 个问题: {', '.join(issues)}")
else:
    print(f"  ✗ 数据质量: 需要改进")
    print(f"  ✗ 发现 {len(issues)} 个问题: {', '.join(issues)}")

# ==================== 10. 抽样检查 ====================
print("\n【步骤 10】数据抽样检查...")
print("\n  前 5 条数据:")
sample_df = pd.DataFrame({f: data[f] for f in FIELDS[:4]})
print(sample_df.head())

print("\n  后 5 条数据:")
print(sample_df.tail())

print("\n" + "=" * 80)
print("验证完成！")
print("=" * 80)
