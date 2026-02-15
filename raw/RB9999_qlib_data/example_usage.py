"""
RB9999.XSGE 数据使用示例

展示如何在 Python 中加载和使用合并后的期货数据
"""

import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 70)
print("RB9999.XSGE 数据使用示例")
print("=" * 70)

# ============ 数据加载 ============
print("\n【1. 数据加载】")

# 数据路径
data_dir = Path('/mnt/d/quant/RB9999_qlib_data')
parquet_file = data_dir / 'RB9999_XSGE_all.parquet'
csv_file = data_dir / 'RB9999_XSGE_all.csv'

print(f"加载数据：{parquet_file}")
df = pd.read_parquet(parquet_file)

print(f"✅ 数据加载成功")
print(f"   行数：{len(df):,}")
print(f"   列数：{len(df.columns)}")
print(f"   日期范围：{df['date'].min()} 到 {df['date'].max()}")


# ============ 基本分析 ============
print("\n【2. 基本分析】")

# 2.1 价格统计
print("\n--- 2.1 价格统计 ---")
print(f"平均开盘价：{df['open'].mean():.2f}")
print(f"平均收盘价：{df['close'].mean():.2f}")
print(f"平均最高价：{df['high'].mean():.2f}")
print(f"平均最低价：{df['low'].mean():.2f}")
print(f"平均成交均价：{df['vwap'].mean():.2f}")

# 2.2 成交量分析
print("\n--- 2.2 成交量分析 ---")
print(f"平均成交量：{df['volume'].mean():.2f}")
print(f"成交量中位数：{df['volume'].median():.2f}")
print(f"最大单分钟成交：{df['volume'].max():,.0f}")
print(f"总成交金额：{df['money'].sum():,.0f}")

# 2.3 持仓量分析（期货重要指标）
print("\n--- 2.3 持仓量分析 ---")
print(f"平均持仓量：{df['open_interest'].mean():,.0f}")
print(f"持仓量中位数：{df['open_interest'].median():,.0f}")
print(f"最大持仓量：{df['open_interest'].max():,.0f}")
print(f"持仓量标准差：{df['open_interest'].std():,.0f}")


# ============ 技术指标计算 ============
print("\n【3. 技术指标计算】")

# 3.1 简单移动平均
print("\n--- 3.1 移动平均线 ---")
df_sorted = df.sort_values('date')
df_sorted['MA5'] = df_sorted['close'].rolling(window=5).mean()
df_sorted['MA20'] = df_sorted['close'].rolling(window=20).mean()
df_sorted['MA60'] = df_sorted['close'].rolling(window=60).mean()

latest_date = df_sorted['date'].max()
latest_price = df_sorted[df_sorted['date'] == latest_date].iloc[0]
print(f"最新日期：{latest_date}")
print(f"  收盘价：{latest_price['close']:.2f}")
print(f"  MA5：{latest_price['MA5']:.2f}")
print(f"  MA20：{latest_price['MA20']:.2f}")
print(f"  MA60：{latest_price['MA60']:.2f}")

# 3.2 价格变化率
print("\n--- 3.2 价格变化率 ---")
df_sorted['price_change'] = df_sorted['close'].pct_change() * 100
df_sorted['price_change_5min'] = df_sorted['close'].pct_change(5) * 100

latest_change = df_sorted[df_sorted['date'] == latest_date].iloc[0]
print(f"最新单分钟变化率：{latest_change['price_change']:.4f}%")
print(f"最近5分钟变化率：{latest_change['price_change_5min']:.4f}%")

# 3.3 波动率
print("\n--- 3.3 波动率 ---")
df_sorted['high_low_range'] = df_sorted['high'] - df_sorted['low']
df_sorted['daily_volatility'] = df_sorted.groupby(df_sorted['date'].dt.date)['high_low_range'].transform('max')

print(f"平均波动幅度（高-低）：{df_sorted['high_low_range'].mean():.2f}")
print(f"最大波动幅度：{df_sorted['high_low_range'].max():.2f}")


# ============ 期货特有分析 ============
print("\n【4. 期货特有分析】")

# 4.1 持仓量与价格关系
print("\n--- 4.1 持仓量与价格关系 ---")
# 按日期分组分析
daily_summary = df_sorted.groupby(df_sorted['date'].dt.date).agg({
    'close': 'last',
    'open_interest': 'last',
    'volume': 'sum'
}).reset_index()

# 计算相关性
correlation = df_sorted['close'].corr(df_sorted['open_interest'])
print(f"价格与持仓量相关系数：{correlation:.4f}")

# 4.2 涨跌分析
print("\n--- 4.2 涨跌统计 ---")
df_sorted['is_up'] = df_sorted['close'] > df_sorted['open']
up_ratio = df_sorted['is_up'].mean() * 100
print(f"上涨比例：{up_ratio:.2f}%")
print(f"下跌比例：{100 - up_ratio:.2f}%")

# 4.3 涨停跌停统计
print("\n--- 4.3 涨停跌停统计 ---")
# 接近涨停：涨幅 > 9.8%（10% 涨停 - 0.2%）
# 接近跌停：跌幅 > 9.8%
limit_threshold = 0.098
df_sorted['near_limit_up'] = (df_sorted['close'] - df_sorted['open']) / df_sorted['open'] > limit_threshold
df_sorted['near_limit_down'] = (df_sorted['open'] - df_sorted['close']) / df_sorted['open'] > limit_threshold

up_limit_count = df_sorted['near_limit_up'].sum()
down_limit_count = df_sorted['near_limit_down'].sum()
total_minutes = len(df_sorted)

print(f"接近涨停次数：{up_limit_count} ({up_limit_count/total_minutes*100:.3f}%)")
print(f"接近跌停次数：{down_limit_count} ({down_limit_count/total_minutes*100:.3f}%)")


# ============ Qlib 格式准备 ============
print("\n【5. Qlib 格式准备】")

# 5.1 核心字段映射
print("\n--- 5.1 Qlib 标准字段映射 ---")
qlib_fields_map = {
    'date': 'datetime',      # 日期时间（索引）
    'open': 'open',           # 开盘价 ($open)
    'close': 'close',         # 收盘价 ($close)
    'high': 'high',           # 最高价 ($high)
    'low': 'low',             # 最低价 ($low)
    'volume': 'volume',       # 成交量 ($volume)
    'vwap': 'vwap',           # 成交均价 ($vwap)
    'factor': 'factor',       # 复权因子 ($factor)
    'open_interest': 'open_interest'  # 持仓量（自定义字段）
}

print("字段映射：")
for csv_col, qlib_col in qlib_fields_map.items():
    status = "✅" if csv_col in df.columns else "❌"
    print(f"  {status} {csv_col:20s} -> {qlib_col}")

# 5.2 导出 Qlib 兼容格式
print("\n--- 5.2 导出 Qlib 兼容 CSV ---")
qlib_output_file = data_dir / 'RB9999_XSGE_for_qlib.csv'
qlib_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'vwap', 'factor']
df_sorted[qlib_columns].to_csv(qlib_output_file, index=False)
print(f"✅ 已导出：{qlib_output_file}")
print(f"   文件大小：{qlib_output_file.stat().st_size / 1024 / 1024:.2f} MB")

# 5.3 按年份拆分（可选）
print("\n--- 5.3 按年份拆分数据 ---")
for year in [2023, 2024, 2025]:
    year_data = df_sorted[df_sorted['date'].dt.year == year]
    if len(year_data) > 0:
        year_file = data_dir / f'RB9999_XSGE_{year}.parquet'
        year_data.to_parquet(year_file, index=False)
        print(f"✅ {year}年：{len(year_data):,} 行 -> {year_file.name}")


# ============ 数据统计摘要 ============
print("\n【6. 数据统计摘要】")
print("\n--- 6.1 数据完整性 ---")
print(f"总数据点：{len(df):,}")
print(f"无缺失值：{'✅ 是' if df.isnull().sum().sum() == 0 else '❌ 否'}")
print(f"数据跨度：{(df['date'].max() - df['date'].min()).days} 天")

print("\n--- 6.2 存储效率 ---")
print(f"Parquet 格式大小：{parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
print(f"完整 CSV 大小（17 列）：{csv_file.stat().st_size / 1024 / 1024:.2f} MB")
print(f"Qlib CSV 大小（8 列）：{qlib_output_file.stat().st_size / 1024 / 1024:.2f} MB")
print(f"Parquet 相比完整 CSV 节省：{(1 - parquet_file.stat().st_size / csv_file.stat().st_size) * 100:.1f}%")

print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)

print("\n下一步：")
print("1. 使用 Qlib 的 dump_bin.py 将数据转换为 Qlib 格式")
print("2. 命令示例：")
print(f"   cd /mnt/d/quant/qlib")
print(f"   python scripts/dump_bin.py dump_all \\")
print(f"       --csv_path {data_dir} \\")
print(f"       --qlib_dir ~/.qlib/qlib_data/cn_data \\")
print(f"       --include_fields open,close,high,low,volume,vwap,factor \\")
print(f"       --file_suffix csv")
