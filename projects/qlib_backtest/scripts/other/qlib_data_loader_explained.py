#!/usr/bin/env python3
"""
Qlib 数据加载器详解

Qlib 的数据加载机制：
1. DataProvider: 数据提供者，负责从不同来源加载数据
2. LocalDataProvider: 从本地文件系统加载数据
3. D 接口: 用户主要使用的数据访问接口

数据结构要求：
instruments/
  {freq}/              # 频率：1min, 5min, 15min, 60min, day 等
    {feature}/         # 特征：$open, $high, $low, $close, $volume 等
      {instrument}.csv # 合约数据，格式：datetime, value

重采样机制：
Qlib 支持从高频数据重采样到低频数据
例如：从 1min 重采样到 5min, 15min, 60min, day
"""

import pandas as pd
from pathlib import Path

print("="*60)
print("Qlib 数据加载器详解")
print("="*60)

print("\n1. 数据结构要求")
print("-"*60)
print("""
qlib_data/
├── calendars/              # 交易日历
│   ├── 1min.txt           # 1分钟级别的时间点
│   ├── 5min.txt           # 5分钟级别的时间点
│   ├── 15min.txt          # 15分钟级别的时间点
│   ├── 60min.txt          # 60分钟级别的时间点
│   └── day.txt            # 日级别的时间点
│
└── instruments/            # 合约数据
    ├── 1min/             # 1分钟数据
    │   ├── $open/        # 开盘价
    │   │   ├── RB9999.XSGE.csv
    │   │   └── ...
    │   ├── $high/        # 最高价
    │   ├── $low/         # 最低价
    │   ├── $close/       # 收盘价
    │   ├── $volume/      # 成交量
    │   └── ...
    ├── 5min/             # 5分钟数据（可选，可重采样）
    ├── 15min/            # 15分钟数据（可选，可重采样）
    ├── 60min/            # 60分钟数据（可选，可重采样）
    └── day/              # 日线数据（可选，可重采样）
""")

print("2. 数据文件格式")
print("-"*60)
print("""
每个 CSV 文件格式：
$close/RB9999.XSGE.csv:

datetime,close
2023-01-03 09:01:00,4069.0
2023-01-03 09:02:00,4056.0
2023-01-03 09:03:00,4052.0
...

注意事项：
- 第一列必须是 datetime 索引
- 第一行必须是列名（可以是任意名称，值会被使用）
- 不能有多列，每个文件只存储一个特征
""")

print("3. Qlib 重采样机制")
print("-"*60)
print("""
Qlib 支持两种重采样方式：

方式一：预处理重采样（推荐）
- 使用脚本将 1min 数据重采样到其他频率
- 生成对应频率的数据文件
- 优点：速度快，数据一致性好
- 缺点：需要额外存储空间

方式二：运行时重采样
- Qlib 在运行时从 1min 数据动态重采样
- 优点：节省存储空间
- 缺点：速度较慢

重采样规则：
- open:  该时间段的第一个开盘价
- high:  该时间段的最高价
- low:   该时间段的最低价
- close: 该时间段的最后一个收盘价
- volume: 该时间段的成交量之和
- amount: 该时间段的成交额之和
- vwap: 成交额/成交量
- open_interest: 该时间段最后一个持仓量
""")

print("4. 数据加载流程")
print("-"*60)
print("""
用户调用 D.features()
    ↓
Qlib 检查请求的频率（freq）
    ↓
如果该频率的数据存在 → 直接加载
    ↓
如果该频率的数据不存在 → 尝试从更高频率重采样
    ↓
根据 calendars/{freq}.txt 确定时间点
    ↓
加载对应时间段的数据
    ↓
返回 DataFrame
""")

print("\n" + "="*60)
