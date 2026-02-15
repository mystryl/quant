# 数据目录整理说明

**整理时间**: 2026-02-15 21:40

## 数据目录结构

```
data/
├── raw/                          # 原始CSV数据（4个文件）
│   ├── backtest_1min.csv
│   ├── backtest_5min.csv
│   ├── backtest_15min.csv
│   └── backtest_60min.csv
│
├── qlib_data/                    # Qlib数据（按频率组织）
│   ├── calendars/
│   └── instruments/
│       ├── 1min/
│       │   └── RB9999.XSGE.csv
│       └── day/
│           └── RB9999.XSGE.csv
│
├── qlib_data_v2/                 # Qlib数据（按合约和字段组织）
│   ├── calendars/
│   └── instruments/
│       ├── 1min/
│       │   ├── $close/
│       │   │   └── RB9999.XSGE.csv
│       │   ├── $open/
│       │   ├── $high/
│       │   └── ...
│       └── day/
│           └── ...
│
└── qlib_data_multi_freq/         # 多频率数据（15min, 5min, 60min）
    ├── calendars/
    └── instruments/
        ├── 15min/
        │   ├── $close/
        │   │   └── RB9999.XSGE.csv
        │   └── ...
        ├── 5min/
        └── 60min/
```

## 数据使用情况

### 1. `/mnt/d/quant/qlib_data` - 原始Qlib数据（外部）

**位置**: `/mnt/d/quant/qlib_data`
**格式**: 按合约组织，每个合约有独立的字段文件

```
/mnt/d/quant/qlib_data/
├── instruments/
│   ├── RB9999.XSGE/
│   │   ├── open.csv
│   │   ├── close.csv
│   │   ├── high.csv
│   │   ├── low.csv
│   │   ├── volume.csv
│   │   ├── amount.csv
│   │   ├── vwap.csv
│   │   └── open_interest.csv
│   └── all.txt
└── calendars/
```

**使用场景**:
- 主要数据源
- 脚本 `qlib_supertrend_enhanced.py` 用于读取1分钟数据
- 脚本 `prepare_data_v2.py` 用于转换成v2格式

### 2. `data/qlib_data` - Qlib数据（按频率组织）

**格式**: 按频率组织，每个频率下有合约CSV文件

```
data/qlib_data/
├── instruments/
│   ├── 1min/
│   │   └── RB9999.XSGE.csv  (包含所有OHLCV数据)
│   └── day/
│       └── RB9999.XSGE.csv
└── calendars/
```

**使用场景**:
- 用于Qlib初始化
- 脚本 `prepare_data.py` - 准备此格式数据
- 脚本 `debug_qlib.py` - 调试Qlib数据加载
- 脚本 `simple_strategy.py` - 简单策略回测
- 脚本 `check_data_format.py` - 检查数据格式

### 3. `data/qlib_data_v2` - Qlib数据（按合约和字段组织）

**格式**: 和 `/mnt/d/quant/qlib_data` 相同，按合约和字段组织

```
data/qlib_data_v2/
├── instruments/
│   ├── 1min/
│   │   ├── $close/RB9999.XSGE.csv
│   │   ├── $open/RB9999.XSGE.csv
│   │   └── ...
│   └── day/
└── calendars/
```

**使用场景**:
- 从原始数据转换而来
- 脚本 `prepare_data_v2.py` - 准备此格式数据
- 脚本 `debug_qlib_v2.py` - 调试Qlib数据加载

### 4. `data/qlib_data_multi_freq/` - 多频率数据

**格式**: 按频率和字段组织

```
data/qlib_data_multi_freq/
├── instruments/
│   ├── 15min/
│   │   ├── $close/RB9999.XSGE.csv
│   │   └── ...
│   ├── 5min/
│   └── 60min/
```

**使用场景**:
- 脚本 `qlib_supertrend_enhanced.py` 用于读取15分钟、5分钟、60分钟数据
- 脚本 `multi_freq_backtest.py` - 多频率回测

### 5. `data/raw/` - 原始CSV数据

**文件**:
- `backtest_1min.csv` - 1分钟K线数据
- `backtest_5min.csv` - 5分钟K线数据
- `backtest_15min.csv` - 15分钟K线数据
- `backtest_60min.csv` - 60分钟K线数据

**使用场景**:
- 原始数据备份
- 可能用于某些需要CSV格式数据的脚本

## 数据格式对比

| 目录 | 格式 | 使用场景 |
|------|------|---------|
| `/mnt/d/quant/qlib_data` | 按合约组织 | 主要数据源 |
| `data/qlib_data` | 按频率组织 | Qlib标准格式 |
| `data/qlib_data_v2` | 按合约和字段组织 | 转换格式 |
| `data/qlib_data_multi_freq` | 按频率和字段组织 | 多频率数据 |
| `data/raw/` | CSV文件 | 原始备份 |

## 脚本路径更新

以下脚本已更新路径以适配新目录结构：

1. `scripts/strategy/qlib_supertrend_enhanced.py`
   - `RESAMPLED_DIR` 更新为 `/mnt/d/quant/qlib_backtest/data/qlib_data_multi_freq`

2. `scripts/check/check_data_format.py`
   - `data_file` 更新为 `/mnt/d/quant/qlib_backtest/data/qlib_data/instruments/1min/RB9999.XSGE.csv`

3. `scripts/data/prepare_data.py`
   - `QLIB_DIR` 更新为 `/mnt/d/quant/qlib_backtest/data/qlib_data`

4. `scripts/data/prepare_data_v2.py`
   - `QLIB_DIR` 更新为 `/mnt/d/quant/qlib_backtest/data/qlib_data_v2`

5. `scripts/debug/debug_qlib.py`
   - `provider_uri` 更新为 `/mnt/d/quant/qlib_backtest/data/qlib_data`

6. `scripts/debug/debug_qlib_v2.py`
   - `provider_uri` 更新为 `/mnt/d/quant/qlib_backtest/data/qlib_data_v2`

7. `scripts/strategy/simple_strategy.py`
   - `provider_uri` 更新为 `/mnt/d/quant/qlib_backtest/data/qlib_data`

## 数据保留说明

### 需要保留的数据

1. **`/mnt/d/quant/qlib_data`** - 原始Qlib数据（不在qlib_backtest内）
   - 主要数据源，**不能删除**

2. **`data/qlib_data_multi_freq/`** - 多频率数据
   - 回测脚本正在使用，**不能删除**

3. **`data/qlib_data/`** - Qlib数据（按频率组织）
   - 多个脚本在使用，**不能删除**

### 可以删除的数据

**`data/qlib_data_v2/`** - Qlib数据（按合约和字段组织）
- 仅用于 `prepare_data_v2.py` 和 `debug_qlib_v2.py`
- 两个都是调试/准备脚本，不是主要回测脚本
- 可以删除以节省空间

**如果决定删除 `qlib_data_v2`**:
```bash
rm -rf /mnt/d/quant/qlib_backtest/data/qlib_data_v2/
```

同时需要更新/删除相关脚本：
- `scripts/data/prepare_data_v2.py` - 可以保留或删除
- `scripts/debug/debug_qlib_v2.py` - 可以保留或删除

## 建议

### 数据保留方案

**推荐**: 保留所有数据目录

理由：
- `qlib_data_v2` 虽然当前使用不多，但保留了另一种格式数据
- 不同格式的数据可能在特定场景下有用
- 磁盘空间充足的情况下，保留备选格式无妨

### 数据精简方案

如果需要节省空间：
```bash
# 删除 qlib_data_v2
rm -rf /mnt/d/quant/qlib_backtest/data/qlib_data_v2/

# 同时可以删除相关脚本（可选）
rm /mnt/d/quant/qlib_backtest/scripts/data/prepare_data_v2.py
rm /mnt/d/quant/qlib_backtest/scripts/debug/debug_qlib_v2.py
```

注意：删除后如果将来需要v2格式数据，需要重新运行 `prepare_data_v2.py`。
