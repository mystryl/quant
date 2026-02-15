# 数据目录清理说明

**整理时间**: 2026-02-15 21:45

## 清理概述

根据用户要求，保留了正在使用的数据目录，删除了重复和未使用的数据。

---

## 数据目录现状

```
data/
├── raw/                          # 原始CSV数据（4个文件）✅ 保留
│   ├── backtest_1min.csv
│   ├── backtest_5min.csv
│   ├── backtest_15min.csv
│   └── backtest_60min.csv
│
├── qlib_data/                    # Qlib数据（按频率组织）✅ 保留
│   ├── calendars/
│   └── instruments/
│       ├── 1min/RB9999.XSGE.csv
│       └── day/RB9999.XSGE.csv
│
└── qlib_data_multi_freq/         # 多频率数据 ✅ 保留
    ├── calendars/
    └── instruments/
        ├── 15min/
        ├── 5min/
        └── 60min/
```

---

## 保留的数据目录

### 1. `/mnt/d/quant/qlib_data` - 原始Qlib数据（外部）

**位置**: `/mnt/d/quant/qlib_data`（不在qlib_backtest目录内）
**格式**: 按合约组织，每个合约有独立的字段文件
**状态**: ✅ **保留** - 主要数据源，不能删除

**使用脚本**:
- `scripts/strategy/qlib_supertrend_enhanced.py` - 读取1分钟数据

### 2. `data/qlib_data/` - Qlib数据（按频率组织）

**格式**: 按频率组织，每个频率下有合约CSV文件
**状态**: ✅ **保留** - 多个脚本正在使用

**使用脚本**:
- `scripts/data/prepare_data.py` - 准备此格式数据
- `scripts/debug/debug_qlib.py` - 调试Qlib数据加载
- `scripts/strategy/simple_strategy.py` - 简单策略回测
- `scripts/check/check_data_format.py` - 检查数据格式

### 3. `data/qlib_data_multi_freq/` - 多频率数据

**格式**: 按频率和字段组织
**状态**: ✅ **保留** - 回测脚本正在使用

**使用脚本**:
- `scripts/strategy/qlib_supertrend_enhanced.py` - 读取15min/5min/60min数据
- `scripts/other/multi_freq_backtest.py` - 多频率回测

### 4. `data/raw/` - 原始CSV数据

**文件**: 4个原始CSV文件
**状态**: ✅ **保留** - 原始数据备份

---

## 删除的数据目录

### `data/qlib_data_v2/` - Qlib数据（按合约和字段组织）

**格式**: 和 `/mnt/d/quant/qlib_data` 相同
**状态**: ❌ **删除** - 仅用于调试脚本

**原使用脚本**:
- `scripts/data/prepare_data_v2.py` - 准备此格式数据（可删除）
- `scripts/debug/debug_qlib_v2.py` - 调试Qlib数据加载（可删除）

**删除原因**:
- 仅用于调试/准备脚本，不是主要回测脚本
- 与 `/mnt/d/quant/qlib_data` 格式重复
- 节省磁盘空间

---

## 相关脚本处理

### 可删除的脚本

以下脚本仅用于已删除的 `qlib_data_v2/` 数据，可以一并删除：

1. `scripts/data/prepare_data_v2.py` - 准备v2格式数据
2. `scripts/debug/debug_qlib_v2.py` - 调试v2格式数据

**如果将来需要v2格式数据**，可以重新运行 `prepare_data_v2.py`（需要先恢复脚本或从文档恢复）。

### 当前保留的脚本

所有主回测脚本、数据处理脚本都已更新路径，正常使用：

- ✅ `scripts/strategy/qlib_supertrend_enhanced.py` - 主回测脚本
- ✅ `scripts/data/prepare_data.py` - 数据准备脚本
- ✅ `scripts/debug/debug_qlib.py` - 调试脚本
- ✅ `scripts/strategy/simple_strategy.py` - 简单策略
- ✅ `scripts/check/check_data_format.py` - 数据检查
- ✅ 其他所有回测和优化脚本

---

## 清理命令

如果需要删除 `qlib_data_v2/`：

```bash
# 删除 qlib_data_v2 目录
rm -rf /mnt/d/quant/qlib_backtest/data/qlib_data_v2/

# 可选：删除相关脚本
rm /mnt/d/quant/qlib_backtest/scripts/data/prepare_data_v2.py
rm /mnt/d/quant/qlib_backtest/scripts/debug/debug_qlib_v2.py
```

---

## 清理后验证

验证数据目录：

```bash
# 查看data目录
ls -la /mnt/d/quant/qlib_backtest/data/

# 应该看到：
# raw/
# qlib_data/
# qlib_data_multi_freq/
# （qlib_data_v2/ 已删除）
```

验证脚本路径：

```bash
# 检查脚本是否还有引用已删除的目录
grep -r "qlib_data_v2" /mnt/d/quant/qlib_backtest/scripts/
```

如果有输出，说明还有脚本引用，需要手动更新或删除。

---

## 数据备份建议

在删除 `qlib_data_v2/` 之前，建议：

1. **确认不需要此格式数据**
   - 检查是否有其他脚本或项目使用此格式
   - 确认可以从 `/mnt/d/quant/qlib_data` 重新生成

2. **可选备份**
   - 如果磁盘空间充足，可以先打包备份：
     ```bash
     cd /mnt/d/quant/qlib_backtest/data
     tar -czf qlib_data_v2_backup.tar.gz qlib_data_v2/
     ```

3. **测试运行**
   - 删除前运行一个回测脚本，确保没有依赖问题

---

## 最终数据目录结构

```
data/
├── raw/                    # 原始CSV数据（4个文件）
├── qlib_data/              # Qlib数据（按频率组织）
└── qlib_data_multi_freq/   # 多频率数据
```

**总大小**: 约 1.5GB（取决于数据量）

**删除的空间**: 取决于 `qlib_data_v2/` 的大小

---

## 总结

✅ **保留的数据**:
- `/mnt/d/quant/qlib_data` - 原始数据源
- `data/qlib_data/` - Qlib标准格式
- `data/qlib_data_multi_freq/` - 多频率数据
- `data/raw/` - 原始CSV备份

❌ **删除的数据**:
- `data/qlib_data_v2/` - 重复格式数据

✅ **更新的脚本**:
- 7个脚本已更新路径
- 所有主回测脚本正常工作

---

## 相关文档

详细的数据目录使用说明：
- **`docs/guides/DATA_ORGANIZATION.md`** - 数据目录详细说明
- **`docs/guides/BACKTEST_REPORTS_ORGANIZATION.md`** - 回测报告整理说明
