# 文件结构整理报告

**整理时间**: 2026-02-15 21:35

## 整理概述

将 `/mnt/d/quant/qlib_backtest/` 目录下的文件从混乱状态整理为清晰的目录结构。

---

## 新的目录结构

```
/mnt/d/quant/qlib_backtest/
├── scripts/                    # 所有脚本（27个文件）
│   ├── backtest/              # 回测脚本（6个）
│   ├── strategy/              # 策略脚本（3个）
│   ├── optimize/              # 优化脚本（3个）
│   ├── test/                  # 测试脚本（4个）
│   ├── data/                  # 数据处理脚本（3个）
│   ├── debug/                 # 调试脚本（2个）
│   ├── check/                 # 检查脚本（2个）
│   └── other/                 # 其他脚本（4个）
│
├── docs/                       # 文档（20个文件）
│   ├── guides/                # 使用指南（8个）
│   ├── strategies/            # 策略文档（3个）
│   ├── archive/               # 历史报告归档（9个）
│   └── tools/                 # 工具文档（空）
│
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据（4个CSV）
│   ├── qlib_data/             # qlib数据（已移动）
│   ├── qlib_data_multi_freq/  # 多频率数据（已移动）
│   └── qlib_data_v2/          # qlib数据v2（已移动）
│
├── results/                    # 历史回测结果（10个文件）
│   ├── old/                   # 旧回测结果（8个CSV）
│   └── charts/                # 图表（2个PNG）
│
├── backtest_reports/           # 【新建】未来回测报告根目录
├── memory/                     # 记忆目录（保留）
└── __pycache__/               # Python缓存（保留）
```

---

## 文件移动统计

| 类型 | 原有数量 | 整理后 | 说明 |
|------|---------|--------|------|
| 脚本文件 | 27个 | scripts/ | 按功能分类到8个子目录 |
| 文档文件 | 20个 | docs/ | 按类型分类到3个子目录 |
| 原始数据 | 4个 | data/raw/ | 回测数据CSV |
| 回测结果 | 8个 | results/old/ | 历史回测结果CSV |
| 图表 | 2个 | results/charts/ | PNG图表 |
| 数据目录 | 3个 | data/ | qlib相关目录 |

---

## 主要改进

### 1. 脚本分类明确
- 按功能将27个脚本分为8个类别
- 每个类别有独立的子目录
- 便于查找和管理

### 2. 文档结构清晰
- 使用指南集中到 guides/
- 策略文档集中到 strategies/
- 历史报告归档到 archive/
- 避免文档混淆

### 3. 数据统一管理
- 所有数据集中在 data/ 目录
- 原始数据和qlib数据分开存储
- 便于数据管理和备份

### 4. 结果分离归档
- 历史回测结果归档到 results/old/
- 图表文件单独存储
- 避免与未来回测结果混淆

### 5. 未来回测报告
- 创建 backtest_reports/ 作为新回测报告根目录
- 每次回测生成独立文件夹
- 便于追踪和对比

---

## 使用建议

### 运行脚本
```bash
# 回测脚本
python scripts/backtest/qlib_supertrend_enhanced.py

# 优化脚本
python scripts/optimize/optimize_supertrend_optuna.py

# 测试脚本
python scripts/test/test_supertrend_simple.py
```

### 查看文档
```bash
# 使用指南
cat docs/guides/SUPERTREND_USAGE.md

# 策略文档
cat docs/strategies/QLIB_STRATEGIES_EXPLAINED.md

# 历史报告
cat docs/archive/SUPERTRENT_ENHANCED_OPTIMIZATION_REPORT.md
```

### 访问数据
```bash
# 原始数据
ls data/raw/

# Qlib数据
ls data/qlib_data/
ls data/qlib_data_multi_freq/
ls data/qlib_data_v2/
```

### 查看历史结果
```bash
# 回测结果
ls results/old/

# 图表
ls results/charts/
```

---

## 后续工作

### 1. 创建报告生成器
创建 `scripts/other/report_generator.py` 用于：
- 自动生成回测报告文件夹
- 文件夹命名：`{YYYYMMDD}_{HHMM}_{strategy}_{key_params}`
- 创建标准目录结构（README.md, SUMMARY.md, results/, code/）
- 保存参数、结果、图表

### 2. 调整现有脚本
修改回测脚本使其：
- 自动调用报告生成器
- 生成结构化的回测报告
- 避免覆盖历史结果

### 3. 数据备份
考虑定期备份：
- data/ 目录
- backtest_reports/ 目录
- 关键配置文件

---

## 整理完成 ✓

所有文件已成功整理到新的目录结构中。
根目录现在干净整洁，便于日常使用。
