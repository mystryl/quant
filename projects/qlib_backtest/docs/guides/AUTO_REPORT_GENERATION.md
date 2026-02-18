# 回测报告自动化说明

**更新时间**: 2026-02-15 21:55

---

## ✅ 自动报告生成器已创建

现在每次回测都会**自动生成结构化报告**，无需手动整理！

---

## 🚀 快速开始

### 方法1：使用便捷函数

在你的回测脚本末尾添加：

```python
from other.report_generator import create_report

# 运行回测
results = run_backtest(df, strategy_name)
params = {'period': 50, 'multiplier': 20, 'n': 3}

# 自动生成报告
report_dir = create_report(
    strategy_name=strategy_name,
    params=params,
    results=results
)

print(f"报告已生成到: {report_dir}")
```

### 方法2：使用模板脚本

复制 `scripts/backtest/backtest_with_auto_report.py` 作为你的回测脚本模板。

---

## 📁 报告目录结构

每次回测会自动生成：

```
backtest_reports/
└── 20260215_2154_SuperTrend_SF14Re_period50_multiplier20_n3/
    ├── README.md           # 参数和配置说明
    ├── SUMMARY.md          # 结果摘要和结论
    ├── results/            # 结果文件
    │   ├── backtest_results.csv  # 交易明细
    │   └── metrics.json           # 性能指标
    ├── code/               # 源代码备份
    │   └── backtest_script.py
    └── charts/             # 图表文件
        ├── equity_curve.png
        └── drawdown_chart.png
```

---

## 🎯 文件夹命名规则

自动生成，格式：

```
{YYYYMMDD}_{HHMM}_{strategy}_{key_params}
```

**示例**：
- `20260215_2154_SuperTrend_SF14Re_period50_multiplier20_n3`
- `20260215_2200_Optuna_Optimization_sharpe_100trials`

---

## 📖 修改现有回测脚本

### 3步集成自动报告生成

#### 步骤1：导入报告生成器

```python
import sys
from pathlib import Path

# 添加脚本目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

# 导入报告生成器
from other.report_generator import create_report
```

#### 步骤2：准备配置信息

```python
# 数据配置
data_config = {
    '数据来源': 'Qlib数据',
    '频率': freq,
    '年份': f"{start_date[:4]}-{end_date[:4]}",
    '合约': 'RB9999.XSGE',
    '数据长度': f"{len(df)} 根K线"
}

# 回测配置
backtest_config = {
    '初始资金': '1,000,000 CNY',
    '交易手续费': '万分之一',
    '滑点': '1跳'
}
```

#### 步骤3：生成报告

```python
# 生成报告
report_dir = create_report(
    strategy_name=results['strategy_name'],
    params={
        'period': period,
        'multiplier': multiplier,
        'n': n,
        'freq': freq
    },
    results=results,
    results_df=df_strategy,
    data_config=data_config,
    backtest_config=backtest_config,
    source_file=__file__
)

print(f"✅ 报告已生成到: {report_dir}")
```

---

## 📚 完整文档

详细的使用指南请参阅：
**`docs/guides/REPORT_GENERATOR_GUIDE.md`**

包含：
- 完整API参考
- 代码示例
- 常见问题
- 最佳实践

---

## 📦 已创建的文件

### 核心脚本

1. **`scripts/other/report_generator.py`**
   - 报告生成器核心模块
   - 自动生成文件夹、README、SUMMARY、指标JSON

2. **`scripts/backtest/backtest_with_auto_report.py`**
   - 回测脚本模板
   - 可直接复制使用

### 文档

3. **`docs/guides/REPORT_GENERATOR_GUIDE.md`**
   - 完整使用指南
   - API参考和示例

---

## ✨ 自动报告生成的优势

### 手动整理 vs 自动生成

| 对比项 | 手动整理 | 自动生成 |
|--------|---------|---------|
| 文件夹命名 | 手动输入，容易出错 | 自动生成，规范统一 |
| 目录结构 | 手动创建 | 自动创建 |
| README.md | 手动编写 | 自动生成 |
| SUMMARY.md | 手动编写 | 自动生成 |
| metrics.json | 手动编写 | 自动生成 |
| 代码备份 | 手动复制 | 自动复制 |
| 图表保存 | 手动保存 | 自动保存 |
| 时间成本 | 5-10分钟 | **0分钟** |

---

## 🎯 回测报告规范

### 必需字段

`results` 字典必须包含：

```python
results = {
    'strategy_name': '策略名称',
    'total_trades': 总交易次数,
    'cumulative_return': 累计收益,
    'annual_return': 年化收益,
    'max_drawdown': 最大回撤,
    'sharpe_ratio': 夏普比率,
    'win_rate': 胜率,
    'buy_hold_return': 买入持有收益,
    'stopped_out_count': 止损次数（可选）
}
```

### 可选字段

- `data_config` - 数据配置
- `backtest_config` - 回测配置
- `benchmark_results` - 基准对比
- `source_file` - 源代码备份
- `charts` - 图表文件

---

## 📊 示例输出

运行回测后，会看到：

```
==========================================================================
✅ 回测完成！
==========================================================================
报告目录: /mnt/d/quant/qlib_backtest/backtest_reports/20260215_2154_SuperTrend_SF14Re_period50_multiplier20_n3

目录结构:
  /mnt/d/quant/qlib_backtest/backtest_reports/20260215_2154_SuperTrend_SF14Re_period50_multiplier20_n3/
    README.md      - 参数和配置说明
    SUMMARY.md     - 结果摘要和结论
    results/
      *.csv       - 回测结果CSV
      metrics.json - 性能指标JSON
    code/
      *.py        - 源代码备份
    charts/
      *.png       - 图表文件
```

---

## 🔧 已更新的脚本

### 已更新路径的脚本

以下脚本已更新数据路径，可以直接使用：

- ✅ `scripts/strategy/qlib_supertrend_enhanced.py`
- ✅ `scripts/data/prepare_data.py`
- ✅ `scripts/debug/debug_qlib.py`
- ✅ `scripts/strategy/simple_strategy.py`
- ✅ `scripts/check/check_data_format.py`

### 待集成自动报告的脚本

以下脚本可以集成自动报告生成：

- 📝 `scripts/strategy/qlib_supertrend_enhanced.py` - 推荐优先集成
- 📝 `scripts/strategy/simple_strategy.py`
- 📝 其他回测脚本

---

## 🎯 下一步行动

### 推荐操作

1. **修改主回测脚本**
   - 修改 `scripts/strategy/qlib_supertrend_enhanced.py`
   - 集成自动报告生成
   - 测试运行一次

2. **运行一次测试**
   - 验证报告生成正常
   - 检查目录结构
   - 确认文件内容正确

3. **应用到其他脚本**
   - 修改其他回测脚本
   - 统一使用报告生成器
   - 保持报告格式一致

---

## 📖 文档导航

| 文档 | 说明 |
|------|------|
| `REPORT_GENERATOR_GUIDE.md` | 完整使用指南 |
| `BACKTEST_REPORTS_ORGANIZATION.md` | 回测报告整理说明 |
| `DATA_ORGANIZATION.md` | 数据目录说明 |
| `DATA_CLEANUP.md` | 数据清理说明 |

---

## ✅ 总结

**已创建**：
- ✅ 报告生成器模块
- ✅ 回测脚本模板
- ✅ 完整使用指南

**功能**：
- ✅ 自动生成文件夹
- ✅ 自动生成README.md
- ✅ 自动生成SUMMARY.md
- ✅ 自动保存CSV结果
- ✅ 自动保存JSON指标
- ✅ 自动备份源代码
- ✅ 自动保存图表

**优势**：
- ✅ 节省时间（每次5-10分钟 → 0分钟）
- ✅ 规范统一
- ✅ 易于查找和对比
- ✅ 便于版本控制

---

**状态**: ✅ 已完成
**更新时间**: 2026-02-15 21:55
