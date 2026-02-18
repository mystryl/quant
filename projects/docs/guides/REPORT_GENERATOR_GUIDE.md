# 自动报告生成器使用指南

**版本**: 1.0
**创建时间**: 2026-02-15

---

## 📋 概述

回测报告生成器可以自动将回测结果整理到标准的目录结构中，生成：
- ✅ README.md - 参数和配置说明
- ✅ SUMMARY.md - 结果摘要和结论
- ✅ results/ - CSV结果和JSON指标
- ✅ code/ - 源代码备份
- ✅ charts/ - 图表文件

---

## 🚀 快速开始

### 1. 基本用法

在你的回测脚本末尾添加：

```python
from report_generator import create_report

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

### 2. 完整用法

```python
from report_generator import create_report
import matplotlib.pyplot as plt

# 运行回测
results = run_backtest(df, strategy_name)
params = {'period': 50, 'multiplier': 20, 'n': 3}

# 准备数据配置
data_config = {
    '数据来源': 'Qlib数据',
    '频率': '15min',
    '年份': '2023',
    '合约': 'RB9999.XSGE',
    '数据长度': '6,480 根K线'
}

# 准备回测配置
backtest_config = {
    '初始资金': '1,000,000 CNY',
    '交易手续费': '万分之一',
    '滑点': '1跳'
}

# 生成图表
fig1, ax1 = plt.subplots()
# ... 绘制图表 ...

fig2, ax2 = plt.subplots()
# ... 绘制图表 ...

# 自动生成报告（包含所有内容）
report_dir = create_report(
    strategy_name=strategy_name,
    params=params,
    results=results,
    results_df=df_with_signals,  # 包含信号的DataFrame
    data_config=data_config,
    backtest_config=backtest_config,
    source_file=__file__,  # 备份源代码
    charts={
        'equity_curve.png': fig1,
        'drawdown_chart.png': fig2
    }
)

print(f"报告已生成到: {report_dir}")
```

---

## 📁 报告目录结构

生成的报告遵循标准结构：

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

文件夹名称自动生成，格式：

```
{YYYYMMDD}_{HHMM}_{strategy}_{key_params}
```

### 示例

```
20260215_2154_SuperTrend_SF14Re_period50_multiplier20_n3
20260215_2200_Optuna_Optimization_sharpe_100trials
20260215_2215_SuperTrend_Enhanced_period10_multiplier3
```

### 关键参数提取规则

自动从参数字典中提取最重要的2-3个参数：
- `period` - ATR周期
- `multiplier` - ATR倍数
- `n` - 突破确认系数
- `freq` - 频率
- `year` - 年份
- `trials` - 试验次数（用于优化）

---

## 📊 结果字典格式

`results` 字典必须包含以下字段：

```python
results = {
    'strategy_name': 'SuperTrend_SF14Re',      # 策略名称
    'total_trades': 5,                         # 总交易次数
    'cumulative_return': 0.12345,              # 累计收益（小数）
    'annual_return': 0.67890,                  # 年化收益（小数）
    'max_drawdown': 0.09876,                  # 最大回撤（小数）
    'sharpe_ratio': 1.23,                      # 夏普比率
    'win_rate': 52.5,                          # 胜率（百分比）
    'buy_hold_return': 0.05678,                # 买入持有收益（小数）
    'stopped_out_count': 3                      # 止损次数（可选）
}
```

---

## 📝 参数字典格式

`params` 字典包含策略的所有参数：

```python
params = {
    'period': 50,                    # ATR周期
    'multiplier': 20,                # ATR倍数
    'n': 3,                         # 突破确认系数
    'trailing_stop_rate': 80,       # 跟踪止损率
    'max_holding_period': 100,      # 最大持仓周期
    'freq': '15min',                # 数据频率
    'year': 2023                    # 测试年份
}
```

---

## 🔧 修改现有回测脚本

### 步骤1：导入报告生成器

在脚本开头添加：

```python
import sys
from pathlib import Path

# 添加脚本目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

# 导入报告生成器
from other.report_generator import create_report
```

### 步骤2：准备配置信息

在回测完成后，添加配置：

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

### 步骤3：生成报告

在脚本末尾，替换原来的打印和保存逻辑：

```python
# 生成报告
report_dir = create_report(
    strategy_name=results['strategy_name'],
    params={
        'period': period,
        'multiplier': multiplier,
        'n': n,
        'trailing_stop_rate': trailing_stop_rate,
        'freq': freq
    },
    results=results,
    results_df=df_strategy,
    data_config=data_config,
    backtest_config=backtest_config,
    source_file=__file__
)

print(f"\n{'='*60}")
print("✅ 回测完成！")
print(f"{'='*60}")
print(f"报告目录: {report_dir}")
```

---

## 📖 完整示例

### 示例：修改后的回测脚本

```python
#!/usr/bin/env python3
"""
SuperTrend 回测脚本（带自动报告生成）
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加脚本目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

# 导入报告生成器
from other.report_generator import create_report

# 你的策略函数
def generate_signal(df, period, multiplier, n):
    # ... 生成信号逻辑 ...
    return df

# 你的回测函数
def run_backtest(df, strategy_name):
    # ... 回测逻辑 ...
    results = {
        'strategy_name': strategy_name,
        'total_trades': 10,
        'cumulative_return': 0.15,
        'annual_return': 0.8,
        'max_drawdown': 0.1,
        'sharpe_ratio': 1.5,
        'win_rate': 55,
        'buy_hold_return': 0.05,
        'stopped_out_count': 3
    }
    return results

# 主函数
def main():
    # 参数设置
    period = 50
    multiplier = 20
    n = 3
    freq = '15min'
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    # 加载数据
    df = load_data(freq, start_date, end_date)

    # 生成信号
    df = generate_signal(df, period, multiplier, n)

    # 运行回测
    results = run_backtest(df, 'SuperTrend_SF14Re')

    # 准备配置
    data_config = {
        '数据来源': 'Qlib数据',
        '频率': freq,
        '年份': f"{start_date[:4]}-{end_date[:4]}",
        '合约': 'RB9999.XSGE',
        '数据长度': f"{len(df)} 根K线"
    }

    backtest_config = {
        '初始资金': '1,000,000 CNY',
        '交易手续费': '万分之一',
        '滑点': '1跳'
    }

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
        results_df=df,
        data_config=data_config,
        backtest_config=backtest_config,
        source_file=__file__
    )

    print(f"\n{'='*60}")
    print("✅ 回测完成！")
    print(f"{'='*60}")
    print(f"报告目录: {report_dir}")

if __name__ == "__main__":
    main()
```

---

## 🎨 生成图表

如果需要保存图表，使用 `charts` 参数：

```python
import matplotlib.pyplot as plt

# 生成图表
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df.index, df['cumulative_returns'])
ax1.set_title('资金曲线')
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df.index, df['drawdown'])
ax2.set_title('回撤曲线')
fig2.tight_layout()

# 生成报告（包含图表）
report_dir = create_report(
    strategy_name=results['strategy_name'],
    params=params,
    results=results,
    charts={
        'equity_curve.png': fig1,
        'drawdown_chart.png': fig2
    }
)

# 关闭图表
plt.close('all')
```

---

## 📚 API 参考

### create_report()

完整参数：

```python
create_report(
    strategy_name: str,           # 策略名称
    params: dict,                  # 参数字典
    results: dict,                 # 回测结果字典
    results_df: pd.DataFrame = None,  # 回测结果DataFrame（可选）
    data_config: dict = None,      # 数据配置（可选）
    backtest_config: dict = None,  # 回测配置（可选）
    benchmark_results: dict = None,  # 基准结果（可选）
    source_file: str = None,       # 源代码文件路径（可选）
    charts: dict = None,           # 图表字典 {filename: fig}（可选）
    base_dir: str = None           # 报告根目录（可选）
) -> Path
```

**返回值**：
- `Path` - 报告目录路径

---

## ✅ 检查清单

使用报告生成器前，检查：

- [ ] `results` 字典包含所有必需字段
- [ ] `params` 字典包含策略参数
- [ ] `strategy_name` 参数正确
- [ ] 如需保存图表，已创建 matplotlib 图表对象
- [ ] 如需备份代码，`source_file` 设置为 `__file__`

---

## 🐛 常见问题

### 1. 导入错误

**问题**：`ModuleNotFoundError: No module named 'other.report_generator'`

**解决**：确保添加脚本目录到路径：
```python
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))
from other.report_generator import create_report
```

### 2. 文件夹名称包含特殊字符

**问题**：策略名称包含空格或特殊字符

**解决**：报告生成器会自动处理，但建议使用简洁的名称：
```python
strategy_name = "SuperTrend_SF14Re"  # ✅ 好
strategy_name = "SuperTrend (SF14Re)"  # ❌ 避免括号
```

### 3. 图表未保存

**问题**：传递了图表但未保存

**解决**：确保图表文件名以 `.png` 结尾：
```python
charts = {
    'equity_curve.png': fig,  # ✅ 正确
    'equity_curve': fig       # ❌ 缺少.png
}
```

---

## 📚 相关文档

- **BACKTEST_REPORTS_ORGANIZATION.md** - 回测报告整理说明
- **DATA_ORGANIZATION.md** - 数据目录说明
- **DATA_CLEANUP.md** - 数据清理说明

---

## 💡 最佳实践

1. **每次回测都生成报告**
   - 保持历史记录
   - 便于对比分析

2. **使用有意义的参数名称**
   - 便于识别和搜索
   - 避免歧义

3. **包含数据配置**
   - 明确数据来源
   - 记录数据长度

4. **备份源代码**
   - 便于复现结果
   - 版本控制

5. **定期整理报告**
   - 删除过时的报告
   - 保留重要的对比报告

---

**更新时间**: 2026-02-15
