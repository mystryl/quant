# AKQuant SuperTrend 回测项目

## 项目简介

使用 AKQuant 框架对 SuperTrend 趋势跟踪策略进行回测，生成可视化报告。

## 配置

- **品种**：RB9999（螺纹钢期货）
- **时间**：2024 年
- **频率**：15 分钟、60 分钟
- **参数组合**：
  - (7, 3.0)
  - (10, 3.0)
  - (14, 2.5)
- **初始资金**：100,000
- **佣金**：0.0001（万分之一）

## 目录结构

```
akquant_supertrend/
├── prepare_akquant_data.py    # 数据准备脚本
├── supertrend_strategy.py     # SuperTrend 策略实现
├── backtest_supertrend.py     # 回测脚本
├── README.md                 # 本文件
├── data/                    # 数据文件
│   ├── RB9999_2024_15min.csv
│   └── RB9999_2024_60min.csv
└── reports/                 # 回测报告
    ├── backtest_summary.csv
    ├── RB9999_supertrend_15min_p7_m3.0.html
    ├── RB9999_supertrend_15min_p10_m3.0.html
    ├── RB9999_supertrend_15min_p14_m2.5.html
    ├── RB9999_supertrend_60min_p7_m3.0.html
    ├── RB9999_supertrend_60min_p10_m3.0.html
    └── RB9999_supertrend_60min_p14_m2.5.html
```

## 快速开始

### 1. 准备数据

```bash
python3 prepare_akquant_data.py
```

### 2. 运行回测

```bash
python3 backtest_supertrend.py
```

### 3. 查看报告

在浏览器中打开 `reports/` 目录下的 HTML 文件。

## 回测结果

### 60 分钟级别

| 参数 | 收益率 | 夏普 | 最大回撤 | 胜率 | 交易次数 |
|------|---------|------|----------|------|---------|
| (7, 3.0) | -2.17% | -7.10 | 2.17% | 43.84% | 219 |
| (10, 3.0) | -2.44% | -7.32 | 2.44% | 42.80% | 236 |
| (14, 2.5) | -2.78% | -6.92 | 2.78% | 44.13% | 247 |

### 15 分钟级别

| 参数 | 收益率 | 夏普 | 最大回撤 | 胜率 | 交易次数 |
|------|---------|------|----------|------|---------|
| (7, 3.0) | -3.96% | -12.43 | 3.96% | 43.79% | 427 |
| (10, 3.0) | -4.99% | -11.65 | 4.99% | 40.00% | 505 |
| (14, 2.5) | -5.57% | -12.21 | 5.57% | 42.65% | 565 |

## 结论

1. **所有参数组合都产生负收益**，这与之前 Qlib 回测结果一致
2. **60 分钟级别表现优于 15 分钟级别**，损失较小
3. **60 分钟 (7, 3.0) 组合表现最好**：-2.17% 收益，夏普 -7.10
4. **高频交易导致更多亏损**：15 分钟级别交易次数远高于 60 分钟

## 技术要点

### SuperTrend 指标计算

使用纯 NumPy 实现 SuperTrend 指标，包括：
- ATR (Average True Range) 计算
- 基本上下轨计算
- 最终上下轨计算（考虑前一日状态）
- 趋势方向判断

### 策略逻辑

```python
# 当收盘价 > SuperTrend 时，做多
if close > supertrend and last_close <= supertrend:
    buy()

# 当收盘价 < SuperTrend 时，平多仓
if close < supertrend and last_close >= supertrend:
    close_position()
```

### AKQuant 特性

- **快速回测**：Rust 引擎，性能优于纯 Python 框架
- **可视化报告**：一键生成包含权益曲线、回撤、交易分析的完整 HTML 报告
- **自动进度显示**：实时显示回测进度
- **多参数支持**：支持批量运行不同参数组合

## 报告内容

每个回测生成的 HTML 报告包含：

- 📋 核心指标概览（12个指标）
  - 总收益率、年化收益率、夏普比率
  - 最大回撤、波动率、胜率
  - 盈亏比、凯利公式
- 📈 权益曲线与回撤
- 📅 月度收益热力图
- 📊 收益分布分析
- 🔄 滚动指标（夏普、回撤）
- 💹 交易盈亏分布
- ⏱️ 盈亏 vs 持仓时间

## 下一步建议

1. 尝试不同的参数范围（更大的 period，不同的 multiplier）
2. 考虑结合其他指标过滤信号（如成交量、趋势强度）
3. 探索止损止盈机制
4. 对比其他品种的表现
5. 尝试不同的市场环境（牛熊市）

## 参考资料

- [AKQuant 官方文档](https://akquant.akfamily.xyz/)
- [AKQuant GitHub](https://github.com/akfamily/akquant)
- [SuperTrend 指标原理](https://www.tradingview.com/script/P5QPu9e/)

## 许可

本项目仅供学习和研究使用。
