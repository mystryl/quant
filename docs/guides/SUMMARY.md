# Qlib 回测案例总结

## 项目信息

- **合约**: RB9999.XSGE（螺纹钢主力连续合约）
- **时间范围**: 2023-01-01 ~ 2023-12-31
- **数据频率**: 1分钟 K线
- **数据量**: 82,770 行（2023年）

## 数据说明

原始数据位于 `/mnt/d/quant/qlib_data/`：

```
qlib_data/
├── calendars/
│   ├── day.txt          # 727 个交易日
│   └── 1min.txt         # 248,535 个时间点
└── instruments/
    ├── all.txt          # 合约列表
    └── RB9999.XSGE/
        ├── open.csv
        ├── high.csv
        ├── low.csv
        ├── close.csv
        ├── volume.csv
        ├── amount.csv
        ├── vwap.csv
        └── open_interest.csv
```

## 回测结果

### 策略：布林带均值回归

**策略逻辑**：
- 买入信号：价格 < 布林带下轨（MA - 2×标准差）
- 卖出信号：价格 > 布林带上轨（MA + 2×标准差）
- 参数：20周期移动平均线，2倍标准差

**回测表现**：
- **总交易次数**: 9,668
- **累计收益**: 92.18%
- **年化收益**: 61.19%
- **最大回撤**: 2.25%
- **夏普比率**: 8.11
- **胜率**: 60.76%

**基准对比**：
- **买入持有收益**: -0.40%

### 结果文件

- 详细数据: `/mnt/d/quant/qlib_backtest/backtest_results_direct.csv`
- 收益曲线: `/mnt/d/quant/qlib_backtest/backtest_curve.png`

## 实现说明

### 为什么不直接使用 qlib 数据加载器？

qlib 的数据格式要求比较严格，需要特定的目录结构和文件格式。经过多次尝试，发现：

1. qlib 期望的格式：`instruments/freq/feature_name/instrument.csv`
2. 每个合约的每个特征需要单独一个文件
3. 即使按照格式准备，数据加载仍然为空

**解决方案**：直接使用 pandas 读取原始 CSV 数据，进行回测。这种方式更灵活、更直观。

### 回测脚本

`/mnt/d/quant/qlib_backtest/backtest_direct.py`

**特点**：
- 直接读取原始 CSV 数据
- 纯 Python 实现，不依赖复杂框架
- 易于理解和修改
- 自动生成回测报告和图表

## 使用方法

### 运行回测

```bash
cd /mnt/d/quant/qlib_backtest
python3 backtest_direct.py
```

### 查看结果

```bash
# 查看详细数据
head -20 backtest_results_direct.csv

# 查看收益曲线
eog backtest_curve.png  # 或使用其他图片查看器
```

## 下一步建议

### 1. 优化策略

- 尝试不同的技术指标（RSI、MACD、KDJ等）
- 优化布林带参数（周期、标准差倍数）
- 加入止损止盈机制
- 增加过滤条件（如趋势确认、成交量确认）

### 2. 风险管理

- 设置最大持仓时间
- 限制单笔交易金额
- 加入仓位管理

### 3. 多合约回测

- 添加更多合约进行组合回测
- 研究合约间相关性
- 分散投资降低风险

### 4. 更长时间范围

- 扩展到2024年数据
- 测试策略在不同市场环境下的表现

## 技术栈

- Python 3.12
- pandas: 数据处理
- numpy: 数值计算
- matplotlib: 可视化

## 注意事项

⚠️ **重要提示**：
1. 这是简单的技术指标策略，仅供学习参考
2. 回测结果不代表未来表现
3. 实盘交易需要考虑滑点、手续费等成本
4. 建议进行充分的风险评估后再做投资决策

## 文件清单

```
/mnt/d/quant/qlib_backtest/
├── prepare_data.py              # 数据转换脚本（qlib格式）
├── prepare_data_v2.py           # 数据转换脚本v2（qlib格式）
├── simple_strategy.py           # qlib策略（未完成）
├── backtest_direct.py           # ✅ 直接回测脚本（推荐）
├── debug_qlib.py                # qlib调试脚本
├── debug_qlib_v2.py            # qlib调试脚本v2
├── check_data_format.py         # 数据格式检查
├── backtest_results_direct.csv  # 回测结果数据
├── backtest_curve.png           # 收益曲线图
├── README.md                     # 原始说明
└── SUMMARY.md                    # 本文件
```

---

**创建时间**: 2025-02-15  
**最后更新**: 2025-02-15
