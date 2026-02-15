# Qlib 回测案例

## 数据说明

- 数据来源: `/mnt/d/quant/qlib_data/`
- 合约: RB9999.XSGE (螺纹钢主连)
- 时间范围: 2023-01-03 ~ 2023-12-31
- 数据频率: 1分钟 K线
- 字段: open, high, low, close, volume, amount, vwap, open_interest

## 使用步骤

### 1. 准备数据

```bash
cd /mnt/d/quant/qlib_backtest
python3 prepare_data.py
```

### 2. 运行回测

```bash
python3 simple_strategy.py
```

## 回测策略

### 简单均值回归策略 (布林带)

- **买入信号**: 价格低于布林带下轨（MA - 2×标准差）
- **卖出信号**: 价格高于布林带上轨（MA + 2×标准差）
- **参数**: 20周期移动平均线，2倍标准差

## 下一步

可以尝试更复杂的策略：
1. 多因子策略
2. 机器学习预测
3. 风险管理优化
4. 多合约组合
