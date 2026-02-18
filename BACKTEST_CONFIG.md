# 回测配置标准

## 概述

本项目中的**所有回测必须考虑真实交易环境**，包括但不限于以下要素。这些参数对回测结果的真实性和可靠性至关重要。

---

## 必需的交易成本参数

### 1. 交易手续费 (Transaction Fees)
- **开仓手续费**: 按成交金额的百分比计算
- **平仓手续费**: 按成交金额的百分比计算
- **印花税**: 仅平仓时收取（中国股市标准为 0.1%）

**常见费率参考**：
- 股票：万分之一点五至万分之三（0.015% - 0.03%）
- 期货：成交金额的万分之一至万分之三
- 加密货币：0.04% - 0.1% (Maker/Taker 不同)

### 2. 滑点 (Slippage)
滑点是指实际成交价格与预期价格之间的差异。

**滑点设置建议**：
- 股票：1-2 个跳动点（tick）
- 期货：1-3 个跳动点
- 加密货币：0.01% - 0.05%
- 高频交易：需要更精确的滑点模型

**滑点类型**：
- 固定滑点：每次交易固定点数
- 百分比滑点：按价格百分比
- 动态滑点：根据订单量和市场深度

### 3. 仓位控制 (Position Control)

#### 开仓比例 (Position Sizing)
- 单笔开仓金额占总资金的百分比
- 建议：5% - 30%（根据策略风险度）
- 可使用固定金额、固定比例、风险平价等方法

#### 最大持仓比例 (Max Position Ratio)
- 单个标的的最大持仓限制
- 全部持仓的最大限制
- 建议：单股不超过 20%，总体不超过 80%

#### 保证金 (Margin)
- 杠杆倍数
- 保证金比例
- 追保线和平仓线设置

**期货/加密货币保证金示例**：
- 维持保证金：5% - 10%
- 初始保证金：10% - 15%
- 强平风险阈值：保证金低于维持保证金的 50%

---

## 回测框架配置示例

### Qlib 配置

```python
from qlib.contrib.evaluate import risk_analysis
from qlib.backtest import backtest, executor

# 交易成本配置
trade_cost = {
    "buy_cost": 0.0003,      # 买入手续费 0.03%
    "sell_cost": 0.0003,     # 卖出手续费 0.03%
    "stamp_duty": 0.001,     # 印花税 0.1%
    "slippage": 0.0001,      # 滑点 0.01%
}

# 仓位控制配置
position_config = {
    "max_position_ratio": 0.8,     # 最大持仓比例 80%
    "single_position_ratio": 0.2,  # 单只股票最大 20%
    "initial_capital": 1000000,    # 初始资金
    "leverage": 1.0,               # 杠杆倍数
}
```

### Backtrader 配置

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    def __init__(self):
        # 设置手续费
        self.commission = bt.analyzers.Commission(
            commission=0.0003,      # 0.03% 手续费
            mult=1.0,
            margin=0.1,             # 保证金 10%
        )

    # 仓位控制
    def order_size(self, price):
        # 单次开仓不超过总资金的 20%
        max_position = self.broker.getvalue() * 0.2
        size = int(max_position / price)
        return size
```

---

## 回测报告必需指标

所有回测报告必须包含以下经过成本调整后的指标：

### 收益指标
- 总收益率
- 年化收益率
- 累计收益曲线

### 风险指标
- 最大回撤 (Maximum Drawdown)
- 夏普比率 (Sharpe Ratio)
- 卡尔玛比率 (Calmar Ratio)
- 波动率 (Volatility)

### 交易统计
- 总交易次数
- 胜率 (Win Rate)
- 盈亏比 (Profit/Loss Ratio)
- 平均持仓时间
- 总手续费支出
- 总滑点成本

### 仓位分析
- 平均仓位使用率
- 最大仓位使用率
- 仓位分布图

---

## 检查清单

每次回测前确认：

- [ ] 已设置开仓手续费
- [ ] 已设置平仓手续费
- [ ] 已设置印花税（如适用）
- [ ] 已配置滑点模型
- [ ] 已设置最大持仓比例
- [ ] 已设置单笔开仓比例
- [ ] 已配置保证金参数（如使用杠杆）
- [ ] 已计算总交易成本
- [ ] 报告包含风险调整后收益指标

---

## 常见错误

❌ **不使用交易成本**
导致虚高的回测收益，实盘无法复制。

❌ **忽略滑点**
高频交易中滑点可能吞噬大部分利润。

❌ **满仓交易**
回测可能表现良好，但实盘风险过大，容易爆仓。

❌ **不考虑保证金**
使用杠杆时忽略保证金要求，导致实际无法执行。

---

## 参考资源

- [Qlib 交易成本配置](https://qlib.readthedocs.io/en/latest/component/backtest.html)
- [Backtrader 手续费设置](https://www.backtrader.com/docu/commission/)
- [真实交易成本研究](https://www.sciencedirect.com/science/article/pii/S0378426614000234)

---

**重要提示**：未考虑真实交易成本的回测结果是毫无意义的。所有策略开发必须从一开始就将这些因素纳入模型。
