# 多时间框架趋势策略 - 完整方案

**版本**：1.0
**创建日期**：2026-02-19
**框架**：Qlib + Python
**目标**：商品指数合约中长期趋势跟踪策略

---

## 一、策略核心思想

**多时间框架趋势一致性交易**：
- 大周期（日线/4小时）：判断主趋势方向
- 中周期（1小时）：寻找入场时机
- 小周期（15分钟）：精确入场点

**核心逻辑**：
1. 大周期EMA交叉 + VWAP确认 = 趋势方向
2. 中周期寻找回调/反弹机会
3. 小级别在支撑/阻力位入场
4. Supertrend动态止损 + 固定ATR止盈

---

## 二、大周期趋势判断（趋势过滤器）

### 2.1 EMA交叉系统（主信号）

**配置**：
- EMA短周期：17期
- EMA长周期：31期
- 参考：TradingView主流EMA黄金交叉

**信号定义**：
```
看涨趋势：EMA(17) > EMA(31)
看跌趋势：EMA(17) < EMA(31)
趋势确认：交叉发生并稳定（至少3根K线保持方向）
```

**过滤条件**：
- 交叉后必须保持3根K线以上（避免假交叉）
- 交叉角度 > 30度（避免震荡市场）

### 2.2 VWAP趋势确认（辅助过滤）

**VWAP参数**：
- 会话内计算（默认：4小时日线用20根K线会话）
- 品量加权平均价格

**确认逻辑**：
```
看涨确认条件：
- 价格 > VWAP
- 价格 > EMA(31)
- EMA(17) > EMA(31)

看跌确认条件：
- 价格 < VWAP
- 价格 < EMA(31)
- EMA(17) < EMA(31)
```

**VWAP作用**：
- 过滤弱势趋势（如EMA交叉但价格在VWAP下方）
- 提供动态支撑/阻力位

### 2.3 大周期趋势状态机

```python
TrendState = {
    'direction': 'bullish' | 'bearish' | 'neutral',
    'strength': 'strong' | 'weak' | 'none',
    'since': datetime,
    'ema_17': float,
    'ema_31': float,
    'vwap': float,
    'price': float,
    'is_confirmed': bool
}
```

**状态转换规则**：
```
中性 → 看涨：
  - EMA(17)上穿EMA(31)
  - 价格 > VWAP
  - 持续3根K线确认

看涨 → 看跌：
  - EMA(17)下穿EMA(31)
  - 价格 < VWAP
  - 持续3根K线确认

看涨/看跌 → 中性：
  - EMA接近（<0.5%差距）
  - 价格在VWAP附近（<1%）
  - 连续5根K线无明确方向
```

---

## 三、中周期入场时机（择时）

### 3.1 信号逻辑

**看涨入场条件**（大周期看涨）：
```
条件1（回调）：
  - 价格回调至EMA(31)或VWAP附近（<1%）
  - RSI(14) > 30（未超卖）
  - 成交量放大（>20日均量）

条件2（突破）：
  - 价格突破局部高点
  - 成交量确认（>20日均量）
  - EMA(17)向上角度

条件3（形态）：
  - 出现看涨吞没形态
  - 形态在支撑位附近（EMA(31)或VWAP）
```

**看跌入场条件**（大周期看跌）：
```
条件1（反弹）：
  - 价格反弹至EMA(31)或VWAP附近（<1%）
  - RSI(14) < 70（未超买）
  - 成交量放大

条件2（跌破）：
  - 价格跌破局部低点
  - 成交量确认
  - EMA(17)向下角度

条件3（形态）：
  - 出现看跌吞没形态
  - 形态在阻力位附近
```

### 3.2 入场信号机

```python
EntrySignal = {
    'id': str,
    'timestamp': datetime,
    'bar_index': int,
    'trend_direction': 'bullish' | 'bearish',  # 大周期方向
    'entry_type': 'pullback' | 'breakout' | 'pattern',
    'entry_price': float,
    'vwap': float,
    'ema_31': float,
    'ema_17': float,
    'rsi': float,
    'volume': float,
    'volume_avg': float,
    'local_high_low': float,  # 局部高点/低点
    'is_confirmed': bool
}
```

### 3.3 信号过滤条件

**必须满足**：
1. 大周期趋势确认（EMA交叉 + VWAP一致）
2. 价格在入场区域附近（EMA(31)或VWAP ±1%）
3. RSI不极端（30-70之间）
4. 成交量 > 20日平均（至少1.2倍）

**可选过滤**：
1. 波动率过滤（ADX > 25）
2. 多时间框架一致性（15分钟同方向）
3. 避市时间过滤（避免开盘/收盘前后30分钟）

---

## 四、小级别精确入场（执行）

### 4.1 精确入场价格

**限价单策略**（推荐）：
```
看涨入场价 = 支撑位 - 缓冲（0.1%或2 tick）
看跌入场价 = 阻力位 + 缓冲（0.1%或2 tick）

支擈位选择（优先级）：
  1. VWAP（动态支擈）
  2. EMA(31)（趋势线支擈）
  3. 前期低点（结构性支擈）
```

**市价单策略**（快进快出）：
```
条件：市场快速突破，等待限价单可能错过
触发：价格突破局部高点/低点 + 成交量确认
```

### 4.2 入场时机确认

**二次确认机制**：
```
方法1：K线确认
  - 等待收盘确认支撑/阻力有效
  - 出现看涨/看跌K线形态

方法2：小周期确认（可选）
  - 检查15分钟图表
  - 15分钟价格在入场区域附近
  - 15分钟EMA方向与1小时一致
```

---

## 五、止损止盈设计（风险控制）

### 5.1 Supertrend动态止损

**Supertrend参数**：
- ATR周期：10
- ATR因子：3.0
- 计算：基于ATR的动态支擈/阻力线

**止损逻辑**：
```python
看涨止损：
  - 初始：Supertrend下线
  - 移动：跟随Supertrend下线上移
  - 盈利保护：浮盈 > 2×风险时，移至盈亏平衡点

看跌止损：
  - 初始：Supertrend上线
  - 移动：跟随Supertrend上线下移
  - 盈利保护：浮盈 > 2×风险时，移至盈亏平衡点
```

**Supertrend优势**：
- 自适应波动率
- 动态调整支擈位
- 避免固定止损被打

### 5.2 ATR固定止损（备选）

**参数**：
- ATR周期：14
- ATR倍数：1.5
- 计算：入场价 ± ATR(14) × 1.5

**使用场景**：
- Supertrend计算失败或异常时
- 波动率极低时（ATR < 历史均值50%）
- 测试对比时使用

### 5.3 止盈策略

**多目标止盈**（推荐）：

```
目标1（保守）：
  - 价格 = 入场价 + (入场价 - 止损) × 1.0
  - 平仓比例：30%

目标2（标准）：
  - 价格 = 入场价 + (入场价 - 止损) × 2.0
  - 平仓比例：40%

目标3（激进）：
  - 价格 = 入场价 + (入场价 - 止损) × 3.0
  - 平仓比例：30%（剩余移动止损跟踪）
```

**替代止盈方案**：

| 方案 | 止盈依据 | 适用场景 |
|------|---------|---------|
| 下一结构位 | 下一个支擈/压力位 | 有明显结构时 |
| Supertrend反转 | 价格突破Supertrend对侧 | 趋势反转信号 |
| 时间止盈 | 持仓超过N根K线 | 震荡市场 |

### 5.4 止损止盈机

```python
TradeRisk = {
    'entry_price': float,
    'stop_loss': float,
    'initial_risk': float,  # 入场价到止损的距离
    'tp1_price': float,
    'tp2_price': float,
    'tp3_price': float,
    'supertrend_sl': float,  # 当前Supertrend止损位
    'atr': float,  # 当前ATR值
    'is_trailing': bool,
    'breakeven_reached': bool
}
```

---

## 六、仓位管理（资金管理）

### 6.1 风险控制原则

**单笔交易风险**：
```
风险比例：1%（可调整0.5%-2%）
风险金额 = 账户资金 × 1%
仓位大小 = 风险金额 / |入场价 - 止损|
```

**持仓限制**：
```
同时持仓：≤ 3个
同一方向：≤ 2个
同一合约：≤ 1个
连续亏损：3次后暂停交易10根K线
```

**最大回撤控制**：
```
回撤限制：10%
触发条件：当前浮亏 > 账户资金 × 10%
应对：停止新开仓，等待回撤恢复至5%以下
```

### 6.2 仓位大小计算

```python
def calculate_position_size(account_balance, entry_price, stop_loss, risk_pct=0.01):
    risk_amount = account_balance * risk_pct
    initial_risk = abs(entry_price - stop_loss)
    position_size = risk_amount / initial_risk
    return position_size
```

**示例**：
```
账户资金：100,000
风险比例：1%
入场价：100
止损价：99

风险金额：100,000 × 1% = 1,000
初始风险：|100 - 99| = 1
仓位大小：1,000 / 1 = 1,000 单
```

### 6.3 仓位调整策略

**信号强度调整**：
```
强信号（HPZ、多重确认）：
  - 仓位：标准仓位的1.2倍

普通信号：
  - 仓位：标准仓位

弱信号（单次确认）：
  - 仓位：标准仓位的0.8倍
```

**波动率调整**：
```
高波动率（ATR > 历史均值150%）：
  - 仓位：标准仓位的0.7倍

正常波动率（ATR在50%-150%）：
  - 仓位：标准仓位

低波动率（ATR < 历史均值50%）：
  - 仓位：标准仓位的1.3倍
```

---

## 七、回测框架设计

### 7.1 回测参数配置

```python
BacktestConfig = {
    # 数据配置
    'instrument': str,  # 商品指数合约代码
    'start_date': str,  # "2024-01-01"
    'end_date': str,  # "2024-12-31"
    'data_freq': str,  # "1H", "4H", "1D"

    # 趋势参数
    'ema_short_period': 17,
    'ema_long_period': 31,
    'vwap_session_bars': 20,

    # 入场参数
    'rsi_period': 14,
    'volume_avg_period': 20,
    'pullback_tolerance': 0.01,  # 1%

    # 止损止盈参数
    'supertrend_atr_period': 10,
    'supertrend_atr_factor': 3.0,
    'atr_sl_period': 14,
    'atr_sl_multiplier': 1.5,
    'tp1_rr': 1.0,  # Risk:Reward倍数
    'tp2_rr': 2.0,
    'tp3_rr': 3.0,

    # 仓位管理
    'risk_per_trade': 0.01,  # 1%
    'max_positions': 3,
    'max_drawdown_pct': 0.10,  # 10%

    # 交易成本
    'slippage_bps': 5,  # 5个基点滑点
    'commission_per_unit': 2,  # 每手2元手续费
    'slippage_pct': 0.0005  # 0.05%价格滑点
}
```

### 7.2 回测指标计算

**交易指标**：
```python
TradeMetrics = {
    'total_trades': int,
    'winning_trades': int,
    'losing_trades': int,
    'win_rate': float,
    'avg_win': float,
    'avg_loss': float,
    'profit_factor': float,  # 总盈利 / 总亏损
    'expectancy': float,  # 平均每笔期望
}
```

**收益指标**：
```python
ReturnMetrics = {
    'total_pnl': float,
    'total_pnl_pct': float,
    'annual_return': float,  # 年化收益
    'max_drawdown': float,
    'max_drawdown_pct': float,
    'sharpe_ratio': float,
    'sortino_ratio': float,
    'calmar_ratio': float,
}
```

**效率指标**：
```python
EfficiencyMetrics = {
    'avg_days_per_trade': float,
    'avg_bars_per_trade': int,
    'trade_frequency': float,  # 每月交易数
    'capital_utilization': float,  # 资金使用率
}
```

### 7.3 回测流程

```
1. 数据加载
   - 使用ParquetDataProvider导入数据
   - 转换为Qlib格式（如果需要）

2. 指标计算
   - 计算EMA(17)、EMA(31)、VWAP
   - 计算RSI、ATR、Supertrend
   - 存储所有指标到数据集

3. 信号识别
   - 大周期：EMA交叉 + VWAP确认
   - 中周期：入场信号（回调/突破/形态）
   - 小周期：精确入场点

4. 交易模拟
   - 记录所有入场/出场
   - 模拟滑点和手续费
   - 追踪止损止盈

5. 性能评估
   - 计算所有回测指标
   - 生成 equity curve
   - 分析交易分布

6. 报告生成
   - 生成详细报告
   - 可视化结果
   - 参数敏感性分析
```

### 7.4 实际要素考虑

**滑点**：
- 商品指数：5-10个基点
- 快速市场：10-20个基点
- 限价单：0-5个基点

**手续费**：
- 每手固定手续费：1-3元
- 每手百分比：0.01%-0.05%

**流动性**：
- 避免开仓/平仓在低流动性时段
- 大单可能分批执行

**市场影响**：
- 大仓位可能冲击价格
- 考虑实际成交量限制

---

## 八、多Agent协作方案

### 8.1 Agent职责划分

**Agent 1：数据处理Agent**
- 职责：数据加载、清洗、转换
- 工具：ParquetDataProvider、Pandas
- 输出：干净的OHLCV数据

**Agent 2：信号分析Agent**
- 职责：计算技术指标、识别信号
- 工具：TA-Lib、NumPy
- 输出：大周期趋势、中周期入场信号

**Agent 3：风险管理Agent**
- 职责：计算仓位、止损止盈
- 工具：ATR、Supertrend
- 输出：交易指令（入场、止损、止盈）

**Agent 4：回测执行Agent**
- 职责：模拟交易、计算指标
- 工具：Qlib、自定义回测引擎
- 输出：回测结果、报告

**Agent 5：综合报告Agent**
- 职责：汇总结果、生成报告
- 工具：Matplotlib、Jinja2
- 输出：HTML报告、图表

### 8.2 Agent协作流程

```
开始
  ↓
[Agent 1] 数据处理
  ├─ 加载Parquet数据
  ├─ 转换为Qlib格式
  └─ 输出：df = get_data()
  ↓
[Agent 2] 信号分析
  ├─ 计算技术指标
  ├─ 识别趋势信号
  └─ 输出：signals = analyze_trend()
  ↓
[Agent 3] 风险管理
  ├─ 计算仓位大小
  ├─ 设置止损止盈
  └─ 输出：trades = calculate_risk()
  ↓
[Agent 4] 回测执行
  ├─ 模拟交易
  ├─ 计算回测指标
  └─ 输出：results = backtest()
  ↓
[Agent 5] 综合报告
  ├─ 汇总结果
  ├─ 生成报告
  └─ 输出：report = generate_report()
  ↓
完成
```

### 8.3 Agent间通信

**共享数据结构**：
```python
SharedContext = {
    'data': pd.DataFrame,  # K线数据
    'indicators': dict,  # 技术指标
    'trend_signals': list,  # 趋势信号
    'entry_signals': list,  # 入场信号
    'trades': list,  # 交易记录
    'metrics': dict,  # 回测指标
    'config': dict,  # 配置参数
}
```

**通信机制**：
- 内存共享（单机多进程）
- 消息队列（异步通信）
- 文件系统（持久化中间结果）

---

## 九、实现技术栈

### 9.1 核心依赖

```python
# 数据处理
import pandas as pd
import numpy as np

# 技术指标
import talib
import ta

# 回测框架
import qlib
from qlib.data import D
from qlib.contrib.evaluate import risk_analysis

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 配置管理
import yaml
from dataclasses import dataclass
```

### 9.2 自定义模块

```
projects/multi_timeframe_trend_strategy/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── default.yaml  # 默认配置
│   └── backtest.yaml  # 回测配置
├── data/
│   ├── __init__.py
│   ├── provider.py  # 数据提供者
│   └── preprocessor.py  # 数据预处理
├── indicators/
│   ├── __init__.py
│   ├── trend.py  # 趋势指标
│   ├── signal.py  # 信号识别
│   └── supertrend.py  # Supertrend计算
├── risk/
│   ├── __init__.py
│   ├── position.py  # 仓位管理
│   └── stoploss.py  # 止损止盈
├── backtest/
│   ├── __init__.py
│   ├── engine.py  # 回测引擎
│   └── metrics.py  # 指标计算
├── agents/
│   ├── __init__.py
│   ├── data_agent.py
│   ├── signal_agent.py
│   ├── risk_agent.py
│   ├── backtest_agent.py
│   └── report_agent.py
├── report/
│   ├── __init__.py
│   ├── generator.py  # 报告生成器
│   └── templates/  # 报告模板
└── utils/
    ├── __init__.py
    ├── helpers.py
    └── logger.py
```

---

## 十、参数调优策略

### 10.1 关键参数范围

**EMA周期**：
```
EMA短周期：10-25期（默认17）
EMA长周期：20-50期（默认31）
调优目标：最大化夏普比率
```

**VWAP会话**：
```
会话长度：10-50根K线（默认20）
调优目标：最大化信号准确率
```

**Supertrend参数**：
```
ATR周期：5-20期（默认10）
ATR因子：2.0-5.0（默认3.0）
调优目标：最大化风险收益比
```

**止损止盈倍数**：
```
止盈倍数：1.0-4.0（默认1.0/2.0/3.0）
调优目标：最大化盈利因子
```

### 10.2 参数调优方法

**Walk-Forward分析**：
```
1. 将数据分为训练集（70%）、验证集（15%）、测试集（15%）
2. 在训练集上优化参数
3. 在验证集上验证
4. 在测试集上评估
5. 滚动窗口重复上述步骤
```

**网格搜索**：
```
对每个参数生成候选项
遍历所有组合（全网格搜索）
或使用优化算法（贝叶斯优化）
选择最优参数组合
```

**敏感性分析**：
```
固定其他参数，逐个调优每个参数
分析参数变化对结果的影响
选择稳健的参数（非过拟合）
```

---

## 十一、风险提示

### 11.1 策略风险

1. **滞后风险**：EMA和VWAP是滞后指标，可能错过趋势初期
2. **震荡风险**：在震荡市场中，EMA交叉可能产生频繁假信号
3. **参数敏感**：策略表现对参数选择敏感，需充分调优
4. **执行风险**：实盘可能因滑点、延迟导致与回测偏差
5. **过拟合风险**：过度调优可能导致策略在实盘失效

### 11.2 使用建议

1. **充分回测**：至少回测3年数据，覆盖牛市和熊市
2. **模拟盘验证**：实盘前至少模拟盘测试1个月
3. **小资金起步**：实盘从小资金开始，逐步加大
4. **持续监控**：定期检查策略表现，必要时调整参数
5. **分散风险**：不要在单一品种上过度集中

---

## 十二、执行计划

### 阶段1：基础框架搭建（ClaudeCode完成）
- [ ] 创建项目目录结构
- [ ] 实现数据加载模块
- [ ] 实现指标计算模块
- [ ] 实现信号识别模块

### 阶段2：风险管理实现（ClaudeCode完成）
- [ ] 实现止损止盈模块
- [ ] 实现仓位管理模块
- [ ] 实现Supertrend动态止损

### 阶段3：回测引擎开发（ClaudeCode完成）
- [ ] 实现回测引擎
- [ ] 实现指标计算模块
- [ ] 实现报告生成模块

### 阶段4：多Agent协作（多Agent完成）
- [ ] 划分数据、信号、风控、回测、报告Agent
- [ ] 实现Agent间通信
- [ ] 测试协作流程

### 阶段5：回测验证（ClaudeCode完成）
- [ ] 选择测试合约（如HC8888.XSGE）
- [ ] 执行回测（2022-2024）
- [ ] 分析结果
- [ ] 参数调优

### 阶段6：实盘准备
- [ ] 模拟盘测试
- [ ] 小资金实盘
- [ ] 监控和报告

---

## 十三、代码审查报告（2026-02-19）

### 13.1 项目进展对照

#### 阶段1：基础框架搭建 ✅ 已完成
| 任务 | 状态 | 文件 |
|------|------|------|
| 创建项目目录结构 | ✅ | 完整目录结构 |
| 实现数据加载模块 | ✅ | `data/provider.py`, `data/parquet_provider.py` |
| 实现指标计算模块 | ✅ | `indicators/trend.py`, `indicators/supertrend.py` |
| 实现信号识别模块 | ✅ | `indicators/signal.py` |

#### 阶段2：风险管理实现 ✅ 已完成
| 任务 | 状态 | 文件 |
|------|------|------|
| 实现止损止盈模块 | ✅ | `backtest/engine.py` |
| 实现仓位管理模块 | ✅ | `backtest/engine.py:_check_entry()` |
| 实现Supertrend动态止损 | ✅ | `indicators/supertrend.py` |

#### 阶段3：回测引擎开发 ✅ 已完成（部分）
| 任务 | 状态 | 文件 |
|------|------|------|
| 实现回测引擎 | ✅ | `backtest/engine.py` |
| 实现指标计算模块 | ✅ | 各indicator模块 |
| 实现报告生成模块 | ⚠️ 部分完成 | CSV/YAML输出，无可视化 |

#### 阶段4：多Agent协作 ❌ 未开始
- `agents/` 目录存在但所有文件为空

#### 阶段5：回测验证 ⚠️ 部分完成
- 已有回测结果：`results/HC8888.XSGE_*`
- 参数调优：未进行

#### 阶段6：实盘准备 ❌ 未开始

### 13.2 代码质量评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | ⭐⭐⭐⭐ | 模块化良好，扩展性有提升空间 |
| 代码质量 | ⭐⭐⭐⭐ | 可读性强，存在功能性bug |
| 数据管理 | ⭐⭐⭐⭐⭐ | `ParquetDataProvider`设计优秀 |
| 回测引擎 | ⭐⭐⭐ | 基础完整，分批止盈有bug |
| 文档完整度 | ⭐⭐⭐⭐ | 计划详尽，注释充分 |
| 测试覆盖 | ⭐⭐ | 有自测代码，无单元测试 |

**综合评分：⭐⭐⭐⭐ (3.7/5)**

### 13.3 各模块详细评价

#### data/parquet_provider.py ⭐⭐⭐⭐⭐
**优点：**
- 统一的数据访问接口设计优秀
- 智能缓存机制（LRU + Parquet/Qlib双格式）
- 完善的异常处理
- 配置与实现分离

#### indicators/ 各模块 ⭐⭐⭐⭐
**优点：**
- 趋势判断、Supertrend计算逻辑正确
- 信号质量评分系统设计优秀
- 数据结构使用dataclass，可读性强

**问题：**
- `signal._align_daily_trend()` 使用bisect逐行匹配，效率低
- 趋势强度阈值硬编码

#### backtest/engine.py ⭐⭐⭐
**优点：**
- 事件驱动回测框架完整
- Trade数据结构设计完善
- 动态止损跟踪实现正确

**问题：**
- **分批止盈未正确实现**（触发TP1后应平30%，当前平全部）
- 代码有重复（看涨/看跌止损计算）

#### strategy.py ⭐⭐⭐
**优点：**
- 流程清晰（7步）
- 输出信息丰富

**问题：**
- 使用print而非logging模块
- 缺少异常处理
- 硬编码初始资金100000

### 13.4 问题清单（按优先级）

#### 🔴 高优先级
| 问题 | 文件位置 | 影响 |
|------|----------|------|
| 分批止盈未实现 | `backtest/engine.py:351-386` | 回测结果不准确 |
| 多Agent未实现 | `agents/` | 无法并行计算 |

#### 🟡 中优先级
| 问题 | 文件位置 | 影响 |
|------|----------|------|
| 信号质量评分未使用 | `EntrySignal.quality_score` | 功能浪费 |
| 配置与PLAN不一致 | EMA60 vs EMA17/31 | 策略偏离设计 |
| 时间对齐效率低 | `signal._align_daily_trend()` | 性能问题 |

#### 🟢 低优先级
| 问题 | 文件位置 | 影响 |
|------|----------|------|
| 使用print而非logging | `strategy.py` | 日志管理混乱 |
| 缺少单元测试 | 全局 | 代码质量保障弱 |
| 缺少可视化报告 | `report/` | 结果分析不便 |

### 13.5 设计与实现差异

| PLAN设计 | 实际实现 | 影响 |
|----------|----------|------|
| EMA17/EMA31交叉 | EMA60单均线 | 策略简化 |
| VWAP会话计算 | 滚动VWAP | 实现方式不同 |
| 15分钟精确入场 | 仅1小时 | 功能缺失 |
| 分批止盈30%/40%/30% | 触发即全平 | **功能缺陷** |
| Walk-Forward调优 | 未实现 | 参数优化缺失 |
| 多Agent协作 | 未实现 | 性能优化缺失 |

### 13.6 优先修复建议

#### 立即修复（影响功能正确性）
1. **修复分批止盈bug**
   - 文件：`backtest/engine.py`
   - 位置：`_manage_positions()` 方法
   - 改进：TP1触发时平仓30%，继续持有剩余

2. **补充可视化报告**
   - 创建 `report/generator.py`
   - 生成权益曲线图、回撤图、交易分布图

#### 近期改进（提升代码质量）
3. **重构日志系统**
   - 将 `print` 替换为 `logging` 模块
   - 配置化日志级别和输出

4. **添加单元测试**
   - 创建 `tests/` 目录
   - 覆盖核心指标计算逻辑

5. **实现信号质量评分应用**
   - 在回测入场决策中使用 `quality_score`
   - 设置质量阈值过滤低质量信号

#### 中期优化（性能与扩展）
6. **优化时间对齐**
   - 使用 `pandas.merge_asof()` 替代bisect
   - 提升日线/小时线对齐性能

7. **配置与设计对齐**
   - 决策：保持EMA60或改回EMA17/31
   - 更新PLAN.md或代码以保持一致

#### 长期规划（功能扩展）
8. **多Agent实现**（根据需求）
   - 数据Agent、信号Agent、回测Agent、报告Agent
   - 使用消息队列或共享内存通信

9. **参数优化系统**
   - Walk-Forward分析
   - 网格搜索或贝叶斯优化

### 13.7 测试合约回测结果

**合约：** HC8888.XSGE（热卷指数）
**期间：** 2023-01-01 至 2024-12-31
**频率：** 1小时

回测已完成，结果保存在 `results/` 目录：
- `trades_HC8888.XSGE_*.csv` - 交易明细
- `equity_HC8888.XSGE_*.csv` - 权益曲线
- `summary_HC8888.XSGE_*.yaml` - 回测摘要

**注意：** 由于分批止盈bug，结果可能不准确，建议修复后重新回测。

---

**下一步**：根据优先级修复问题，或退出plan模式继续开发。
