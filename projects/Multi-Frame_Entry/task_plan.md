# Task Plan: Multi-Frame Entry Strategy
<!--
  WHAT: 构建多品种市场状态识别模型 + 滚动训练框架 + 回测验证系统
  WHY: 使用机器学习识别趋势/震荡市场状态，通过滚动训练适应市场变化
  WHEN: 2026-02-21 更新，已完成模型训练，准备进入回测阶段
-->

## Goal
实现**多层时间框架策略**：
- **大级别（60min）**: ML模型预测趋势概率，识别交易环境
- **小级别（5/15min）**: MSB+OB策略寻找精确入场点
- **出场管理**: 4层动态止损系统
通过回测验证策略的实战价值。

## Current Phase
Phase 6: 回测框架设计与实现

## 项目状态总览

| Phase | 任务 | 状态 | 说明 |
|-------|------|------|------|
| Phase 1 | 数据预处理 | ✅ 完成 | 多周期数据（1min → 5/15/60min/day） |
| Phase 2 | 标签构建 | ✅ 完成 | 10根K线窗口，二分类标签（趋势/震荡） |
| Phase 3 | 特征工程 | ✅ 完成 | 57个技术指标特征 |
| Phase 4 | 模型训练（年度） | ✅ 完成 | 4品种×5年=20个模型 |
| Phase 5 | 滚动训练优化 | ✅ 完成 | 4品种×20窗口=80个模型（18月窗口） |
| Phase 6 | **回测框架** | 🔄 进行中 | **当前阶段** |
| Phase 7 | 参数优化 | ⏳ 待定 | 24月窗口对比等 |
| Phase 8 | 风险控制 | ⏳ 待定 | 仓位管理、止损止盈 |

---

## ✅ 已完成阶段

### Phase 1: 数据预处理 ✅
- 多周期重采样：1min → 5min/15min/60min/1day
- 数据质量：988,140行，覆盖2014-2025
- 严格时间索引对齐
- **输出**: `/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq/`

### Phase 2: 标签构建 ✅
- **标签类型**: 二分类（趋势=1, 震荡=0）
- **窗口选择**: 10根K线（10小时）
- **Regime过滤**: 高波动vs低波动分类
- **防未来函数**: shift(-10)严格隔离
- **输出**: `data/labels/binary_labels_10bars.csv`, `volatility_regime_labels.csv`

### Phase 3: 特征工程 ✅
- **特征数量**: 57个技术指标
- **特征类别**: 斜率、强度、结构、波动率、技术指标
- **防未来函数**: 所有特征shift(1)
- **输出**: `data/features/binary_features.csv`

### Phase 4: 年度滚动训练 ✅
- **方法**: 逐年滚动（1年训练 → 预测下1年）
- **品种**: HC8888, I8888, AU8888, CF8888
- **结果**:
  - AU8888 (黄金): AUC=0.6446 ⭐
  - CF8888 (郑棉): AUC=0.5823
  - I8888 (铁矿石): AUC=0.5678
  - HC8888 (热卷): AUC=0.5658
- **输出**: `models/rolling/*.pkl` (20个模型)

### Phase 5: 季度滚动训练优化 ✅ (2026-02-21完成)
- **方法**: 18月训练窗口 → 预测未来3月，每季度滚动
- **总窗口**: 20个季度窗口（2021-2025）
- **结果对比**:
  | 品种 | 年度滚动AUC | 季度滚动AUC | 改善 | 稳定性(标准差) |
  |------|-------------|-------------|------|----------------|
  | **AU8888 黄金** | 0.6446 | **0.6537** | **+1.4%** | **0.0526** ⭐ |
  | CF8888 郑棉 | 0.5823 | 0.5840 | +0.3% | 0.0542 |
  | I8888 铁矿石 | 0.5678 | 0.5812 | +2.4% | 0.0768 |
  | HC8888 热卷 | 0.5658 | 0.5758 | +1.8% | 0.0787 |

- **关键发现**:
  - ✅ 18月滚动全面优于年度滚动
  - ✅ AU8888最稳定（2024年标准差仅0.0170）
  - ✅ 季度预测更符合实际交易节奏
- **输出**:
  - `scripts/rolling_train_3month.py` (500+行)
  - `models/rolling_3month/*.pkl` (80个模型)
  - `models/training_results_3month/滚动训练总结报告.xlsx`

---

## 🔄 当前阶段：Phase 6 - 多层时间框架回测系统

### 策略架构设计

```
┌─────────────────────────────────────────────────────────┐
│              多层时间框架策略架构                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  大级别（60min）             小级别（5/15min）             │
│  ┌──────────────┐            ┌──────────────┐           │
│  │ ML趋势模型    │──信号过滤──→│ MSB+OB策略   │           │
│  │              │            │              │           │
│  │ • P(trend)   │    触发      │ • MSB突破    │──入场──┐   │
│  │ • Regime     │            │ • OB订单块   │         │   │
│  │ • 概率阈值   │            │ • 动量过滤   │         │   │
│  └──────────────┘            └──────────────┘         │   │
│                                                  ↓        │   │
│        ↓                                      持仓管理    │   │
│   只在趋势环境交易                               ↓        │   │
│  • P(trend) > 0.6                        ┌──────────┐ │   │
│  • 高波动Regime                        │ 出场系统  │←┘   │
│                                        │          │     │
│  回测品种：AU8888（黄金）                │ 1.初始止损│     │
│  时间：2021-2025                       │ 2.保本    │     │
│  模型：80个季度模型                      │ 3.追踪    │     │
│                                        │ 4.结构破坏 │     │
│                                        └──────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 核心逻辑

#### 1️⃣ 大级别过滤（60min）
**作用**：判断是否应该交易，以及交易方向

```python
# 每60min K线执行一次
if trend_proba > 0.6 and regime == '高波动':
    trading_mode = '多头趋势'
    allow_long = True
    allow_short = False
elif trend_proba < 0.4 and regime == '高波动':
    trading_mode = '空头趋势'
    allow_long = False
    allow_short = True
else:
    trading_mode = '观望'
    allow_long = False
    allow_short = False
```

#### 2️⃣ 小级别入场（5/15min）
**作用**：在大级别允许交易时，寻找精确入场点

```python
# 只在 trading_mode != '观望' 时运行
if trading_mode == '多头趋势':
    # 寻找看涨MSB+OB信号
    msb_bullish = detect_msb_bullish(5min_data)
    if msb_bullish:
        ob_zone = find_order_block()
        if price_in_ob_zone(ob_zone):
            entry_long()

elif trading_mode == '空头趋势':
    # 寻找看跌MSB+OB信号
    msb_bearish = detect_msb_bearish(5min_data)
    if msb_bearish:
        ob_zone = find_order_block()
        if price_in_ob_zone(ob_zone):
            entry_short()
```

#### 3️⃣ 出场管理（4层控制）
**作用**：保护利润，控制风险

```python
def check_exit_signal(position, current_bar):
    # 层1：初始止损（入场时锁定）
    if not position.break_even_triggered:
        if hit_initial_stop():
            return '初始止损'

    # 层2：保本机制
    if position.pnl_r >= 1.0:
        position.stop_loss = position.entry_price
        position.break_even_triggered = True

    # 层3：趋势追踪止损
    if position.break_even_triggered:
        new_stop = calculate_trailing_stop()
        position.stop_loss = max(position.stop_loss, new_stop)

    # 层4：结构破坏强制退出
    if min_hold_bars_exceeded():
        if structure_broken():
            return '结构破坏'

    # 检查是否触发止损
    if hit_stop_loss():
        return '止损'
```

---

### Task 6.1: 回测框架设计 ⏳
- [x] 确定回测品种：**AU8888黄金**（AUC=0.6537，表现最佳）
- [x] 确定回测时间范围：**2021-2025**（与80个季度模型对应）
- [ ] 设计交易规则（参考MSB+OB策略）

  **入场条件**：
  - 主要条件：模型预测 P(trend) > 阈值
  - 过滤条件1：只在高波动Regime交易（模型训练时的假设）
  - 过滤条件2：成交量确认（可选）

  **出场机制**（4层控制，参考入场出场方案.md）：
  1. **初始止损**：结构止损（近期swing low/high）
  2. **保本机制**：盈利 >= 1R 后，止损移到入场价
  3. **趋势追踪止损**：
     - 方案A：EMA20 ± 0.5*ATR（推荐）
     - 方案B：结构高低点追踪
  4. **结构破坏强制退出**：
     - 多单：出现Lower Low或模型预测翻转
     - 空单：出现Higher High或模型预测翻转

  **仓位管理**：
  - 基础：固定手数1手（简化）
  - 进阶：概率加权仓位（0.5~1.0倍）

  **最小持仓时间**：5根K线（防止被噪音洗出）

- [ ] 设置交易成本
  - 手续费：万分之一（0.01%）
  - 滑点：1 tick（5min级别）
  - 保证金：8%（期货标准）

**回测数据架构**：
```
data/multi_symbol/AU8888.XSGE/
├── raw_1min.parquet          # 原始1min数据
├── 5min.parquet              # 重采样5min（用于MSB+OB）
├── 15min.parquet             # 重采样15min（用于MSB+OB）
└── 60min.parquet             # 重采样60min（用于ML模型预测）
```

### Task 6.2: 实现多层时间框架回测引擎 ⏳

**核心模块**：
1. **`backtest/mlt_framework.py`** - 多层时间框架主引擎
   - 协调60min（ML）和5min（MSB+OB）数据
   - 实现时间对齐和信号传递
   - 管理交易状态机

2. **`backtest/trend_filter.py`** - ML趋势过滤器
   - 加载80个季度模型
   - 按60min时间轴滚动预测
   - 输出：trading_mode（多头/空头/观望）

3. **`backtest/msb_ob_entry.py`** - MSB+OB入场模块
   - 参考实现：`/Users/mystryl/Documents/Quant/projects/qlib_msb_ob/strategy/msb_ob_strategy.py`
   - 在5min级别检测MSB信号
   - 识别Order Block区域
   - 生成精确入场信号

4. **`backtest/exit_manager.py`** - 4层出场管理
   - 参考设计：`入场出场方案.md`
   - 实现多层止损逻辑
   - 结构破坏检测
   - 保本和追踪止损

5. **`backtest/position_manager.py`** - 持仓管理
   - 持仓状态跟踪
   - 资金管理
   - 风险控制（最大3个持仓）

**数据流**：
```
60min数据 → ML模型 → trading_mode信号
                          ↓
5min数据 ─────────────→ MSB+OB策略 → 入场信号
                          ↓
                    持仓管理 → 出场管理 → 交易记录
```
- [ ] 创建 `backtest/rolling_backtest.py`
  - 参考设计：`/Users/mystryl/Documents/Quant/projects/qlib_msb_ob/strategy/msb_ob_strategy.py`
  - 数据结构：Position, Order, Trade
- [ ] 加载80个季度模型
  - 按时间顺序匹配模型窗口
  - 实现模型切换逻辑
- [ ] 生成交易信号
  - 每日滚动预测
  - 概率阈值过滤
  - Regime过滤（只用高波动数据）
- [ ] 订单管理
  - 开仓逻辑
  - 持仓跟踪（多层止损）
  - 平仓逻辑
  - 资金管理

### Task 6.3: 回测执行 ⏳

**回测流程**：
```python
# 伪代码
for date in trading_dates_2021_to_2025:
    # 步骤1：获取60min数据，运行ML模型
    df_60min = get_60min_data(date)
    trend_proba = predict_with_rolling_model(df_60min)
    regime = detect_regime(df_60min)

    # 步骤2：确定交易模式
    if trend_proba > 0.6 and regime == '高波动':
        trading_mode = '多头'
    elif trend_proba < 0.4 and regime == '高波动':
        trading_mode = '空头'
    else:
        trading_mode = '观望'

    # 步骤3：如果不在观望模式，在5min级别寻找入场
    if trading_mode != '观望':
        df_5min = get_5min_data(date)

        # 检测MSB+OB信号（只检测对应方向的信号）
        if trading_mode == '多头':
            entry_signals = detect_bullish_msb_ob(df_5min)
        else:
            entry_signals = detect_bearish_msb_ob(df_5min)

        # 执行入场
        for signal in entry_signals:
            if can_open_position():
                open_position(signal)

    # 步骤4：管理现有持仓
    for position in active_positions:
        exit_signal = check_exit_conditions(position)
        if exit_signal:
            close_position(position)

    # 步骤5：记录状态
    record_daily_state(date, positions, equity)
```

**记录的指标**：
- 每笔交易：入场时间/价格、出场时间/价格、盈亏、持仓时长、出场原因
- 每日状态：权益、持仓数、暴露度、回撤
- ML模型状态：每季度的模型切换、预测概率变化

### Task 6.4: 结果分析与报告 ⏳

**Excel报告结构**：
1. **交易明细表**
   - 入场时间、入场价格、出场时间、出场价格
   - 盈亏、盈亏R、持仓时长
   - 出场原因分类
   - ML模型窗口ID

2. **绩效汇总表**
   - 总收益率、年化收益率
   - 夏普比率、最大回撤
   - 胜率、盈亏比
   - 平均持仓时长

3. **多层框架分析**
   - ML过滤效果（多少信号被过滤）
   - MSB+OB入场质量
   - 大小级别配合效果

4. **月度收益表**
   - 每月交易次数、胜率、盈亏
   - 月度收益分布

**可视化图表**：
1. 净值曲线（含回撤阴影）
2. 月度收益热力图
3. 出场原因分布饼图
4. 持仓时长分布直方图
5. ML预测概率时序图

**基准对比**：
- 买入持有（2021-2025）
- 纯MSB+OB策略（无ML过滤）
- 多层策略（完整版本）

**Status**: pending

---

## 📊 策略对比分析

| 维度 | 纯MSB+OB | 纯ML模型 | **多层策略（组合）** |
|------|----------|----------|---------------------|
| 入场精度 | 中等（技术信号） | 高（概率估计） | **高（ML指导方向，MSB精确定位）** |
| 交易频率 | 高（每突破） | 低（每日预测） | **中（趋势环境才交易）** |
| 胜率 | 中等 | 中等 | **高（双重过滤）** |
| 盈亏比 | 中等 | 低 | **高（趋势追踪出场）** |
| 回撤控制 | 弱 | 弱 | **强（多层止损）** |
| 市场适应性 | 中等 | 低 | **高（Regime过滤）** |

---

## 🔑 关键优势

### 1. 时间框架解耦
- 大级别判断"是否应该交易"（战略）
- 小级别决定"何时入场"（战术）

### 2. 双重过滤
- ML模型过滤市场环境（趋势/震荡/高波动）
- MSB+OB过滤入场时机（精确点位）

### 3. 风险控制升级
- 结构止损（参考MSB+OB）
- 趋势追踪（参考入场出场方案）
- 最小持仓时间（防止噪音）

---

---

## ⏳ 待定阶段

### Phase 7: 参数优化（可选）
- [ ] 24个月窗口对比测试
- [ ] 不同阈值测试（0.5, 0.6, 0.7）
- [ ] 不同止损止盈参数测试
- **Status**: pending

### Phase 8: 风险控制增强（可选）
- [ ] 概率加权仓位管理
- [ ] Regime切换仓位调整
- [ ] 高波动降低仓位
- **Status**: pending

---

## Key Questions

### 当前阶段（Phase 6）关键参数设计：

**参考策略**：`/Users/mystryl/Documents/Quant/projects/qlib_msb_ob/strategy/msb_ob_strategy.py`
**出场方案**：`/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/入场出场方案.md`

#### 1️⃣ 入场阈值选择
- **推荐：0.6** - 平衡胜率和交易频率
- 可选对比测试：0.5, 0.6, 0.7

#### 2️⃣ 出场方案（参考入场出场方案.md）
```python
EXIT_CONFIG = {
    "trailing_method": "ema",        # ema / structure / chandelier
    "ema_period": 20,
    "atr_mult": 0.5,
    "break_even_r": 1.0,              # 盈利1R后保本
    "min_hold_bars": 5,               # 最小持仓5根K线
    "atr_period": 14,
    "structure_break": True           # 结构破坏强制退出
}
```

#### 3️⃣ 仓位管理
- **基础版**：固定手数1手
- **进阶版**：概率加权 = base_position * (P(trend) / 0.6)

#### 4️⃣ 风险控制
- 初始止损：结构止损（最近swing low/high）
- 最大单笔风险：2%账户资金
- 最大持仓数：3个（参考MSB+OB策略）

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 使用18月滚动训练 | 平衡稳定性和适应性，优于年度滚动 |
| 选择AU8888作为首选品种 | AUC最高(0.6537)，标准差最小(0.0526) |
| 季度预测窗口 | 3个月符合实际交易节奏 |
| 先做单品种回测 | 验证框架可行性，再扩展到多品种 |
| 严格防未来函数 | 所有特征shift(1)，标签用未来数据 |
| 时间序列分割 | 禁止shuffle，严格按时间顺序 |

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Phase 3-4 代码已实现但文档未更新 | 1 | 更新task_plan.md，标记为完成 |
| Phase 5 编号重复 | 1 | 重新整理阶段编号 |

---

## 下一步行动

**立即任务**：
1. 回答Phase 6的4个关键问题（交易策略、阈值、仓位、止损）
2. 设计回测框架细节
3. 实现回测引擎
4. 执行AU8888回测

**预计耗时**：2-3小时

---

## Notes

### 策略设计原则
1. **时间框架分层**：大级别看方向，小级别找点位
2. **信号过滤**：ML过滤环境，MSB+OB过滤时机
3. **风险优先**：多层止损，结构破坏强制退出
4. **纪律性**：最小持仓时间，不被噪音洗出

### 数据准备状态
- ✅ 60min数据：ML模型预测
- ✅ 5/15min数据：MSB+OB入场
- ✅ 80个季度模型：覆盖2021-2025
- ✅ 所有数据已对齐时间索引

### 实现优先级
1. 先实现简化版（60min + 5min）
2. 测试框架可行性
3. 优化细节（出场管理、资金管理）
4. 扩展到多品种

---

**最后更新**: 2026-02-21 16:00
**当前状态**: Phase 5完成，Phase 6规划完成
**下一阶段**: 实现多层时间框架回测系统
