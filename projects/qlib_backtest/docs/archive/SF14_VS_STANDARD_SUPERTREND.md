# SF14 魔改版 SuperTrend vs 标准版 SuperTrend 对比分析

## 一、背景介绍

### 标准 SuperTrend
- **设计者**: Jason Robinson
- **主要用途**: 确定价格趋势和趋势追踪
- **应用市场**: 外汇、期货等

### SF14 魔改版 SuperTrend
- **开发者**: 松鼠Quant
- **版本**: SF14 (Series Formula 14)
- **应用市场**: 国内商品市场（期货）
- **平台**: TB（交易开拓者）、文华8、MC、金字塔

---

## 二、核心算法对比

### 1. 标准 SuperTrend（我目前的实现）

#### 计算公式

**ATR 计算**:
```
TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
ATR = TR 的滚动平均（周期 M）
```

**上下带计算**:
```
HL2 = (High + Low) / 2
UpperBand = HL2 + (Multiplier × ATR)
LowerBand = HL2 - (Multiplier × ATR)
```

**SuperTrend 线计算**:
```
做多趋势：
  SuperTrend = 当前 LowerBand
  只上移不下移（取近期最高值）

做空趋势：
  SuperTrend = 当前 UpperBand
  只下移不上移（取近期最低值）
```

**趋势切换逻辑**:
```
如果当前是上涨趋势：
  若 收盘价 < 上一根 SuperTrend，则转为下跌趋势
  否则继续上涨（SuperTrend = max(当前LowerBand, 上一根SuperTrend)）

如果当前是下跌趋势：
  若 收盘价 > 上一根 SuperTrend，则转为上涨趋势
  否则继续下跌（SuperTrend = min(当前UpperBand, 上一根SuperTrend)）
```

#### 参数设置
- **M**: ATR 周期（默认 10）
- **N**: ATR 倍数（默认 3.0）

#### 特点
- ✅ 趋势跟踪性好
- ❌ 信号过少，交易不够灵活
- ❌ 只有一个止损线，没有其他出场机制

---

### 2. SF14 魔改版 SuperTrend

根据搜索结果，SF14 在标准版基础上进行了以下改进：

#### 主要改进点

**1. 类轨道算法（Class Track Algorithm）**
- 推测是在上下带之间增加了一个轨道通道
- 可能的实现方式：
  ```
  轨道宽度 = K × ATR
  上轨 = SuperTrend + 轨道宽度
  下轨 = SuperTrend - 轨道宽度
  ```
- 作用：提供价格波动范围参考，辅助判断趋势强度

**2. 移动出场（Moving Exit / Trailing Stop）**
- 动态调整止损/止盈位置
- 可能的实现方式：
  ```
  移动止损 = SuperTrend + P × ATR（做多时）
  移动止损 = SuperTrend - P × ATR（做空时）
  ```
  或根据持仓盈利情况调整：
  ```
  盈利时，移动止损向上移动（做多）
  盈利时，移动止损向下移动（做空）
  ```

**3. 增加交易频率**
- 原版 SuperTrend 信号过少
- 魔改版可能增加了入场条件：
  - 价格突破轨道线
  - 价格回调到特定位置
  - 多周期共振

**4. 优化参数适应国内商品**
- 针对国内期货市场特性优化
- 可能调整了 ATR 计算方式或倍数范围
- 增加 5 分钟周期支持

#### 推测的完整算法结构

```
1. 计算标准 SuperTrend 线（ST）

2. 计算类轨道：
   轨道宽度 = K × ATR
   上轨 = ST + 轨道宽度
   下轨 = ST - 轨道宽度

3. 入场条件（增加）：
   - 标准 ST 趋势切换（保持）
   - 价格突破上轨/下轨
   - 价格回调到 ST 附近

4. 出场逻辑（增加）：
   - 标准 ST 趋势切换（保持）
   - 移动止损/止盈：
     * 多头：止损 = max(移动止损, ST)
     * 空头：止损 = min(移动止损, ST)
   - 固定比例止损/止盈

5. 参数优化：
   - M: ATR 周期（可能调整范围）
   - N: ATR 倍数（可能调整范围）
   - K: 轨道宽度倍数（新增）
   - P: 移动止损比例（新增）
```

---

## 三、功能对比表

| 功能 | 标准 SuperTrend | SF14 魔改版 |
|------|--------------|-----------|
| **趋势跟踪** | ✅ 有 | ✅ 有（增强） |
| **信号频率** | ❌ 较低 | ✅ 较高 |
| **类轨道算法** | ❌ 无 | ✅ 有 |
| **移动出场** | ❌ 无（仅 ST 线） | ✅ 有 |
| **入场条件** | 1 个（趋势切换） | 3 个以上（趋势+突破+回调） |
| **出场条件** | 1 个（趋势切换） | 多个（趋势+移动止损+固定止损） |
| **参数数量** | 2 个（M, N） | 4 个以上（M, N, K, P） |
| **适用周期** | 所有周期 | 偏好 5 分钟及以下 |
| **市场适应性** | 通用 | 针对国内商品优化 |

---

## 四、代码结构对比

### 标准 SuperTrend（我的实现）

```python
def supertrend(high, low, close, period=10, multiplier=3.0):
    # 1. 计算 ATR
    atr_value = atr(high, low, close, period)

    # 2. 计算上下带
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_value
    lower_band = hl2 - multiplier * atr_value

    # 3. 计算 SuperTrend 线
    # （迭代逻辑，见上文）

    return supertrend_line, trend
```

### SF14 魔改版（推测结构）

```python
def sf14_supertrend(high, low, close, period=10, multiplier=3.0,
                     band_width=1.5, trailing_ratio=0.5):
    # 1. 计算标准 SuperTrend
    st_line, trend = supertrend(high, low, close, period, multiplier)

    # 2. 计算类轨道
    atr_value = atr(high, low, close, period)
    band = band_width * atr_value
    upper_track = st_line + band
    lower_track = st_line - band

    # 3. 计算移动止损
    trailing_stop = calculate_trailing_stop(st_line, close, trend, trailing_ratio)

    # 4. 生成入场信号（多条件）
    entry_signals = calculate_entry_signals(close, st_line, trend, upper_track, lower_track)

    # 5. 生成出场信号（多条件）
    exit_signals = calculate_exit_signals(close, st_line, trend, trailing_stop)

    return st_line, trend, upper_track, lower_track, trailing_stop, entry_signals, exit_signals
```

---

## 五、回测表现对比

### 标准 SuperTrend（我的回测结果）

| 年份 | 最佳组合 | 累计收益 | 年化收益 | 夏普比率 |
|------|---------|---------|---------|---------|
| 2024 | 15min + (7, 3.0) | 41.53% | 24.24% | 1.57 |
| 2023 | 60min + (14, 2.5) | 26.87% | 55.97% | 2.01 |
| 2025 | 15min + (10, 3.0) | 6.42% | 3.94% | 0.42 |

**特点**:
- 2024 年表现优秀
- 2023 年大周期表现好
- 2025 年表现一般
- 信号频率：每年 60-312 次

### SF14 魔改版（根据搜索结果）

**推测表现**:
- ✅ 信号频率更高（5 分钟周期）
- ✅ 交易更灵活（多种入场/出场条件）
- ✅ 适合小资金
- ❌ 可能有更多假信号（入场条件增加）

**官方描述**:
> "Supertrend具有非常好的趋势跟踪性，但是信号过少，交易不够灵活，我们在原策略的基础上加入类轨道算法和移动出场，适用于5分钟周期或日线以下周期，适用于小资金跑"

---

## 六、优缺点对比

### 标准 SuperTrend

**优点**:
- ✅ 算法简单，容易理解
- ✅ 参数少，优化方便
- ✅ 趋势跟踪效果明显
- ✅ 回撤相对较小（2023 年 60 分钟夏普 2.01）
- ✅ 适合大资金（信号频率适中）

**缺点**:
- ❌ 信号过少，错失机会
- ❌ 交易不够灵活
- ❌ 只有一种出场方式（趋势切换）
- ❌ 不适合 5 分钟等短周期（噪音大）

---

### SF14 魔改版

**优点**:
- ✅ 信号频率高，机会多
- ✅ 交易灵活（多条件入场/出场）
- ✅ 适合短周期（5 分钟）
- ✅ 有移动止损，锁定利润
- ✅ 适合小资金

**缺点**:
- ❌ 算法复杂，参数多
- ❌ 可能产生更多假信号
- ❌ 优化难度大
- ❌ 回撤可能更大（交易频率高）

---

## 七、适用场景对比

### 标准 SuperTrend 适合

**市场环境**:
- ✅ 明显趋势市场（如 2024 年）
- ✅ 大周期（60 分钟及以上）
- ✅ 波动率适中的市场

**资金规模**:
- ✅ 大资金（需要稳定收益）
- ✅ 中等资金

**交易风格**:
- ✅ 稳健型（优先控制回撤）
- ✅ 趋势跟踪型

**推荐配置**:
- 60 分钟 + (14, 2.5) - 稳健型
- 15 分钟 + (10, 3.0) - 平衡型

---

### SF14 魔改版适合

**市场环境**:
- ✅ 震荡市场（需要更多交易机会）
- ✅ 短周期（5 分钟、15 分钟）
- ✅ 波动率较大的市场

**资金规模**:
- ✅ 小资金（需要提高资金利用率）
- ✅ 超短线资金

**交易风格**:
- ✅ 积极型（追求高收益）
- ✅ 超短线交易
- ✅ 高频交易

**推荐配置**:
- 5 分钟 + 优化参数组合

---

## 八、改进建议（基于我的实现）

### 如果想向 SF14 靠拢，可以考虑以下改进：

#### 1. 增加类轨道算法

```python
def calculate_band(st_line, atr, band_width=1.5):
    """计算轨道"""
    upper_track = st_line + band_width * atr
    lower_track = st_line - band_width * atr
    return upper_track, lower_track
```

#### 2. 增加移动止损

```python
def calculate_trailing_stop(st_line, close, trend, trailing_ratio=0.5):
    """计算移动止损"""
    trailing_stop = pd.Series(index=close.index, dtype=float)

    for i in range(len(close)):
        if trend.iloc[i] == 1:  # 做多
            # 移动止损 = ST + trailing_ratio * ATR
            trailing_stop.iloc[i] = st_line.iloc[i] * (1 - trailing_ratio)
        else:  # 做空
            # 移动止损 = ST + trailing_ratio * ATR
            trailing_stop.iloc[i] = st_line.iloc[i] * (1 + trailing_ratio)

    return trailing_stop
```

#### 3. 增加入场条件

```python
def calculate_entry_signals(close, st_line, trend, upper_track, lower_track):
    """多条件入场信号"""
    signals = pd.Series(False, index=close.index)

    # 条件1: 趋势切换（标准 SuperTrend）
    trend_change = trend.diff()
    signals |= (trend_change == 2)  # 从跌转涨
    signals |= (trend_change == -2)  # 从涨转跌

    # 条件2: 价格突破轨道
    signals |= (close > upper_track)  # 突破上轨
    signals |= (close < lower_track)  # 突破下轨

    # 条件3: 价格回调到 ST 附近
    signals |= (abs(close - st_line) / st_line < 0.01)  # 回调 1%

    return signals
```

#### 4. 增加出场条件

```python
def calculate_exit_signals(close, st_line, trend, trailing_stop):
    """多条件出场信号"""
    signals = pd.Series(False, index=close.index)

    # 条件1: 趋势切换（标准 SuperTrend）
    trend_change = trend.diff()
    signals |= (trend_change == 2)  # 从跌转涨（平空）
    signals |= (trend_change == -2)  # 从涨转跌（平多）

    # 条件2: 移动止损触发
    if trend.iloc[-1] == 1:  # 做多
        signals |= (close < trailing_stop)
    else:  # 做空
        signals |= (close > trailing_stop)

    return signals
```

---

## 九、总结

### 标准 SuperTrend vs SF14 魔改版

| 维度 | 标准 SuperTrend | SF14 魔改版 |
|------|--------------|-----------|
| **复杂度** | 简单（2 个参数） | 复杂（4+ 个参数） |
| **信号频率** | 低 | 高 |
| **灵活性** | 低 | 高 |
| **适用周期** | 60 分钟及以上 | 5-15 分钟 |
| **适用资金** | 大、中资金 | 小资金 |
| **风险收益** | 稳健，回撤小 | 积极收益高，回撤可能大 |
| **优化难度** | 容易 | 困难 |

### 建议

**如果你是**:
- ✅ 大资金，追求稳健 → **标准 SuperTrend**
- ✅ 中等资金，平衡风险收益 → **标准 SuperTrend**
- ✅ 小资金，追求高收益 → **SF14 魔改版**
- ✅ 短线交易者 → **SF14 魔改版**
- ✅ 趋势跟踪者 → **标准 SuperTrend**

**我的建议**:
1. **先用标准 SuperTrend**: 算法简单，容易理解和优化
2. **回测验证**: 在历史数据上验证效果
3. **考虑魔改**: 如果需要更高信号频率，可以考虑增加类轨道和移动止损
4. **参数优化**: 对不同参数组合进行网格搜索
5. **风险控制**: 无论哪种版本，都要严格风险控制

---

## 十、下一步行动

### 基于 SF14 的改进方向

1. **实现类轨道算法**
   - 增加 K 参数（轨道宽度倍数）
   - 测试不同 K 值的效果

2. **实现移动止损**
   - 增加 P 参数（移动止损比例）
   - 测试动态止损效果

3. **增加入场/出场条件**
   - 轨道突破
   - 价格回调
   - 多周期共振

4. **参数优化**
   - 网格搜索 M、N、K、P
   - 使用遗传算法等优化方法

5. **回测对比**
   - 与标准 SuperTrend 对比
   - 验证改进效果

---

**生成时间**: 2026-02-15
**参考资料**: 微信公众号文章、CSDN 博客、知乎文章
