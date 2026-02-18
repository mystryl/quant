# MSB-OB 策略简介

## 策略概述

**MSB-OB (Market Structure Break & Order Block)** 是一个基于市场结构突破和订单块的高性能期货交易策略。该策略由 LuxAlgo 开发，结合了价格行为分析和机构订单流识别。

**项目路径**: `projects/qlib_msb_ob/`

---

## 核心逻辑

### 1. 市场结构突破 (MSB)

- **看涨 MSB**: 收盘价突破前高，且动量 Z-Score > 设定阈值
- **看跌 MSB**: 收盘价跌破前低，且动量 Z-Score < 设定阈值

### 2. 机构订单块 (Order Block)

- **识别逻辑**: 在 MSB 发生后，回溯查找最近的反向K线作为订单块
- **质量评分**: 根据动量强度和成交量 percentile 计算 (0-100分)
- **高性能区域 (HPZ)**: 评分 > 80 的订单块标记为高概率区域

### 3. 动态管理

- **订单块缓解**: 价格再次进入订单块区域时视为"缓解"
- **重叠处理**: 可选择隐藏重叠的订单块
- **历史显示**: 保留或隐藏已缓解的订单块

---

## 策略特点

### 优势

✅ **高可靠性**: 统计显示缓解率通常 > 50%
✅ **风险可控**: 每个订单块都有明确的进场和止损位置
✅ **机构视角**: 识别真实的市场参与区域
✅ **多维分析**: 结合动量、成交量、价格结构

### 适用市场

- **期货合约**: 热门商品期货 (HC, RB, AU, CU 等)
- **时间周期**: 1分钟级别效果最佳
- **市场状态**: 趋势市场和震荡市场均适用

---

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Pivot Lookback | 7 | 枢轴点回看周期 |
| MSB Momentum Z-Score | 0.5 | MSB 确认的动量阈值 |
| Max Active OBs | 10 | 最大活跃订单块数量 |
| Extend Broken OBs | false | 是否延伸已破位的订单块 |

### 交易时段 (可选)

- 伦敦时段: 08:00-17:00
- 纽约时段: 13:00-22:00
- 东京时段: 00:00-09:00
- 悉尼时段: 21:00-06:00

---

## Qlib 实现版本

### 版本演进

1. **v01_original_pine** - Pine Script 原版移植
   - 完整还原 Pine Script 逻辑
   - 1分钟数据回测
   - 手续费滑点设置: 0.2/0.2

2. **v02_simple_entry** - 简化入场版本
   - 优化入场逻辑
   - 减少误信号

3. **v03_hybrid** - 混合优化版本
   - 结合 v01 和 v02 优势
   - 增强过滤器

### 使用方法

```bash
cd projects/qlib_msb_ob

# 运行回测
python run_backtest.py

# 运行简单版本
python run_simple_msb.py
```

---

## 性能指标

### 典型结果 (HC8888 主力连续)

- **年化收益率**: 待定
- **最大回撤**: 待定
- **夏普比率**: 待定
- **订单块可靠性**: > 50%

---

## Pine Script 源码

完整的 Pine Script v5 源码见下方，用于在 TradingView 上可视化验证。

---

## 注意事项

⚠️ **风险提示**
- 历史表现不代表未来收益
- 建议在模拟环境充分测试
- 实盘交易需严格风险控制
- 订单块区域仅为参考，需结合其他分析

---

## 参考资料

- **原始作者**: LuxAlgo
- **许可证**: CC BY-NC-SA 4.0
- **TradingView**: [Market Structure Break & OB Probability Toolkit](https://www.tradingview.com/script/...)

---

## Pine Script 源码

```pine
// This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
// © LuxAlgo

//@version=6
indicator("Market Structure Break & OB Probability Toolkit [LuxAlgo]", "LuxAlgo - Market Structure Break & OB Probability Toolkit", overlay = true, maxboxescount = 500, maxlinescount = 500, maxlabelscount = 500)
//---------------------------------------------------------------------------------------------------------------------}
// Constants
//---------------------------------------------------------------------------------------------------------------------{
color BULLCOLOR      = #089981
color BEARCOLOR      = #f23645
color HPZBULL        = #1de9b6
color HPZBEAR        = #ff5252
color NEUTRALCOLOR   = color.gray

// Table Constants
DATA                  = #DBDBDB
HEADERS                 = #808080
BACKGROUND              = #161616
BORDERS                 = #2E2E2E

TOPRIGHT               = 'Top Right'
BOTTOMRIGHT            = 'Bottom Right'
BOTTOMLEFT             = 'Bottom Left'

TINY                    = 'Tiny'
SMALL                   = 'Small'
NORMAL                  = 'Normal'
LARGE                   = 'Large'
HUGE                    = 'Huge'

//---------------------------------------------------------------------------------------------------------------------}
// Inputs
//---------------------------------------------------------------------------------------------------------------------{
DASHBOARDGROUP         = 'Dashboard'
MSGROUP                = "Market Structure"
VISUALSGROUP           = "Visuals"
OBGROUP                = "Order Blocks"
SESSIONSGROUP          = "Sessions"

// Dashboard Inputs
showStatsInput          = input.bool(   true,       'Show Advanced Analytics', group = DASHBOARDGROUP)
dashboardPositionInput  = input.string( TOPRIGHT,  'Position',                group = DASHBOARDGROUP, options = [TOPRIGHT,BOTTOMRIGHT,BOTTOMLEFT])
dashboardSizeInput      = input.string( SMALL,      'Size',                    group = DASHBOARDGROUP, options = [TINY,SMALL,NORMAL,LARGE,HUGE])

// Market Structure
pivotLenInput           = input.int(7, "Pivot Lookback", minval = 1, group = MSGROUP)
msbZScoreInput          = input.float(0.5, "MSB Momentum Z-Score", minval = 0.1, step = 0.1, group = MSGROUP)

// Visuals
modeInput               = input.string("Historical", "Display Mode", options = ["Historical", "Present"], group = VISUALSGROUP)

// Order Blocks
obCountInput            = input.int(10, "Max Active OBs", minval = 1, maxval = 100, group = OBGROUP)
extendBrokenInput       = input.bool(false, "Extend Broken OBs", group = OBGROUP)
hideOverlapsInput       = input.bool(false, "Hide Overlapping OBs", group = OBGROUP)

// Sessions
showSessionsInput       = input.bool(true, "Show Session Ranges", group = SESSIONSGROUP)

showLondonInput         = input.bool(false, "London Session", group = SESSIONSGROUP)
londonSessInput         = input.session("0800-1700:1234567", "London Time", group = SESSIONSGROUP)
londonColInput          = input.color(#3b82f6, "London Color", group = SESSIONSGROUP, inline = "London")

showNYInput             = input.bool(false, "New York Session", group = SESSIONSGROUP)
nySessInput             = input.session("1300-2200:1234567", "New York Time", group = SESSIONSGROUP)
nyColInput              = input.color(#f97316, "New York Color", group = SESSIONSGROUP, inline = "NY")

showTokyoInput          = input.bool(false, "Tokyo Session", group = SESSIONSGROUP)
tokyoSessInput          = input.session("0000-0900:1234567", "Tokyo Time", group = SESSIONSGROUP)
tokyoColInput           = input.color(#f43f5e, "Tokyo Color", group = SESSIONSGROUP, inline = "Tokyo")

showSydneyInput         = input.bool(false, "Sydney Session", group = SESSIONSGROUP)
sydneySessInput         = input.session("2100-0600:1234567", "Sydney Time", group = SESSIONSGROUP)
sydneyColInput          = input.color(#eab308, "Sydney Color", group = SESSIONSGROUP, inline = "Sydney")

//---------------------------------------------------------------------------------------------------------------------}
// Types & Collections
//---------------------------------------------------------------------------------------------------------------------{
type OrderBlock
    box        boxId
    label      labelId
    line       pocLineId
    float      qualityScore
    float      top
    float      bottom
    bool       isBull
    bool       isHPZ = false
    bool       mitigated = false
    int        mitigationBar = 0

var obArray = array.new<OrderBlock>()

//---------------------------------------------------------------------------------------------------------------------}
// User-Defined Functions
//---------------------------------------------------------------------------------------------------------------------{
var parsedDashboardPosition = switch dashboardPositionInput
    TOPRIGHT       => position.topright
    BOTTOMRIGHT    => position.bottomright
    BOTTOMLEFT     => position.bottomleft

var parsedDashboardSize     = switch dashboardSizeInput
    TINY            => size.tiny
    SMALL           => size.small
    NORMAL          => size.normal
    LARGE           => size.large
    HUGE            => size.huge

cell(table table, int column, int row, string data, color = #FFFFFF, align = text.alignright, color background = na, float height = 0) =>
    table.cell(column,row,data,textcolor = color, textsize = parsedDashboardSize, texthalign = align, bgcolor = background, height = height)

divider(table table, int row, int lastColumn) =>
    string rowDivider = '━━━━━━━━━━━━━━'
    table.mergecells(0,row,lastColumn,row)
    cell(table,0,row,rowDivider,align = text.aligncenter, height = 0.5, color = BORDERS)

drawSession(bool show, string sessStr, color col, string name) =>
    var box   sBox   = na
    var label sLabel = na
    var float sHigh  = na
    var float sLow   = na

    inSession = not na(time(timeframe.period, sessStr))
    isStart   = inSession and not inSession[1]

    if show and inSession
        if isStart
            sHigh  := high
            sLow   := low
            sBox   := box.new(barindex, sHigh, barindex, sLow, bordercolor = col, borderstyle = line.styledotted, bgcolor = color.new(col, 92))
            sLabel := label.new(barindex, sHigh, name, style = label.stylenone, textcolor = col, size = size.small, textalign = text.aligncenter)
        else
            sHigh := math.max(sHigh, high)
            sLow  := math.min(sLow, low)
            sBox.settop(sHigh)
            sBox.setbottom(sLow)
            sBox.setright(barindex)
            sLabel.setxy(math.round(math.avg(sBox.getleft(), barindex)), sHigh)
    float(na)

//---------------------------------------------------------------------------------------------------------------------}
// Core Calculations
//---------------------------------------------------------------------------------------------------------------------{
float priceChange = ta.change(close)
float avgChange   = ta.sma(priceChange, 50)
float stdChange   = ta.stdev(priceChange, 50)
float momentumZ   = (priceChange - avgChange) / stdChange
float volPercent  = ta.percentrank(volume, 100)

// Pivot Detection
float ph = ta.pivothigh(high, pivotLenInput, pivotLenInput)
float pl = ta.pivotlow(low, pivotLenInput, pivotLenInput)

var float lastPh = na
var float lastPl = na
var int lastPhIdx = na
var int lastPlIdx = na

if not na(ph)
    lastPh := ph
    lastPhIdx := barindex - pivotLenInput
if not na(pl)
    lastPl := pl
    lastPlIdx := barindex - pivotLenInput

// Session Range Logic
if showSessionsInput
    drawSession(showLondonInput, londonSessInput, londonColInput, "London")
    drawSession(showNYInput,     nySessInput,     nyColInput,     "New York")
    drawSession(showTokyoInput,  tokyoSessInput,  tokyoColInput,  "Tokyo")
    drawSession(showSydneyInput, sydneySessInput, sydneyColInput, "Sydney")

// Market Structure Breaks (MSB)
bool isMSBBull = close > lastPh and close[1] <= lastPh and momentumZ > msbZScoreInput
bool isMSBBear = close < lastPl and close[1] >= lastPl and momentumZ < -msbZScoreInput

if isMSBBull
    int mid = math.round(math.avg(lastPhIdx, barindex))
    line.new(lastPhIdx, lastPh, barindex, lastPh, color = color.new(BULLCOLOR, 0), width = 1)
    label.new(mid, lastPh, "MSB", color = color.new(BULLCOLOR, 100), textcolor = BULLCOLOR, style = label.stylelabeldown, size = size.small)

if isMSBBear
    int mid = math.round(math.avg(lastPlIdx, barindex))
    line.new(lastPlIdx, lastPl, barindex, lastPl, color = color.new(BEARCOLOR, 0), width = 1)
    label.new(mid, lastPl, "MSB", color = color.new(BEARCOLOR, 100), textcolor = BEARCOLOR, style = label.stylelabelup, size = size.small)

// Institutional Order Block Logic
if isMSBBull or isMSBBear
    int obIdx = 0
    for i = 1 to 10
        if (isMSBBull and close[i] < open[i]) or (isMSBBear and close[i] > open[i])
            obIdx := i
            break

    float obTop    = high[obIdx]
    float obBottom = low[obIdx]
    float obPOC    = math.avg(obTop, obBottom)

    // Quality Score (0-100)
    float score = math.min(100, (math.abs(momentumZ) * 20) + (volPercent * 0.5))
    bool isHPZ = score > 80

    color baseColor = isMSBBull ? BULLCOLOR : BEARCOLOR
    color hpzColor  = isMSBBull ? HPZBULL   : HPZBEAR

    box   obBox = na
    label obLabel = na
    line  obPoc = na

    // Overlap Logic
    bool overlapping = false
    if hideOverlapsInput and obArray.size() > 0
        for i = 0 to obArray.size() - 1
            OrderBlock o = obArray.get(i)
            if not o.mitigated and ((obTop <= o.top and obTop >= o.bottom) or (obBottom <= o.top and obBottom >= o.bottom))
                overlapping := true
                break

    if not overlapping
        if isHPZ
            obBox := box.new(barindex[obIdx], obTop, barindex + 15, obBottom, bordercolor = color.new(hpzColor, 40), bgcolor = color.new(hpzColor, 80), borderwidth = 1, borderstyle = line.styledashed)
            obPoc := line.new(barindex[obIdx], obPOC, barindex + 15, obPOC, color = color.new(hpzColor, 40), width = 1, style = line.styledashed)
        else
            obBox := box.new(barindex[obIdx], obTop, barindex + 15, obBottom, bordercolor = color.new(baseColor, 60), bgcolor = color.new(baseColor, 85), borderstyle = line.styledashed, borderwidth = 1)
            obPoc := line.new(barindex[obIdx], obPOC, barindex + 15, obPOC, color = color.new(baseColor, 70), style = line.styledashed, width = 1)

        obLabel := label.new(barindex + 18, obPOC, str.tostring(score, "#") + "%", color = color.new(baseColor, 100), textcolor = isHPZ ? hpzColor : baseColor, style = label.stylenone, size = size.tiny, textalign = text.alignleft)

        OrderBlock newOB = OrderBlock.new(obBox, obLabel, obPoc, score, obTop, obBottom, isMSBBull, isHPZ, false, 0)
        obArray.push(newOB)

    if isMSBBull
        lastPh := na
    else
        lastPl := na

// Dynamic Management
var float totalMitigated = 0.0
var float totalObs = 0.0

if (isMSBBull or isMSBBear)
    totalObs += 1

if obArray.size() > 0
    int mitCount = 0
    for i = obArray.size() - 1 to 0
        OrderBlock ob = obArray.get(i)
        if not ob.mitigated
            ob.boxId.setright(barindex + 15)
            ob.labelId.setx(barindex + 18)
            ob.pocLineId.setx2(barindex + 15)

            bool isMitigated = ob.isBull ? low < ob.bottom : high > ob.top
            if isMitigated
                ob.mitigated := true
                ob.mitigationBar := barindex
                totalMitigated += 1
                if modeInput == "Present"
                    ob.boxId.delete()
                    ob.labelId.delete()
                    ob.pocLineId.delete()
                else
                    ob.boxId.setbgcolor(color.new(NEUTRALCOLOR, 90))
                    ob.boxId.setbordercolor(color.new(NEUTRALCOLOR, 70))
                    ob.labelId.settextcolor(color.new(NEUTRALCOLOR, 50))
                    ob.pocLineId.setcolor(color.new(NEUTRALCOLOR, 80))

                    // Default to non-extended for broken blocks
                    ob.boxId.setright(barindex)
                    ob.labelId.setx(barindex)
                    ob.pocLineId.setx2(barindex)
        else
            // Already mitigated. Decide if we extend it.
            mitCount += 1
            bool shouldExtend = extendBrokenInput and mitCount <= 2 and (barindex - ob.mitigationBar <= 50)

            if modeInput == "Present"
                ob.boxId.delete()
                ob.labelId.delete()
                ob.pocLineId.delete()
            else if shouldExtend
                ob.boxId.setright(barindex + 15)
                ob.labelId.setx(barindex + 18)
                ob.pocLineId.setx2(barindex + 15)
            else
                ob.boxId.setright(ob.mitigationBar)
                ob.labelId.setx(ob.mitigationBar)
                ob.pocLineId.setx2(ob.mitigationBar)

    if obArray.size() > obCountInput
        OrderBlock oldOB = obArray.shift()
        oldOB.boxId.delete()
        oldOB.labelId.delete()
        oldOB.pocLineId.delete()

//---------------------------------------------------------------------------------------------------------------------}
// Visuals
//---------------------------------------------------------------------------------------------------------------------{
// Analytics Dashboard
var table table = table.new(parsedDashboardPosition, 2, 5
     , bgcolor          = BACKGROUND
     , borderwidth     = 0
     , framecolor      = BORDERS
     , framewidth      = 1
     , forceoverlay    = false)

if showStatsInput and barstate.islast
    float efficiency = totalObs > 0 ? (totalMitigated / totalObs) * 100 : 0
    int hpzs = 0
    if obArray.size() > 0
        for i = 0 to obArray.size() - 1
            if obArray.get(i).isHPZ and not obArray.get(i).mitigated
                hpzs += 1

    table.mergecells(0, 0, 1, 0)
    cell(table, 0, 0, 'Statistic Analysis', color = DATA, align = text.aligncenter)

    cell(table, 0, 0, 'Statistics', color = DATA, align = text.aligncenter)

    divider(table, 1, 1)

    cell(table, 0, 2, 'Reliability', color = HEADERS, align = text.alignleft)
    cell(table, 1, 2, str.tostring(efficiency, "#.#") + "%", color = efficiency > 50 ? BULLCOLOR : BEARCOLOR)

    divider(table, 3, 1)

    cell(table, 0, 4, 'HP-OB Count', color = HEADERS, align = text.alignleft)
    cell(table, 1, 4, str.tostring(hpzs), color = DATA)
```

---

**创建时间**: 2025-02-19
**最后更新**: 2025-02-19
