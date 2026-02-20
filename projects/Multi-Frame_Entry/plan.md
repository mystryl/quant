# 🎯 项目目标

构建一个：

> 多周期市场状态识别 + 条件入场模型 + Walk-forward 回测框架

数据：1min 主力连续合约  
框架：Qlib  
模型：RandomForest / XGBoost

---

# 🏗 总体工程结构

建议目录结构：

Multi-Frame_Entry/  
│  
├── data/  改为统一的数据目录 ./Quant/data
│   ├── raw/  
│   └── processed/  
│  
├── features/  
│   ├── base\_features.py  
│   ├── trend\_features.py  
│   └── micro\_features.py  
│  
├── labels/  
│   ├── trend\_label.py  
│   └── entry\_label.py  
│  
├── models/  
│   ├── trend\_model.py  
│   └── entry\_model.py  
│  
├── backtest/  
│   ├── walkforward.py  
│   └── strategy.py  
│  
└── main.py

---

# 🔵 阶段一：数据预处理模块

## 🎯 目标

-   清洗 1min 数据
    
-   构造多周期数据（5min / 15min / 60min / 1day）
    
-   对齐时间索引
    
-   转换为 Qlib 格式
    

---

## ClaudeCode 开发任务

### Task 1.1 数据清洗已完成

/Quant/K线数据库

---

### Task 1.2 多周期重采样

从 1min 构造：

-   5min
    
-   15min
    
-   60min
    

要求：

-   OHLC 正确聚合
    
-   成交量求和
    

输出：
Quant/data/qlib_data_multi_freq 
合约\_5min.parquet  
合约\_15min.parquet  
合约\_60min.parquet
合约\_1day.parquet

---

### Task 1.3 Qlib Dataset 构建

构建 Qlib 的 DataHandler：

使用现有的应该足够了。
    

---

# 🔵 阶段二：趋势状态标签构建

## 🎯 目标

构造三分类标签：

1 = 上涨趋势  
0 = 震荡  
\-1 = 下跌趋势

---

## 标签定义（重要）

以 60min 为中周期：

对每根t时刻的60min K线，计算未来10-20根K的收益率
future_return = (df['close'].shift(-20) - df['close']) / df['close']

按阈值贴标签
df['label'] = 0
df.loc[future_return > 0.003, 'label'] = 1   # 涨
df.loc[future_return < -0.003, 'label'] = -1 # 跌
    

注意：

-   标签必须 shift(-60)
    
-   严格防止未来函数污染
    

---

## ClaudeCode 任务

实现：

labels/trend\_label.py

输出字段：

trend\_label

---

# 🔵 阶段三：趋势特征工程

只使用 t 时刻及之前数据。

---

## 建议趋势特征（60min）

### 1️⃣ 斜率类

-   EMA60 slope
    
-   EMA20 slope
    
-   TWAP slope
    
-   线性回归斜率
    

---

### 2️⃣ 趋势强度

-   ADX(14)
    
-   ADX 变化率
    
-   ATR(14)
    
-   ATR / price
    

---

### 3️⃣ 结构类

-   TWAP5 / TWAP60 金叉死叉
    
-   close / EMA60 K线突破
    
-   高低点突破距离

-   多头/空头连续排列

-   均线排列
    

---

### 4️⃣ 波动率

-   rolling std
    
-   Parkinson 波动率
    

---

## ClaudeCode 任务

实现：

features/trend\_features.py

所有特征 shift(1) 避免未来函数污染（look-ahead bias）

---

# 🔵 阶段四：趋势模型训练

使用：

-   RandomForestClassifier
    
-   或 XGBoost
    

---

## 模型目标

输入：trend\_features  
输出：

P(up), P(range), P(down)

---

## 时间序列分割

禁止随机 split。

使用：

train: 2022-23  
valid: 2024  
test: 2025

或 rolling walk-forward。

---

## 输出

保存模型：

models/trend\_model.pkl

输出：

trend\_prob\_up  
trend\_prob\_down  
trend\_prob\_range

---

# 🔵 阶段五：入场模型（5min）

只在：

trend\_prob\_up > 0.6 → 允许做多  
trend\_prob\_down > 0.6 → 允许做空

---

## 入场标签定义

未来 15min：

最大涨幅 > 0.2%  
最大回撤 < 0.15%

满足 → 1  
否则 → 0

---

## 入场特征（5min）

-   RSI
    
-   MACD diff
    
-   布林带位置
    
-   VWAP 偏离
    
-   orderflow proxy（如果有）
    

---

## ClaudeCode 任务

实现：

features/micro\_features.py  
labels/entry\_label.py  
models/entry\_model.py

---

# 🔵 阶段六：策略层

构建策略逻辑：

if trend\_prob\_up > 0.6 and entry\_signal == 1:  
    open long  
  
if trend\_prob\_down > 0.6 and entry\_signal == 1:  
    open short

加入：

-   固定止损
    
-   ATR 止损
    
-   时间止损
    

---

实现：

backtest/strategy.py

---

# 🔵 阶段七：Walk-forward 回测

每 3 个月滚动训练：

train 1年 → test 3个月

输出：

-   年化收益
    
-   Sharpe
    
-   最大回撤
    
-   胜率
    
-   盈亏比
    

实现：

backtest/walkforward.py

---

# 🔵 阶段八：风险控制增强

后续可以加入：

-   概率加权仓位
    
-   regime 切换后仓位减半
    
-   高波动时降低仓位
    

---

# 🎯 ClaudeCode 执行顺序建议

让 ClaudeCode：

1.  先写数据管道
    
2.  再写标签
    
3.  再写特征
    
4.  再写趋势模型
    
5.  再写入场模型
    
6.  最后写策略
    

每一步都要求：

-   单元测试
    
-   防未来函数
    
-   输出可视化
    

---

# ⚠️ 关键风控点

1.  所有特征必须 shift(1)
    
2.  标签必须未来数据
    
3.  回测必须含手续费滑点
    
4.  禁止 shuffle
    
5.  必须时间分割
    

---

# 🚀 进一步优化（后期）

-   用 XGBoost 替代 RF
    
-   用概率做仓位控制
    
-   加入 HMM 做 regime
    
-   加入特征重要性稳定性分析