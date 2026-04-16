# 多因子量化分析系统 - 系统设计文档

## 1. 系统概述

### 1.1 目标
设计并实现一个多因子量化分析系统,能够:
- 给定因子,根据回测数据计算 IC、IR 等性能指标
- 评估因子可靠性,考虑周期对齐
- 支持多种策略场景(看涨、看跌、波动率策略)
- 避免未来函数,仅在性能评估时使用未来收益率

### 1.2 核心设计原则
1. **避免未来函数**: 因子计算时仅使用历史数据,未来收益率仅在性能评估时使用
2. **符合交易规则**: 采用 Qlib 的 T+1 到 T+2 Label 设计,符合中国交易规则
3. **灵活的周期对齐**: 支持不同因子的周期特性,自动或手动调整
4. **多策略场景**: 同一因子在不同市场环境下进行评估
5. **可复用数据层**: 利用现有数据基础设施,避免重复开发

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户接口层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  CLI 命令行  │  │  Python API  │  │   可视化报告生成器    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        业务逻辑层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │因子计算引擎  │  │性能评估引擎  │  │   策略场景分析器     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │周期对齐模块  │  │可靠性评估器  │  │   报告生成模块       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        数据访问层                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  统一数据接口 (复用 qlib_backtest 的 SmartDataProvider) │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        数据存储层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Parquet 数据  │  │ Qlib 缓存    │  │ 因子结果库   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块设计

#### 2.2.1 数据访问层 (data/)

**目的**: 复用 qlib_backtest 的 SmartDataProvider

**复用策略**:
```python
# 复用统一数据提供者
from qlib_backtest.scripts.data.unified_data_provider import SmartDataProvider

class FactorDataProvider:
    """因子数据提供者 - 包装 SmartDataProvider"""
    def __init__(self, data_root):
        self.provider = SmartDataProvider(data_root)

    def get_factor_data(self, instrument, fields, start_date, end_date):
        """获取因子计算所需的基础数据"""
        return self.provider.get_data(
            instrument=instrument,
            fields=fields,
            start_date=start_date,
            end_date=end_date
        )
```

**新增功能**:
- 因子数据验证(检查 NaN、异常值)
- 因子标准化(z-score、min-max)
- 因子中性化(行业中性、市值中性)

#### 2.2.2 因子计算引擎 (FactorEngine)

**职责**:
- 因子加载和注册
- 因子计算(支持自定义因子)
- 因子缓存管理
- **未来函数检测** - 静态分析验证因子表达式

**核心类**:

```python
class FactorManager:
    """因子管理器"""

    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.factors = {}
        self.factor_cache = {}
        self.expr_parser = FactorExpressionParser()  # 表达式解析器

    def register_factor(self, name, factor_func):
        """注册自定义因子"""
        # 验证因子表达式无未来函数
        if isinstance(factor_func, str):
            self.expr_parser.validate_no_future_functions(factor_func)
        self.factors[name] = factor_func

    def calculate_factor(self, name, instruments, start_date, end_date):
        """计算因子"""
        factor_func = self.factors.get(name)
        if not factor_func:
            raise ValueError(f"Factor {name} not found")

        return factor_func(
            self.data_provider,
            instruments,
            start_date,
            end_date
        )
```

**未来函数检测机制**:

```python
class FactorExpressionParser:
    """因子表达式解析器 - 检测未来函数"""

    # 禁止的未来函数模式
    FUTURE_PATTERNS = [
        r'Ref\s*\(\s*\$?\w+\s*,\s*-\s*\d+\s*\)',  # Ref($close, -N) where N>0
        r'\$close\[\s*-\s*\d+\s*\]',               # $close[-N] where N>0
        r'Roll\s*\(\s*\$?\w+\s*,\s*\d+\s*\)',      # Roll with positive offset
    ]

    def validate_no_future_functions(self, expression):
        """
        静态分析验证因子表达式不包含未来引用

        Args:
            expression: 因子表达式字符串

        Raises:
            ValueError: 如果检测到未来函数
        """
        for pattern in self.FUTURE_PATTERNS:
            if re.search(pattern, expression):
                raise ValueError(
                    f"表达式包含未来函数: {expression}\n"
                    f"检测到模式: {pattern}\n"
                    "因子计算只能使用历史数据,不能引用未来数据。"
                )

        # 检查是否有负索引
        if self._has_negative_index(expression):
            raise ValueError(
                f"表达式包含负索引(未来引用): {expression}\n"
                "请确保所有索引都是非负整数。"
            )

        return True

    def _has_negative_index(self, expression):
        """检测是否有负索引"""
        # 查找类似 [ -N ] 或 [-N] 的模式
        negative_index_pattern = r'\[\s*-\s*\d+\s*\]'
        return bool(re.search(negative_index_pattern, expression))
```

#### 2.2.3 周期对齐模块 (CycleAligner)

**职责**:
- 自动检测因子周期特性
- 对齐因子数据和未来收益率
- 支持提前/延后对齐

**对齐策略**:

1. **默认对齐 (Qlib 方式)**: T+1 到 T+2
   ```python
   # T日因子 -> 预测 T+2 的收益率
   label = Ref($close, -2) / Ref($close, -1) - 1
   ```

2. **灵活对齐**: 支持自定义偏移量
   ```python
   def align_factor_with_returns(factor_df, price_df, shift=2):
       """
       对齐因子和未来收益率
       shift=2: Qlib 默认 (T+1 to T+2)
       shift=1: jqfactor_analyzer 方式 (T to T+1)
       """
       returns = (price_df.shift(-shift) / price_df.shift(-(shift-1)) - 1)
       return factor_df, returns
   ```

3. **周期检测**: 自动选择最优对齐方式
   ```python
   def detect_best_alignment(factor_df, price_df):
       """
       计算不同对齐周期的 IC,选择最优
       """
       ic_values = {}
       for shift in [1, 2, 3, 5]:
           aligned_factor, aligned_returns = align_factor_with_returns(
               factor_df, price_df, shift
           )
           ic = calc_ic(aligned_factor, aligned_returns)
           ic_values[shift] = ic.mean()

       return max(ic_values, key=ic_values.get)
   ```

#### 2.2.4 性能评估引擎 (PerformanceEvaluator)

**职责**:
- 计算 IC(信息系数)
- 计算 IR(信息比率)
- 计算 ICIR(IC 信息比率)
- 计算多空收益和其他风险指标

**实现**:

```python
from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return

class PerformanceEvaluator:
    """性能评估器 - 统一命名规范"""

    def __init__(self, use_log_return=False):
        self.use_log_return = use_log_return

    def calculate_all(self, pred, label):
        """计算所有指标"""
        # IC 和 Rank IC
        ic, ric = calc_ic(pred, label)

        # 多空收益(基于 IC/IR)
        long_short_r, long_avg_r = calc_long_short_return(
            pred, label, quantile=0.2
        )

        # ICIR
        ic_mean = ic.mean()
        ic_std = ic.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0

        # Rank ICIR
        ric_mean = ric.mean()
        ric_std = ric.std()
        ricir = ric_mean / ric_std if ric_std > 0 else 0

        return {
            'ic': ic,
            'rank_ic': ric,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'rank_ic_mean': ric_mean,
            'rank_ic_std': ric_std,
            'rank_icir': ricir,
            'long_short_return': long_short_r,
            'average_return': long_avg_r,
        }

    def calculate_log_return(self, price_df, shift=2):
        """
        计算对数收益率
        适合长期持有策略或大波动场景
        """
        return np.log(price_df.shift(-shift) / price_df.shift(-(shift-1)))
```

**对数收益率使用场景**:
- 长期投资策略(多日/多周持有)
- 高波动性资产(如加密货币、期货)
- 需要计算复合收益率时

**未来收益计算**:
采用 Qlib 的 T+1 到 T+2 设计:
```python
forward_return = Ref($close, -2) / Ref($close, -1) - 1
```

**设计理由**:
- 符合中国 T+1 交易规则
- T 日收盘时无法买入,只能 T+1 买入,T+2 卖出
- 避免未来数据泄露

#### 2.2.5 策略场景分析器 (StrategyAnalyzer)

**职责**:
- 评估因子在不同市场环境下的表现
- 支持看涨、看跌、波动率策略场景

**场景类型**:

1. **看涨策略 (Bull Market)**:
   - 选取因子值最高的股票做多
   - 评估标准:正向 IC、正向多空收益

2. **看跌策略 (Bear Market)**:
   - 选取因子值最低的股票做空(或反向做多)
   - 评估标准:负向 IC、负向多空收益

3. **波动率策略 (Volatility Strategy)**:
   - 根据市场波动率调整仓位
   - 高波动时降低仓位,低波动时提高仓位

**实现**:

```python
class StrategyAnalyzer:
    """策略场景分析器"""

    def analyze_bull_strategy(self, factor_df, returns_df, top_pct=0.2):
        """
        分析看涨策略
        选取因子值最高的 top_pct 比例股票
        """
        # 每日选择 top_pct 的股票
        selected = factor_df.groupby(level='datetime').apply(
            lambda x: x.nlargest(int(len(x) * top_pct))
        )

        # 计算策略收益
        strategy_returns = returns_df.loc[selected.index].groupby(
            level='datetime'
        ).mean()

        return {
            'strategy_returns': strategy_returns,
            'total_return': (1 + strategy_returns).prod() - 1,
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
        }

    def analyze_bear_strategy(self, factor_df, returns_df, bottom_pct=0.2):
        """
        分析看跌策略
        选取因子值最低的 bottom_pct 比例股票
        """
        selected = factor_df.groupby(level='datetime').apply(
            lambda x: x.nsmallest(int(len(x) * bottom_pct))
        )

        strategy_returns = returns_df.loc[selected.index].groupby(
            level='datetime'
        ).mean()

        return {
            'strategy_returns': strategy_returns,
            'total_return': (1 + strategy_returns).prod() - 1,
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
        }

    def analyze_volatility_strategy(self, factor_df, returns_df, price_df):
        """
        分析波动率策略
        根据市场波动率调整仓位
        """
        # 计算市场波动率(所有股票收益率的标准差)
        market_vol = returns_df.groupby(level='datetime').std()

        # 计算动态仓位(波动率越低,仓位越高)
        position = 1 / (1 + market_vol)  # 简单的反向关系

        # 应用到策略
        bull_result = self.analyze_bull_strategy(factor_df, returns_df)
        adjusted_returns = bull_result['strategy_returns'] * position

        return {
            'adjusted_returns': adjusted_returns,
            'total_return': (1 + adjusted_returns).prod() - 1,
            'sharpe_ratio': adjusted_returns.mean() / adjusted_returns.std() * np.sqrt(252),
        }
```

**其他场景**:
- **牛熊市**: 按市场趋势分组
- **行业轮动**: 按行业分组分析
- **市值分组**: 大中小盘股分组

#### 2.2.6 可靠性评估器 (ReliabilityAssessor)

**职责**:
- 综合评估因子可靠性
- 输出可读的评估报告
- **支持可配置的权重系统**

**评估指标体系**:

| 指标 | 说明 | 优秀标准 | 警告阈值 |
|------|------|----------|----------|
| IC均值 | 因子预测能力 | > 0.03 | 0.01 ~ 0.03 |
| ICIR | IC 稳定性 | > 0.5 | 0.2 ~ 0.5 |
| Rank IC | 非线性预测能力 | > 0.04 | 0.02 ~ 0.04 |
| 多空收益 | 实际交易效果 | > 5% (年化) | 0 ~ 5% |
| 胜率 | 交易成功率 | > 55% | 50 ~ 55% |
| 最大回撤 | 风险控制 | < 15% | 15 ~ 25% |

**权重配置方案**:

```python
# 默认权重配置(基于学术研究和实践经验)
DEFAULT_WEIGHTS = {
    'ic_stability': 0.40,    # IC 稳定性 (ICIR)
    'ic_absolute': 0.20,      # IC 绝对值
    'ir': 0.20,               # 信息比率
    'long_short_return': 0.10, # 多空收益
    'win_rate': 0.10,         # 胜率
}

# 保守型策略权重(更注重稳定性)
CONSERVATIVE_WEIGHTS = {
    'ic_stability': 0.50,    # IC 稳定性权重更高
    'ic_absolute': 0.15,
    'ir': 0.20,
    'long_short_return': 0.10,
    'win_rate': 0.05,
}

# 激进型策略权重(更注重收益)
AGGRESSIVE_WEIGHTS = {
    'ic_stability': 0.30,
    'ic_absolute': 0.20,
    'ir': 0.15,
    'long_short_return': 0.25,  # 多空收益权重更高
    'win_rate': 0.10,
}
```

**权重理论依据**:

1. **IC 稳定性 (40%)**:
   - 参考文献: "Factor Evaluation using Information Coefficient" - Grinold & Kahn
   - 理由: ICIR 是因子稳定性的核心指标,直接影响因子的长期可靠性

2. **IC 绝对值 (20%)**:
   - 参考文献: "Active Portfolio Management" - Grinold & Kahn
   - 理由: IC 均值反映因子的预测能力,是因子的基本属性

3. **IR (20%)**:
   - 参考文献: "Information Ratio and Performance" - Sharpe
   - 理由: IR 综合考虑收益和风险,是实际交易效果的关键指标

4. **多空收益 (10%)**:
   - 参考文献: "Long-Short Equity Strategies" - Jacobs & Levy
   - 理由: 实际交易效果,反映因子的经济价值

5. **胜率 (10%)**:
   - 参考文献: "Win Rate and Trading Performance" - Kaufman
   - 理由: 胜率影响策略的心理承受能力和资金管理

**实现**:

```python
class ReliabilityEvaluator:
    """可靠性评估器"""

    def __init__(self, weights=None, thresholds=None):
        """
        Args:
            weights: 自定义权重配置
            thresholds: 自定义阈值配置
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.thresholds = thresholds or {
            'ic_mean': {'excellent': 0.03, 'good': 0.01},
            'icir': {'excellent': 0.5, 'good': 0.2},
            'rank_ic_mean': {'excellent': 0.04, 'good': 0.02},
            'annual_return': {'excellent': 0.05, 'good': 0.0},
            'win_rate': {'excellent': 0.55, 'good': 0.50},
            'max_drawdown': {'excellent': 0.15, 'good': 0.25},
        }

    def evaluate(self, metrics, scenario_results):
        """
        综合评估因子可靠性

        Returns:
            dict: 包含评分、可靠性等级和建议
        """
        scores = {}

        # 1. IC 稳定性评估 (40%)
        scores['ic_stability'] = self._evaluate_metric(
            metrics['icir'],
            self.thresholds['icir'],
            higher_better=True
        ) * self.weights['ic_stability']

        # 2. IC 绝对值评估 (20%)
        scores['ic_absolute'] = self._evaluate_metric(
            abs(metrics['ic_mean']),
            self.thresholds['ic_mean'],
            higher_better=True
        ) * self.weights['ic_absolute']

        # 3. IR 评估 (20%)
        scores['ir'] = self._evaluate_metric(
            metrics['long_short_return'].mean() / metrics['long_short_return'].std(),
            {'excellent': 1.5, 'good': 1.0},
            higher_better=True
        ) * self.weights['ir']

        # 4. 多空收益评估 (10%)
        annual_return = scenario_results['bull']['total_return']
        scores['long_short_return'] = self._evaluate_metric(
            annual_return,
            self.thresholds['annual_return'],
            higher_better=True
        ) * self.weights['long_short_return']

        # 5. 胜率评估 (10%)
        win_rate = (scenario_results['bull']['strategy_returns'] > 0).mean()
        scores['win_rate'] = self._evaluate_metric(
            win_rate,
            self.thresholds['win_rate'],
            higher_better=True
        ) * self.weights['win_rate']

        # 综合评分
        total_score = sum(scores.values())

        # 可靠性判断
        if total_score >= 0.8:
            reliability = 'A+'
            recommendation = "该因子表现优秀,建议重点使用。"
        elif total_score >= 0.7:
            reliability = 'A'
            recommendation = "该因子表现良好,建议使用。"
        elif total_score >= 0.6:
            reliability = 'B'
            recommendation = "该因子表现一般,建议谨慎使用。"
        elif total_score >= 0.5:
            reliability = 'C'
            recommendation = "该因子表现较差,建议优化后使用。"
        else:
            reliability = 'D'
            recommendation = "该因子不可靠,不建议使用。"

        return {
            'scores': scores,
            'total_score': total_score,
            'reliability': reliability,
            'recommendation': recommendation,
        }

    def _evaluate_metric(self, value, threshold, higher_better=True):
        """评估单个指标"""
        if higher_better:
            if value >= threshold['excellent']:
                return 1.0
            elif value >= threshold['good']:
                return 0.6
            else:
                return 0.2
        else:
            if value <= threshold['excellent']:
                return 1.0
            elif value <= threshold['good']:
                return 0.6
            else:
                return 0.2
```

**因子相关性分析** (新增功能):

```python
class FactorCorrelationAnalyzer:
    """因子相关性分析器"""

    def analyze_correlation(self, factor_dict):
        """
        分析因子之间的相关性

        Args:
            factor_dict: {factor_name: factor_df}

        Returns:
            dict: 相关性矩阵和高度相关因子对
        """
        # 计算相关性矩阵
        corr_matrix = self._calculate_correlation_matrix(factor_dict)

        # 找出高度相关的因子对
        high_corr_pairs = self._find_high_correlation(corr_matrix, threshold=0.7)

        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'recommendation': self._generate_recommendation(high_corr_pairs)
        }

    def _calculate_correlation_matrix(self, factor_dict):
        """计算因子相关性矩阵"""
        import pandas as pd

        # 合并所有因子
        combined = pd.concat(factor_dict, axis=1)
        return combined.corr()

    def _find_high_correlation(self, corr_matrix, threshold=0.7):
        """找出高度相关的因子对"""
        high_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append({
                        'factor1': corr_matrix.index[i],
                        'factor2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        return high_corr

    def _generate_recommendation(self, high_corr_pairs):
        """生成去重建议"""
        if not high_corr_pairs:
            return "所有因子相关性较低,可以组合使用。"

        recommendation = "以下因子对高度相关,建议谨慎组合:\n"
        for pair in high_corr_pairs:
            recommendation += (
                f"- {pair['factor1']} 和 {pair['factor2']}: "
                f"相关系数 {pair['correlation']:.2f}\n"
            )
        return recommendation
```

#### 2.2.7 报告生成器 (ReportGenerator)

**职责**:
- 生成 IC/IR 时间序列图
- 生成因子性能对比图
- 生成可读的分析报告

**报告内容**:
1. **执行摘要**: 可靠性等级、关键指标
2. **IC 分析**: IC 时间序列、IC 分布、ICIR
3. **IR 分析**: 多空收益、夏普比率、最大回撤
4. **策略场景**: 看涨/看跌/波动率环境表现
5. **周期分析**: 最佳周期、周期对齐效果
6. **稳定性分析**: 时间稳定性、市场环境稳定性
7. **相关性分析**: 与其他因子的相关性(新增)
8. **建议**: 是否使用、如何优化

## 3. 数据流

```
原始数据 → DataLoader → FactorEngine → CycleAligner
                                      ↓
                              PerformanceEvaluator
                                      ↓
                              StrategyAnalyzer
                                      ↓
                              ReliabilityAssessor
                                      ↓
                              ReportGenerator
```

## 4. 数据复用方案

### 4.1 可复用的数据组件

#### 4.1.1 SmartDataProvider
**位置**: `/Users/mystryl/Documents/Quant/projects/qlib_backtest/scripts/data/unified_data_provider.py`

**复用方式**:
```python
from qlib_backtest.scripts.data import SmartDataProvider

# 初始化数据提供者
provider = SmartDataProvider(
    data_dir="/path/to/parquet/data",
    cache_dir="/path/to/qlib/cache"
)

# 获取数据
data = provider.get_data(
    instrument="SH600000",
    fields=["open", "high", "low", "close", "volume"],
    start_date="2020-01-01",
    end_date="2020-12-31"
)
```

**优点**:
- 已实现 Parquet 和 Qlib 格式的智能路由
- 支持透明缓存
- 统一的数据访问接口
- 支持多市场、多频率

#### 4.1.2 数据配置
**位置**: `/Users/mystryl/Documents/Quant/projects/qlib_backtest/scripts/data/config.py`

**复用方式**:
```python
from qlib_backtest.scripts.data.config import (
    PARQUET_DATA_DIR,
    STANDARD_FIELDS,
    QLIB_FIELD_PREFIX
)
```

### 4.2 数据层架构

```
多因子分析系统
       ↓
统一数据接口 (新建)
       ↓
SmartDataProvider (复用)
       ↓
数据存储层
```

## 5. 项目目录结构

```
multi_factor_analyzer/
├── task_plan.md              # 任务计划
├── findings.md               # 研究发现
├── progress.md               # 进度记录
├── SYSTEM_DESIGN.md          # 系统设计文档(本文件)
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── setup.py                  # 安装脚本
├── src/                      # 源代码
│   ├── __init__.py
│   ├── cli/                  # 命令行接口
│   │   ├── __init__.py
│   │   └── main.py
│   ├── core/                 # 核心模块
│   │   ├── __init__.py
│   │   ├── factor_engine.py      # 因子计算引擎
│   │   ├── cycle_aligner.py      # 周期对齐模块
│   │   ├── performance_eval.py   # 性能评估引擎
│   │   ├── strategy_analyzer.py  # 策略场景分析
│   │   └── reliability.py        # 可靠性评估
│   ├── data/                 # 数据层
│   │   ├── __init__.py
│   │   ├── provider.py           # 数据提供者
│   │   ├── loader.py             # 数据加载器
│   │   └── validator.py          # 数据验证器
│   └── report/               # 报告生成
│       ├── __init__.py
│       ├── generator.py          # 报告生成器
│       └── visualizer.py         # 可视化
├── tests/                    # 测试
│   ├── __init__.py
│   ├── test_factor_engine.py
│   ├── test_performance.py
│   ├── test_future_guard.py  # 未来函数检测测试
│   └── test_integration.py
├── examples/                 # 示例
│   ├── simple_factor.py
│   ├── complex_factor.py
│   └── batch_analysis.py
└── docs/                     # 文档
    ├── API.md
    ├── USER_GUIDE.md
    └── DEVELOPMENT.md
```

## 6. 关键技术决策

### 6.1 已确定的决策

| 决策 | 理由 |
|------|------|
| 采用 Qlib 的 Label 设计 | 符合中国 T+1 交易规则,避免未来数据泄露 |
| 混合使用 IC/IR 和多空收益 | IC/IR 衡量预测能力,多空收益衡量实际交易效果 |
| 支持多种周期对齐方式 | 不同因子可能有不同的周期特性 |
| 引入策略场景分析 | 同一因子在不同市场环境下表现不同 |
| 复用 SmartDataProvider | 避免重复开发,利用现有数据基础设施 |
| 实现未来函数静态检测 | 在计算前验证因子表达式,避免未来数据泄露 |
| 支持可配置的评估权重 | 不同策略类型可能需要不同的评估重点 |
| 增加因子相关性分析 | 避免多因子组合时的高度相关 |

### 6.2 待确认的问题

1. **对数收益率的使用**
   - **问题**: 是否需要支持对数收益率?
   - **建议**: 默认使用简单收益率,可选支持对数收益率
   - **理由**: 简单收益率更直观,对数收益率在数学上更优雅(收益可加)

2. **Label 周期的灵活性**
   - **问题**: 是否支持 T+N 任意周期?
   - **建议**: 支持,但默认为 T+1 到 T+2
   - **理由**: 不同策略可能需要不同的持有期

3. **自动周期检测的算法**
   - **问题**: 如何自动检测因子的最佳周期?
   - **建议**: 基于网格搜索和 IC 最优化
   - **理由**: 简单直接,易于理解和验证

## 7. 实现优先级

### Phase 1: 核心功能 (必须)
1. 数据加载模块(复用 SmartDataProvider)
2. 因子计算引擎
   - 基础因子计算
   - 未来函数静态检测
3. IC/IR 计算
4. 基础报告生成

### Phase 2: 高级功能 (重要)
1. 周期对齐模块
2. 策略场景分析
3. 可靠性评估
   - 可配置权重系统
   - 因子相关性分析
4. 可视化增强

### Phase 3: 优化功能 (可选)
1. 自动周期检测
2. 批量因子分析
3. Web 界面
4. 性能优化

## 8. 风险和挑战

1. **未来函数检测**: 需要严格的验证机制和静态分析
2. **周期对齐的准确性**: 自动检测可能不够准确
3. **数据质量**: 依赖数据源的质量
4. **性能**: 大规模因子分析可能需要优化
5. **权重配置**: 不同策略可能需要不同的权重配置
6. **因子相关性**: 多因子组合时需要考虑相关性问题

## 9. 下一步行动

1. ✅ 完成系统设计文档
2. ⏳ 创建项目目录结构
3. ⏳ 实现数据加载模块
4. ⏳ 实现因子计算引擎(含未来函数检测)
5. ⏳ 实现性能评估引擎
6. ⏳ 编写测试用例(含未来函数检测测试)
7. ⏳ 生成示例报告
