# 多因子量化分析系统 - 系统架构设计

## 1. 系统概述

### 1.1 设计目标
设计一个多因子量化分析系统，能够：
- 给定因子，根据回测数据计算 IC、IR 等性能指标
- 评估因子可靠性，考虑周期对齐
- 支持多种策略场景（看涨、看跌、波动率策略）
- 避免未来函数，仅在性能评估时使用未来收益率

### 1.2 核心原则
1. **避免未来函数**: 因子计算只使用历史数据，未来收益率仅用于性能评估
2. **周期对齐**: 支持不同周期的因子，自动检测和对齐
3. **多策略场景**: 同一因子在不同市场环境下的性能分析
4. **可靠性评估**: 综合多种指标判断因子可靠性

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户接口层 (UI Layer)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │  因子上传    │  │  参数配置    │  │  报告查看    │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   业务逻辑层 (Business Logic)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 因子管理模块   │  │ 分析引擎模块  │  │ 报告生成模块   │       │
│  │ - 因子加载     │  │ - IC/IR计算  │  │ - 报告模板     │       │
│  │ - 因子预处理   │  │ - 多空收益   │  │ - 可视化      │       │
│  │ - 周期检测     │  │ - 策略分析   │  │ - 导出        │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    数据层 (Data Layer)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 数据源模块    │  │ 数据缓存     │  │ 历史数据     │       │
│  │ - K线数据    │  │ - 因子缓存   │  │ - 分析结果   │       │
│  │ - 基本面数据  │  │ - 结果缓存   │  │ - 历史报告   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    存储层 (Storage Layer)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 文件系统      │  │ 数据库       │  │ 配置文件      │       │
│  │ - CSV/HDF5   │  │ - SQLite    │  │ - YAML       │       │
│  │ - Feather    │  │ - PostgreSQL│  │ - JSON       │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块划分

#### 2.2.1 因子管理模块 (Factor Manager)
**职责**:
- 加载用户提供的因子数据
- 因子预处理（标准化、去极值、中性化）
- 因子周期检测和自动对齐
- 因子验证（检查未来函数、缺失值等）

**核心功能**:
```python
class FactorManager:
    def load_factor(self, factor_data: pd.DataFrame) -> pd.DataFrame
    def preprocess_factor(self, factor: pd.DataFrame) -> pd.DataFrame
    def detect_period(self, factor: pd.DataFrame) -> int
    def align_period(self, factor: pd.DataFrame, target_period: int) -> pd.DataFrame
    def validate_factor(self, factor: pd.DataFrame) -> ValidationResult
    def check_future_function(self, factor: pd.DataFrame) -> bool
```

#### 2.2.2 分析引擎模块 (Analysis Engine)
**职责**:
- 计算各类性能指标（IC、IR、多空收益、胜率等）
- 支持多种策略场景分析
- 因子可靠性评估
- 周期敏感性分析

**核心功能**:
```python
class AnalysisEngine:
    def calculate_ic(self, factor: pd.DataFrame, returns: pd.DataFrame) -> ICResult
    def calculate_ir(self, factor: pd.DataFrame, returns: pd.DataFrame) -> IRResult
    def calculate_long_short_return(self, factor: pd.DataFrame, returns: pd.DataFrame) -> LSResult
    def evaluate_reliability(self, metrics: Metrics) -> ReliabilityResult
    def analyze_by_strategy(self, factor: pd.DataFrame, strategy: Strategy) -> StrategyResult
    def period_sensitivity_analysis(self, factor: pd.DataFrame) -> SensitivityResult
```

#### 2.2.3 数据源模块 (Data Source)
**职责**:
- 提供市场数据（K线、基本面等）
- 计算未来收益率（仅用于评估，不暴露给因子计算）
- 数据缓存管理
- 支持多种数据格式

**核心功能**:
```python
class DataSource:
    def get_price_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame
    def calculate_forward_return(self, prices: pd.DataFrame, period: int = 2) -> pd.DataFrame
    def get_benchmark_data(self, benchmark: str) -> pd.DataFrame
    def cache_data(self, key: str, data: pd.DataFrame) -> None
    def load_cached_data(self, key: str) -> Optional[pd.DataFrame]
```

#### 2.2.4 报告生成模块 (Report Generator)
**职责**:
- 生成因子分析报告
- 可视化各类指标
- 导出多种格式（HTML、PDF、Excel）
- 报告模板管理

**核心功能**:
```python
class ReportGenerator:
    def generate_report(self, analysis_result: AnalysisResult) -> Report
    def visualize_ic(self, ic_series: pd.Series) -> Figure
    def visualize_ir(self, ir_result: IRResult) -> Figure
    def visualize_long_short_return(self, ls_result: LSResult) -> Figure
    def export_report(self, report: Report, format: str, path: str) -> None
```

---

## 3. 数据流设计

### 3.1 主数据流

```
1. 用户上传因子数据
   ↓
2. FactorManager.load_factor() - 加载因子
   ↓
3. FactorManager.validate_factor() - 验证因子（检查未来函数）
   ↓
4. FactorManager.preprocess_factor() - 预处理（标准化、去极值、中性化）
   ↓
5. FactorManager.detect_period() - 检测因子周期
   ↓
6. FactorManager.align_period() - 对齐周期（如需要）
   ↓
7. DataSource.calculate_forward_return() - 计算未来收益率（T+1到T+2）
   ↓
8. AnalysisEngine.calculate_ic() - 计算 IC/ICIR
   ↓
9. AnalysisEngine.calculate_ir() - 计算 IR
   ↓
10. AnalysisEngine.calculate_long_short_return() - 计算多空收益
   ↓
11. AnalysisEngine.analyze_by_strategy() - 多策略场景分析
   ↓
12. AnalysisEngine.evaluate_reliability() - 评估可靠性
   ↓
13. ReportGenerator.generate_report() - 生成报告
   ↓
14. 用户查看报告
```

### 3.2 关键数据格式

#### 3.2.1 因子数据格式
```python
# MultiIndex: (datetime, instrument)
# Columns: [factor_value]
                      factor_value
datetime    instrument
2020-01-01  000001.SZ    0.5234
            000002.SZ   -0.1234
            600000.SH    0.8756
...
```

#### 3.2.2 未来收益率格式
```python
# MultiIndex: (datetime, instrument)
# Columns: [forward_return]
                      forward_return
datetime    instrument
2020-01-01  000001.SZ       0.0234
            000002.SZ      -0.0156
            600000.SH       0.0321
...
```

#### 3.2.3 IC 结果格式
```python
{
    "ic_series": pd.Series,  # 每日 IC
    "mean_ic": float,         # IC 均值
    "std_ic": float,          # IC 标准差
    "ir": float,             # ICIR (IC 均值 / IC 标准差)
    "rank_ic_series": pd.Series,  # 每日 Rank IC
    "mean_rank_ic": float,
    "std_rank_ic": float,
    "rank_ir": float
}
```

---

## 4. 因子评估指标体系

### 4.1 核心指标

| 指标 | 定义 | 计算方式 | 评估标准 |
|------|------|---------|---------|
| **IC (Information Coefficient)** | 因子值与未来收益率的相关系数 | Pearson 相关系数 | \|IC\| > 0.05 为良好 |
| **Rank IC** | 因子值与未来收益率的秩相关系数 | Spearman 秩相关系数 | \|Rank IC\| > 0.05 为良好 |
| **ICIR (IC Information Ratio)** | IC 的信息比率，衡量 IC 稳定性 | IC 均值 / IC 标准差 | \> 0.5 为良好 |
| **IR (Information Ratio)** | 超额收益的信息比率 | 超额收益均值 / 超额收益标准差 | \> 0.5 为良好 |
| **多空收益** | 买入高分位股票、卖出低分位股票的收益 | (Top收益 - Bottom收益) / 2 | 正值且显著为良好 |
| **胜率** | 多空策略盈利的比例 | 盈利天数 / 总天数 | \> 50% 为良好 |
| **最大回撤** | 策略最大损失幅度 | 从峰值到谷底的跌幅 | \-10% 以内为良好 |

### 4.2 策略场景指标

#### 4.2.1 看涨策略
- 多头收益：买入高因子值股票的收益
- IC_up：市场上涨时的 IC 表现
- 胜率_up：市场上涨时的胜率

#### 4.2.2 看跌策略
- 空头收益：卖出低因子值股票的收益（对冲后）
- IC_down：市场下跌时的 IC 表现
- 胜率_down：市场下跌时的胜率

#### 4.2.3 波动率升高策略
- 高波动期表现：市场波动率升高时的因子表现
- Vol_IC：按波动率分组的 IC
- 稳定性测试：不同波动率环境下的因子稳定性

### 4.3 可靠性评估标准

| 评级 | 条件 |
|------|------|
| **优秀** | \|IC\| > 0.05, ICIR > 0.5, 多空收益 > 2%, 胜率 > 55% |
| **良好** | \|IC\| > 0.03, ICIR > 0.3, 多空收益 > 1%, 胜率 > 52% |
| **一般** | \|IC\| > 0.02, ICIR > 0.2, 多空收益 > 0.5%, 胜率 > 50% |
| **较差** | 不满足"一般"条件 |
| **不可用** | \|IC\| < 0.01 或 ICIR < 0.1 |

---

## 5. 周期对齐机制

### 5.1 周期检测

**目标**: 自动检测因子的周期特性

**方法**:
1. 计算因子在不同周期下的 IC
2. 找到 IC 最大的周期
3. 检查周期一致性

**实现**:
```python
def detect_period(factor: pd.DataFrame, returns: pd.DataFrame, max_period: int = 20) -> int:
    """
    检测因子的最佳周期
    """
    period_scores = {}
    for period in range(1, max_period + 1):
        shifted_returns = returns.groupby(level='instrument').shift(-period)
        ic = factor.corrwith(shifted_returns)
        period_scores[period] = abs(ic.mean())

    best_period = max(period_scores.items(), key=lambda x: x[1])[0]
    return best_period
```

### 5.2 周期对齐

**场景**:
- 因子计算使用 N 日数据（如 20 日动量）
- 未来收益率计算使用 M 日数据（如 T+1 到 T+2）
- 需要对齐确保逻辑一致

**方法**:
1. 检测因子周期
2. 对齐未来收益率计算周期
3. 可选：前向或后向对齐

**实现**:
```python
def align_period(factor: pd.DataFrame, target_period: int, direction: str = 'forward') -> pd.DataFrame:
    """
    对齐因子周期
    direction: 'forward' 或 'backward'
    """
    if direction == 'forward':
        # 前向对齐：将因子前移，保持 T+1 到 T+2 的收益率
        aligned_factor = factor.groupby(level='instrument').shift(-target_period)
    elif direction == 'backward':
        # 后向对齐：保持因子，调整收益率计算
        aligned_factor = factor.copy()

    return aligned_factor
```

### 5.3 对数收益率处理

**决策**: 提供对数收益率选项

**原因**:
- 对数收益率在连续复利场景下更精确
- 对数收益率具有可加性：R_total = R_1 + R_2 + ... + R_n
- 简单收益率适合短周期、单次投资场景

**实现**:
```python
def calculate_return(prices: pd.DataFrame, use_log: bool = False) -> pd.DataFrame:
    if use_log:
        # 对数收益率：log(P_t+1 / P_t)
        returns = np.log(prices / prices.groupby(level='instrument').shift(1))
    else:
        # 简单收益率：(P_t+1 - P_t) / P_t
        returns = prices / prices.groupby(level='instrument').shift(1) - 1

    return returns
```

---

## 6. 策略场景分析设计

### 6.1 市场环境分类

```python
class MarketRegime:
    BULL = "bull"       # 看涨市场
    BEAR = "bear"       # 看跌市场
    HIGH_VOL = "high_volatility"   # 高波动市场
    LOW_VOL = "low_volatility"     # 低波动市场
```

### 6.2 策略分析引擎

```python
class StrategyAnalyzer:
    def analyze_by_market_regime(self, factor: pd.DataFrame,
                                 returns: pd.DataFrame,
                                 market_index: pd.Series,
                                 regime: MarketRegime) -> dict:
        """
        在特定市场环境下分析因子表现
        """
        # 1. 识别市场环境
        if regime == MarketRegime.BULL:
            mask = market_index > market_index.quantile(0.6)
        elif regime == MarketRegime.BEAR:
            mask = market_index < market_index.quantile(0.4)
        elif regime == MarketRegime.HIGH_VOL:
            vol = market_index.pct_change().rolling(20).std()
            mask = vol > vol.quantile(0.6)
        elif regime == MarketRegime.LOW_VOL:
            vol = market_index.pct_change().rolling(20).std()
            mask = vol < vol.quantile(0.4)

        # 2. 筛选数据
        factor_filtered = factor[mask.index.intersection(factor.index)]
        returns_filtered = returns[mask.index.intersection(returns.index)]

        # 3. 计算指标
        ic_result = self.calculate_ic(factor_filtered, returns_filtered)
        ls_result = self.calculate_long_short_return(factor_filtered, returns_filtered)

        return {
            "regime": regime,
            "ic": ic_result,
            "long_short_return": ls_result,
            "sample_size": len(factor_filtered)
        }
```

---

## 7. 避免未来函数的实现

### 7.1 核心原则

**严格分离**:
- **因子计算层**: 只使用历史数据，绝对不能使用未来数据
- **性能评估层**: 可以使用未来收益率（T+1 到 T+2）

### 7.2 数据流控制

```python
class DataFlowController:
    def __init__(self):
        self.factor_data = None  # 因子数据（历史）
        self.future_returns = None  # 未来收益率（仅用于评估）

    def load_factor(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        加载因子数据，确保不包含未来信息
        """
        # 验证：检查是否包含未来函数
        if self._check_future_function(factor_data):
            raise ValueError("因子包含未来函数！")

        self.factor_data = factor_data
        return self.factor_data

    def calculate_future_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        计算未来收益率，仅用于性能评估
        不暴露给因子计算逻辑
        """
        # Qlib 方式：Ref($close, -2) / Ref($close, -1) - 1
        # 即 T+1 到 T+2 的收益率
        self.future_returns = (
            prices.groupby(level='instrument').shift(-2) /
            prices.groupby(level='instrument').shift(-1) - 1
        )
        return self.future_returns

    def _check_future_function(self, factor_data: pd.DataFrame) -> bool:
        """
        检查因子是否包含未来函数
        """
        # 1. 检查因子是否使用了未来数据
        # 2. 检查因子是否包含 T 日的未来信息
        # 3. 可以通过元数据标记或代码静态分析
        return False  # 实际实现需要更复杂的逻辑
```

### 7.3 API 设计

```python
# 用户 API - 只暴露安全的接口
class FactorAnalyzer:
    def analyze_factor(self, factor: pd.DataFrame, prices: pd.DataFrame) -> dict:
        """
        分析因子性能

        参数:
            factor: 因子数据（MultiIndex: datetime, instrument）
            prices: 价格数据（MultiIndex: datetime, instrument）

        返回:
            分析结果字典
        """
        # 1. 验证因子（检查未来函数）
        self._validate_factor(factor)

        # 2. 预处理因子
        factor_preprocessed = self._preprocess_factor(factor)

        # 3. 计算未来收益率（内部计算，用户不可见）
        future_returns = self._calculate_future_returns(prices)

        # 4. 计算性能指标
        ic_result = self._calculate_ic(factor_preprocessed, future_returns)
        ir_result = self._calculate_ir(factor_preprocessed, future_returns)

        return {
            "ic": ic_result,
            "ir": ir_result,
            "reliability": self._evaluate_reliability(ic_result, ir_result)
        }
```

---

## 8. 项目结构

```
multi_factor_analyzer/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── factor_manager.py      # 因子管理模块
│   │   ├── analysis_engine.py     # 分析引擎
│   │   ├── data_source.py         # 数据源
│   │   └── report_generator.py    # 报告生成器
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── ic_calculator.py       # IC 计算
│   │   ├── ir_calculator.py       # IR 计算
│   │   └── long_short.py          # 多空收益计算
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── market_regime.py       # 市场环境分类
│   │   └── strategy_analyzer.py   # 策略分析
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # 预处理工具
│   │   ├── alignment.py          # 周期对齐
│   │   └── validators.py          # 验证工具
│   └── config/
│       ├── __init__.py
│       └── settings.py           # 配置文件
├── tests/
│   ├── test_factor_manager.py
│   ├── test_analysis_engine.py
│   ├── test_metrics.py
│   └── test_strategies.py
├── examples/
│   ├── basic_usage.py
│   ├── advanced_usage.py
│   └── custom_factor.py
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── user_guide.md
├── requirements.txt
├── setup.py
└── README.md
```

---

## 9. 技术栈

| 类别 | 技术选型 | 原因 |
|------|---------|------|
| **核心语言** | Python 3.9+ | 量化分析标准语言，生态丰富 |
| **数据处理** | pandas, numpy | 高效数据操作 |
| **性能优化** | numba, cython | 加速计算密集型任务 |
| **可视化** | matplotlib, plotly | 静态和交互式图表 |
| **存储** | HDF5, Feather, SQLite | 高效数据存储和缓存 |
| **报告生成** | jinja2, weasyprint | 模板化报告生成 |
| **测试** | pytest, pytest-cov | 单元测试和覆盖率 |

---

## 10. 待确认问题

1. **Label 计算**: 采用 Qlib 的 T+1 到 T+2，还是 jqfactor_analyzer 的 T 到 T+N？
   - **建议**: 采用 Qlib 方式，更符合中国 T+1 交易规则

2. **对数收益率**: 是否默认使用对数收益率？
   - **建议**: 提供参数让用户选择，默认简单收益率

3. **周期对齐方向**: 前向对齐还是后向对齐？
   - **建议**: 默认后向对齐（不移动因子），提供前向对齐选项

4. **数据源**: 是否接入实时数据，还是只支持本地数据？
   - **建议**: 支持本地数据 + 提供接口接入外部数据源

5. **报告格式**: HTML、PDF、Excel 哪种优先？
   - **建议**: HTML 优先（可交互），支持 PDF 和 Excel 导出

---

## 11. 开发计划

### Phase 1: 基础框架（2 周）
- 项目结构搭建
- 因子管理模块基础功能
- 数据源模块基础功能
- 单元测试框架

### Phase 2: 核心计算（2 周）
- IC/IR 计算模块
- 多空收益计算模块
- 周期对齐模块
- 性能优化

### Phase 3: 策略分析（1 周）
- 市场环境分类
- 策略场景分析
- 可靠性评估

### Phase 4: 报告生成（1 周）
- 报告模板设计
- 可视化模块
- 多格式导出

### Phase 5: 测试与优化（1 周）
- 集成测试
- 性能优化
- 文档完善

**总计**: 约 7 周
