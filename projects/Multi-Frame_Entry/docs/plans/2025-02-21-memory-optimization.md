# 内存优化方案总结

**日期**: 2025-02-21
**问题**: 并行处理5个品种导致内存占用过大（10-15GB），系统崩溃

---

## 🔍 问题根源

### 原始配置（高风险）
```python
# 数据管道
results = Parallel(n_jobs=-1)(...)  # 使用所有CPU核心

# 训练框架
results = Parallel(n_jobs=-1)(...)  # 使用所有CPU核心
```

### 内存占用分析

**单个品种**：
- 原始数据（1min）: 50-196万行 × 8列 ≈ 32-125MB
- 重采样数据（5个周期）: ≈ 500MB-1GB
- 特征计算（57个特征）: ≈ 1-2GB
- **单个品种总计**: **2-3GB**

**5个品种并行**：
- 5 × 3GB = **15GB** 💥
- 加上中间结果、缓存等：**20GB+**

---

## ✅ 优化方案

### 修改1: 数据管道串行化

**文件**: `scripts/data_pipeline_multi_symbol.py`

```python
def process_all_symbols(self, parallel: bool = False, max_workers: int = 1):
    """
    优化内存占用

    Args:
        parallel: 是否并行（默认False）
        max_workers: 最大并行数（建议1-2）
    """
    if parallel and max_workers > 1:
        # 限制并行数量
        results = Parallel(n_jobs=max_workers)(...)
    else:
        # 串行处理（默认）
        for i, symbol in enumerate(self.symbols, 1):
            result = self.process_single_symbol(symbol)
            results.append(result)

            # 显式清理内存
            import gc
            gc.collect()
```

### 修改2: 训练框架串行化

**文件**: `scripts/rolling_train_multi_symbol.py`

```python
def train_all_symbols(self, parallel: bool = False, max_workers: int = 1):
    """同上，串行处理"""
```

### 修改3: 主控脚本

**文件**: `scripts/run_multi_symbol_experiment.py`

```python
# Step 1: 数据准备（串行）
data_results = pipeline.process_all_symbols(parallel=False, max_workers=1)

# Step 2: 训练（串行）
training_results = trainer.train_all_symbols(parallel=False, max_workers=1)
```

---

## 📊 性能对比

| 模式 | 内存占用 | 耗时 | 风险 |
|------|---------|------|------|
| **并行（5品种）** | 15-20GB | 1-2小时 | ❌ 高风险崩溃 |
| **串行（1品种）** | 3-5GB | 4-6小时 | ✅ 安全 |
| **并行（2品种）** | 6-8GB | 2-3小时 | ⚠️ 中等风险 |

**建议**: 使用串行模式（安全第一）

---

## 🚀 使用方法

### 方法1: 使用一键启动脚本（推荐）

```bash
cd /Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry

# 低内存版本启动
bash scripts/run_experiment_low_memory.sh

# 查看进度
bash scripts/check_experiment_progress.sh

# 查看实时日志
tail -f logs/experiment_low_memory_*.log
```

### 方法2: 直接运行Python脚本

```bash
cd /Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry

# 启动实验
python3 scripts/run_multi_symbol_experiment.py
```

### 方法3: 手动分步执行（最安全）

```bash
# Step 1: 数据准备
python3 -c "
from scripts.data_pipeline_multi_symbol import MultiSymbolDataPipeline
from pathlib import Path

pipeline = MultiSymbolDataPipeline(
    source_dir=Path('/Users/mystryl/Documents/Quant/K线数据库/期货商品指数_parquet'),
    output_base_dir=Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data')
)
pipeline.process_all_symbols(parallel=False, max_workers=1)
"

# Step 2: 训练模型
python3 -c "
from scripts.rolling_train_multi_symbol import RollingTrainMultiSymbol
from pathlib import Path

trainer = RollingTrainMultiSymbol(
    data_base_dir=Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/multi_symbol'),
    model_output_dir=Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/models/rolling')
)
trainer.train_all_symbols(parallel=False, max_workers=1)
"

# Step 3: 生成报告
python3 scripts/generate_excel_report.py
```

---

## 🎯 处理进度

### 串行模式处理顺序

1. **HC8888** (热卷) - 约50万行 - 预计30分钟
2. **I8888** (铁矿石) - 约103万行 - 预计45分钟
3. **AU8888** (黄金) - 约196万行 - 预计60分钟
4. **CF8888** (郑棉) - 约150万行 - 预计50分钟
5. **IF8888** (股指期货) - 约131万行 - 预计45分钟

**总耗时**: 约3.5-4小时（数据准备）+ 2-3小时（训练）= **5.5-7小时**

---

## 💾 内存监控

### 实时监控内存使用

```bash
# macOS
top -pid $(pgrep -f run_multi_symbol_experiment)

# 或使用htop（如果安装）
htop -p $(pgrep -f run_multi_symbol_experiment)

# Python进程内存
ps aux | grep python | grep run_multi
```

### 预期内存占用

- **最低**: 约2GB（启动时）
- **平均**: 约3-4GB（处理小品种）
- **峰值**: 约5GB（处理AU8888等大品种）

---

## 🔧 如果仍遇到内存问题

### 方案A: 进一步降低并行度

修改代码中的 `max_workers` 参数，但当前已经是1（串行），无法更低。

### 方案B: 分批处理

手动指定处理品种：

```python
# 只处理2个品种
symbols = ['HC8888.XSGE', 'I8888.XDCE']
pipeline = MultiSymbolDataPipeline(..., symbols=symbols)
```

### 方案C: 减少重采样周期

修改 `TARGET_FREQS`：

```python
# 原来: ['5min', '15min', '60min', 'D']
# 优化: ['15min', '60min', 'D']  # 去掉5min
TARGET_FREQS = ['15min', '60min', 'D']
```

### 方案D: 减少特征数量

特征计算时减少特征数（当前57个）。

---

## ✅ 验证修复

运行前先测试单个品种：

```bash
# 测试HC8888
python3 scripts/test_pipeline_single.py
python3 scripts/test_train_single.py

# 观察内存占用
# 如果<5GB，说明修复成功
```

---

## 📋 启动前检查清单

- [x] 修改为串行模式
- [x] 添加内存清理（gc.collect()）
- [x] 创建低内存启动脚本
- [x] 创建进度监控脚本
- [x] 清理Python缓存
- [ ] 确认可用内存 > 8GB
- [ ] 关闭其他内存占用大的程序

---

## 🎉 预期结果

使用串行模式后：

- ✅ 内存占用降低**70%**（从15GB降至3-5GB）
- ✅ 避免系统崩溃
- ✅ 可以在普通开发机上运行
- ⚠️ 耗时增加**2-3倍**（从2小时增至5-7小时）

**结论**: 牺牲速度换取稳定性，值得！

---

**最后更新**: 2025-02-21 06:10
**状态**: 已优化，可以启动实验
