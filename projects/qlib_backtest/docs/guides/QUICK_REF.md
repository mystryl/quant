# Qlib 数据加载器与重采样 - 快速参考

## 一、数据加载器

### 结构要求
```
qlib_data/
├── calendars/
│   ├── 1min.txt
│   ├── 5min.txt
│   └── ...
└── instruments/
    ├── 1min/
    │   ├── $open/RB9999.XSGE.csv
    │   ├── $close/RB9999.XSGE.csv
    │   └── ...
    └── 5min/
        └── ...
```

### 文件格式
```csv
datetime,feature_name
2023-01-03 09:01:00,4069.0
2023-01-03 09:02:00,4056.0
...
```

## 二、重采样规则

| 字段 | 规则 |
|------|------|
| open | first |
| high | max |
| low | min |
| close | last |
| volume | sum |
| amount | sum |
| vwap | amount/volume |
| open_interest | last |

## 三、回测结果对比

| 频率 | 累计收益 | 年化收益 | 最大回撤 | 夏普比率 |
|------|----------|----------|----------|----------|
| 1min | 92.18% | 61.19% | 2.25% | 8.11 |
| 5min | 10.84% | 7.38% | 2.82% | 1.37 |
| 15min | -2.52% | -1.58% | 5.99% | -0.37 |
| 60min | -9.15% | -4.42% | 10.48% | -1.35 |

## 四、使用命令

```bash
# 重采样
python3 resample_data.py

# 多频率回测
python3 multi_freq_backtest.py

# 查看结果
ls -lh backtest_*.csv
```

## 五、关键发现

✅ **频率越低，效果越好**（均值回归策略）
✅ **1分钟表现最佳**：累计收益 92.18%，夏普比率 8.11
⚠️ **高频交易成本高**：需要考虑手续费和滑点

---

**详细文档**: 见 `QLIB_GUIDE.md`
