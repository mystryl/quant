#!/bin/bash
# 实验进度监控脚本

echo "=========================================="
echo "多品种滚动训练实验 - 进度监控"
echo "=========================================="
echo ""

# 查找最新的日志文件
LOG_FILE=$(ls -t logs/experiment_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到实验日志文件"
    echo ""
    echo "请先启动实验："
    echo "  python3 scripts/run_multi_symbol_experiment.py"
    exit 1
fi

echo "📋 日志文件: $LOG_FILE"
echo "📅 最后更新: $(date -r $LOG_FILE '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查进程
echo "🔍 进程状态:"
if pgrep -f "run_multi_symbol_experiment" > /dev/null; then
    echo "  ✅ 实验正在运行"
    PID=$(pgrep -f "run_multi_symbol_experiment")
    echo "  PID: $PID"
    ps -p $PID -o pid,ppid,cmd,%mem,%cpu,etime
else
    echo "  ⚠️  实验未运行或已完成"
fi
echo ""

# 显示最新进度
echo "📊 最新进度 (最后30行):"
echo "=========================================="
tail -30 $LOG_FILE
echo ""

# 检查输出文件
echo "📁 输出文件:"
echo "=========================================="
if [ -d "data/multi_symbol" ]; then
    echo "✅ 数据目录:"
    for dir in data/multi_symbol/*/; do
        if [ -d "$dir" ]; then
            symbol=$(basename "$dir")
            count=$(find "$dir" -type f | wc -l)
            echo "  $symbol: $count 个文件"
        fi
    done
else
    echo "  ⏳ 数据目录尚未创建"
fi
echo ""

if [ -d "models/rolling" ]; then
    echo "✅ 模型目录:"
    ls -lh models/rolling/*.pkl 2>/dev/null | wc -l | xargs echo "  模型文件数:"
else
    echo "  ⏳ 模型目录尚未创建"
fi
echo ""

# 估算进度
echo "⏱️  进度估算:"
echo "=========================================="
if grep -q "Step 1: 多品种数据准备" $LOG_FILE; then
    if grep -q "Step 2: 年度滚动训练" $LOG_FILE; then
        if grep -q "Step 3: 生成汇总报告" $LOG_FILE; then
            echo "  📊 Step 3: 生成汇总报告 (进行中)"
        else
            echo "  🔄 Step 2: 年度滚动训练 (进行中)"
        fi
    else
        echo "  📦 Step 1: 多品种数据准备 (进行中)"
    fi
else
    echo "  ⏳ 实验尚未开始"
fi
echo ""

echo "💡 提示:"
echo "  - 持续监控: watch -n 10 $0"
echo "  - 查看完整日志: tail -f $LOG_FILE"
echo "  - 停止实验: pkill -f run_multi_symbol_experiment"
echo ""
