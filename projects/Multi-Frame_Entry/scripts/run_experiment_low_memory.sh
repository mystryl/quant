#!/bin/bash
# 多品种实验 - 低内存版本

echo "=========================================="
echo "多品种滚动训练实验 - 低内存版本"
echo "=========================================="
echo ""
echo "配置:"
echo "  - 处理模式: 串行（每次1个品种）"
echo "  - 预计耗时: 4-6小时"
echo "  - 内存占用: 约3-5GB（峰值）"
echo ""

# 创建日志目录
mkdir -p logs

# 清理Python缓存
echo "清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 启动实验（低内存模式）
echo "启动实验..."
cd /Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry
nohup python3 scripts/run_multi_symbol_experiment.py > logs/experiment_low_memory_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "实验已启动，PID: $PID"
echo ""
echo "查看日志:"
echo "  tail -f logs/experiment_low_memory_*.log"
echo ""
echo "查看进度:"
echo "  bash scripts/check_experiment_progress.sh"
echo ""
echo "停止实验:"
echo "  kill $PID"
echo ""
