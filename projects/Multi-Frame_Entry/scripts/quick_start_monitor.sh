#!/bin/bash
# 实时趋势监控 - 快速使用指南

echo "=========================================="
echo "实时趋势监控系统 - 快速使用指南"
echo "=========================================="
echo ""

# 检查Python环境
echo "1. 检查环境..."
if ! command -v python &> /dev/null; then
    echo "   ✗ Python未安装"
    exit 1
fi
echo "   ✓ Python已安装"

# 检查依赖
echo ""
echo "2. 检查依赖..."
python -c "import pandas, numpy, efinance" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ 所有依赖已安装"
else
    echo "   ⚠ 部分依赖缺失，正在安装..."
    pip install pandas numpy efinance akshare pyarrow
fi

# 显示使用方法
echo ""
echo "=========================================="
echo "使用方法："
echo "=========================================="
echo ""
echo "【监控单个品种】"
echo "  python scripts/realtime_monitor.py --symbol HC0    # 热卷"
echo "  python scripts/realtime_monitor.py --symbol RB0    # 螺纹钢"
echo "  python scripts/realtime_monitor.py --symbol I0     # 铁矿石"
echo "  python scripts/realtime_monitor.py --symbol AU0    # 黄金"
echo "  python scripts/realtime_monitor.py --symbol CF0    # 郑棉"
echo ""
echo "【监控所有品种】"
echo "  python scripts/realtime_monitor.py --all"
echo ""
echo "【自定义参数】"
echo "  python scripts/realtime_monitor.py --symbol HC0 --bars 150 --lookback 15"
echo "                                      # 获取150根K线，检测最近15根"
echo ""

# 交互式选择
echo "=========================================="
echo "快速开始："
echo "=========================================="
echo ""
echo "请选择操作："
echo "  1) 监控热卷 (HC0)"
echo "  2) 监控螺纹钢 (RB0)"
echo "  3) 监控铁矿石 (I0)"
echo "  4) 监控黄金 (AU0)"
echo "  5) 监控郑棉 (CF0)"
echo "  6) 监控所有品种"
echo "  7) 运行测试"
echo "  8) 查看帮助"
echo "  0) 退出"
echo ""
read -p "请输入选项 (0-8): " choice

case $choice in
    1)
        echo ""
        echo "正在监控热卷..."
        python scripts/realtime_monitor.py --symbol HC0
        ;;
    2)
        echo ""
        echo "正在监控螺纹钢..."
        python scripts/realtime_monitor.py --symbol RB0
        ;;
    3)
        echo ""
        echo "正在监控铁矿石..."
        python scripts/realtime_monitor.py --symbol I0
        ;;
    4)
        echo ""
        echo "正在监控黄金..."
        python scripts/realtime_monitor.py --symbol AU0
        ;;
    5)
        echo ""
        echo "正在监控郑棉..."
        python scripts/realtime_monitor.py --symbol CF0
        ;;
    6)
        echo ""
        echo "正在监控所有品种..."
        python scripts/realtime_monitor.py --all
        ;;
    7)
        echo ""
        echo "运行测试..."
        python scripts/test_realtime_monitor.py
        ;;
    8)
        echo ""
        python scripts/realtime_monitor.py --help
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac
