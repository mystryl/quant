#!/bin/bash
# 批量测试剩余品种

echo "=========================================="
echo "批量测试剩余品种"
echo "=========================================="
echo ""
echo "待测品种:"
echo "  - AU8888.XSGE (黄金)"
echo "  - CF8888.XZCE (郑棉)"
echo "  - IF8888.CCFX (股指期货)"
echo ""
echo "预计耗时: 3-4分钟"
echo ""

cd /Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry

# 启动批量测试
python3 scripts/test_symbol_batch.py --symbols AU8888.XSGE CF8888.XZCE IF8888.CCFX

echo ""
echo "=========================================="
echo "批量测试完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  数据: data/multi_symbol/"
echo "  模型: models/rolling/"
echo "  报告: models/training_results/training_summary.csv"
echo ""
