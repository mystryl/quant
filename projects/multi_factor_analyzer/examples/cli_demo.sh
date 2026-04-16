#!/bin/bash
# CLI 使用示例脚本

echo "=================================="
echo "多因子量化分析系统 - CLI 使用示例"
echo "=================================="
echo ""

# 1. 显示帮助
echo "1. 显示帮助信息"
echo "命令: python -m src.cli.main --help"
python -m src.cli.main --help
echo ""
echo "按 Enter 继续..."
read

# 2. 验证因子表达式
echo "=================================="
echo "2. 验证因子表达式"
echo "=================================="
echo ""
echo "2.1 验证有效的动量因子"
echo "命令: python -m src.cli.main validate \"\$close / Ref(\$close, 20) - 1\""
python -m src.cli.main validate "\$close / Ref(\$close, 20) - 1"
echo ""
echo "按 Enter 继续..."
read

echo ""
echo "2.2 验证波动率因子"
echo "命令: python -m src.cli.main validate \"Std(\$close, 20) / Mean(\$close, 20)\""
python -m src.cli.main validate "Std(\$close, 20) / Mean(\$close, 20)"
echo ""
echo "按 Enter 继续..."
read

echo ""
echo "2.3 测试未来函数检测（应该报错）"
echo "命令: python -m src.cli.main validate \"Ref(\$close, -5)\""
python -m src.cli.main validate "Ref(\$close, -5)"
echo ""
echo "按 Enter 继续..."
read

# 3. 分析单个因子（使用模拟数据）
echo "=================================="
echo "3. 分析单个因子"
echo "=================================="
echo ""
echo "注意：以下示例需要真实的数据支持"
echo "这里只是演示命令格式"
echo ""
echo "3.1 分析动量因子"
echo "命令示例:"
echo "python -m src.cli.main analyze \\"
echo "  --factor \"\$close / Ref(\$close, 20) - 1\" \\"
echo "  --instruments examples/instruments.txt \\"
echo "  --start 2020-01-01 \\"
echo "  --end 2020-12-31"
echo ""
echo "按 Enter 继续..."
read

echo ""
echo "3.2 分析特定策略（只看涨策略）"
echo "命令示例:"
echo "python -m src.cli.main analyze \\"
echo "  --factor MA20 \\"
echo "  --instruments examples/instruments.txt \\"
echo "  --start 2020-01-01 \\"
echo "  --end 2020-12-31 \\"
echo "  --strategy bull \\"
echo "  --output ma20_analysis.json"
echo ""
echo "按 Enter 继续..."
read

# 4. 批量分析
echo "=================================="
echo "4. 批量分析多个因子"
echo "=================================="
echo ""
echo "4.1 使用配置文件批量分析"
echo "命令示例:"
echo "python -m src.cli.main batch --config examples/factors_config.yaml"
echo ""
echo "配置文件内容 (examples/factors_config.yaml):"
cat examples/factors_config.yaml
echo ""
echo "按 Enter 继续..."
read

echo ""
echo "4.2 批量分析并指定输出目录"
echo "命令示例:"
echo "python -m src.cli.main batch \\"
echo "  --config examples/factors_config.yaml \\"
echo "  --output output/batch_results"
echo ""
echo "按 Enter 继续..."
read

# 5. 生成报告
echo "=================================="
echo "5. 生成分析报告"
echo "=================================="
echo ""
echo "5.1 生成 HTML 报告"
echo "命令示例:"
echo "python -m src.cli.main report \\"
echo "  --input output/batch_results \\"
echo "  --output report.html \\"
echo "  --title \"因子分析报告\""
echo ""
echo "按 Enter 继续..."
read

echo ""
echo "5.2 生成 JSON 报告"
echo "命令示例:"
echo "python -m src.cli.main report \\"
echo "  --input output/batch_results \\"
echo "  --output report.json \\"
echo "  --format json"
echo ""
echo "按 Enter 继续..."
read

# 6. 高级用法
echo "=================================="
echo "6. 高级用法"
echo "=================================="
echo ""
echo "6.1 禁用缓存"
echo "python -m src.cli.main analyze \\"
echo "  --factor MA20 \\"
echo "  --instruments examples/instruments.txt \\"
echo "  --start 2020-01-01 \\"
echo "  --end 2020-12-31 \\"
echo "  --no-cache"
echo ""
echo "按 Enter 继续..."
read

echo ""
echo "6.2 自定义参数"
echo "python -m src.cli.main analyze \\"
echo "  --factor MA20 \\"
echo "  --instruments examples/instruments.txt \\"
echo "  --start 2020-01-01 \\"
echo "  --end 2020-12-31 \\"
echo "  --quantile 0.3 \\"
echo "  --top-pct 0.1 \\"
echo "  --strategy long_short"
echo ""
echo "按 Enter 继续..."
read

echo ""
echo "6.3 并行批量分析"
echo "python -m src.cli.main batch \\"
echo "  --config examples/factors_config.yaml \\"
echo "  --parallel 4"
echo ""
echo "按 Enter 继续..."
read

# 7. 总结
echo "=================================="
echo "7. 常用命令总结"
echo "=================================="
echo ""
echo "7.1 验证表达式"
echo "python -m src.cli.main validate \"<表达式>\""
echo ""
echo "7.2 分析单个因子"
echo "python -m src.cli.main analyze -f \"<因子>\" -i <股票> -s <开始日期> -e <结束日期>"
echo ""
echo "7.3 批量分析"
echo "python -m src.cli.main batch --config <配置文件>"
echo ""
echo "7.4 生成报告"
echo "python -m src.cli.main report -i <输入目录> -o <输出文件>"
echo ""
echo "=================================="
echo "演示完成！"
echo "=================================="
echo ""
echo "更多详细信息请参考: docs/CLI_GUIDE.md"
