#!/usr/bin/env python3
"""
测试单个品种的数据管道

用于验证数据管道是否正常工作
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_pipeline_multi_symbol import MultiSymbolDataPipeline

def main():
    """测试单个品种"""
    # 路径配置
    project_root = Path(__file__).parent.parent
    source_dir = Path('/Users/mystryl/Documents/Quant/K线数据库/期货商品指数_parquet')
    output_base_dir = project_root / 'data'

    # 验证路径
    if not source_dir.exists():
        raise FileNotFoundError(f"源数据目录不存在: {source_dir}")

    output_base_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据管道（测试I8888铁矿石）
    pipeline = MultiSymbolDataPipeline(
        source_dir=source_dir,
        output_base_dir=output_base_dir,
        symbols=['I8888.XDCE']  # 测试铁矿石
    )

    # 处理（不并行）
    results = pipeline.process_all_symbols(parallel=False)

    print("\n测试完成！")
    return results

if __name__ == '__main__':
    main()
