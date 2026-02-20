#!/usr/bin/env python3
"""
测试单个品种的训练框架

用于验证训练流程是否正常工作
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rolling_train_multi_symbol import RollingTrainMultiSymbol

def main():
    """测试单个品种单一年份"""
    # 路径配置
    project_root = Path(__file__).parent.parent
    data_base_dir = project_root / 'data' / 'multi_symbol'
    model_output_dir = project_root / 'models' / 'rolling'

    # 验证路径
    if not data_base_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_base_dir}. 请先运行数据准备脚本")

    model_output_dir.mkdir(parents=True, exist_ok=True)
    (model_output_dir.parent / 'training_results').mkdir(parents=True, exist_ok=True)

    # 创建训练框架（测试I8888铁矿石）
    trainer = RollingTrainMultiSymbol(
        data_base_dir=data_base_dir,
        model_output_dir=model_output_dir,
        symbols=['I8888.XDCE']  # 测试铁矿石
    )

    # 训练（不并行）
    all_results = trainer.train_all_symbols(parallel=False)

    print("\n测试完成！")
    return all_results

if __name__ == '__main__':
    main()
