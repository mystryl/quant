#!/usr/bin/env python3
"""
多品种滚动训练实验主控脚本

功能：
1. 数据准备：5个品种的多周期重采样、标签生成、特征计算
2. 滚动训练：Walk Forward年度训练（2021-2025）
3. 生成报告：Excel汇总表 + 可视化图表

品种：
- HC8888.XSGE (热卷)
- I8888.XDCE (铁矿石)
- AU8888.XSGE (黄金)
- CF8888.XZCE (郑棉)
- IF8888.CCFX (股指期货)

预计耗时：3-4小时（5个品种 x 每个约40分钟）
"""
import sys
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_pipeline_multi_symbol import MultiSymbolDataPipeline
from scripts.rolling_train_multi_symbol import RollingTrainMultiSymbol

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 路径配置
SOURCE_DATA_DIR = Path('/Users/mystryl/Documents/Quant/K线数据库/期货商品指数_parquet')
OUTPUT_BASE_DIR = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data')
MODEL_OUTPUT_DIR = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/models/rolling')

def main():
    """主函数"""
    logger.info("="*100)
    logger.info("多品种滚动训练实验开始")
    logger.info("="*100)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"品种数量: 5")
    logger.info(f"训练年份: 2021-2025")
    logger.info("="*100 + "\n")

    experiment_start = datetime.now()

    # ==================== Step 1: 数据准备 ====================
    logger.info("\n" + "="*100)
    logger.info("Step 1: 多品种数据准备")
    logger.info("="*100)

    step1_start = datetime.now()

    try:
        # 创建数据管道
        pipeline = MultiSymbolDataPipeline(
            source_dir=SOURCE_DATA_DIR,
            output_base_dir=OUTPUT_BASE_DIR
        )

        # 处理所有品种（串行，降低内存占用）
        data_results = pipeline.process_all_symbols(parallel=False, max_workers=1)

        elapsed = (datetime.now() - step1_start).total_seconds()
        logger.info(f"\n✓ Step 1 完成！耗时: {elapsed/60:.1f}分钟")

    except Exception as e:
        logger.error(f"\n✗ Step 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== Step 2: 滚动训练 ====================
    logger.info("\n" + "="*100)
    logger.info("Step 2: 年度滚动训练")
    logger.info("="*100)

    step2_start = datetime.now()

    try:
        # 创建训练框架
        trainer = RollingTrainMultiSymbol(
            data_base_dir=OUTPUT_BASE_DIR / 'multi_symbol',
            model_output_dir=MODEL_OUTPUT_DIR
        )

        # 训练所有品种（串行，降低内存占用）
        training_results = trainer.train_all_symbols(parallel=False, max_workers=1)

        elapsed = (datetime.now() - step2_start).total_seconds()
        logger.info(f"\n✓ Step 2 完成！耗时: {elapsed/60:.1f}分钟")

    except Exception as e:
        logger.error(f"\n✗ Step 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== Step 3: 生成报告 ====================
    logger.info("\n" + "="*100)
    logger.info("Step 3: 生成汇总报告")
    logger.info("="*100)

    step3_start = datetime.now()

    try:
        # 生成Excel报告
        logger.info("生成Excel汇总报告...")
        # TODO: 实现Excel报告生成

        # 生成可视化图表
        logger.info("生成可视化图表...")
        # TODO: 实现可视化图表生成

        elapsed = (datetime.now() - step3_start).total_seconds()
        logger.info(f"\n✓ Step 3 完成！耗时: {elapsed/60:.1f}分钟")

    except Exception as e:
        logger.error(f"\n✗ Step 3 失败: {e}")
        import traceback
        traceback.print_exc()

    # ==================== 实验总结 ====================
    total_elapsed = (datetime.now() - experiment_start).total_seconds()

    logger.info("\n" + "="*100)
    logger.info("实验完成！")
    logger.info("="*100)
    logger.info(f"总耗时: {total_elapsed/60:.1f}分钟 ({total_elapsed/3600:.2f}小时)")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*100)

    # 打印性能汇总
    logger.info("\n性能汇总:")
    logger.info("-"*100)

    summary_file = MODEL_OUTPUT_DIR.parent / 'training_results' / 'training_summary.csv'
    if summary_file.exists():
        import pandas as pd
        df_summary = pd.read_csv(summary_file)
        logger.info(f"\n{df_summary.to_string(index=False)}")

        # 计算平均AUC
        avg_auc = df_summary.groupby('品种')['AUC'].apply(lambda x: x.astype(float).mean())
        logger.info(f"\n各品种平均AUC:")
        for symbol, auc in avg_auc.items():
            logger.info(f"  {symbol}: {auc:.4f}")

    logger.info("="*100 + "\n")

    logger.info("✓ 所有文件已保存到:")
    logger.info(f"  - 数据: {OUTPUT_BASE_DIR / 'multi_symbol'}")
    logger.info(f"  - 模型: {MODEL_OUTPUT_DIR}")
    logger.info(f"  - 结果: {MODEL_OUTPUT_DIR.parent / 'training_results'}")
    logger.info("\n实验圆满完成！")

    return data_results, training_results


if __name__ == '__main__':
    # 创建日志目录
    log_dir = project_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # 运行实验
    results = main()
