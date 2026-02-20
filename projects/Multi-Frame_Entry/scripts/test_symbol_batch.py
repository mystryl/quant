#!/usr/bin/env python3
"""
通用品种测试脚本

支持：
1. 单个品种测试
2. 批量测试多个品种
3. 自动生成汇总报告

用法：
    python3 scripts/test_symbol_batch.py --symbol I8888.XDCE
    python3 scripts/test_symbol_batch.py --symbols I8888.XDCE AU8888.XSGE CF8888.XZCE IF8888.CCFX
"""
import sys
import argparse
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 路径配置
SOURCE_DATA_DIR = Path('/Users/mystryl/Documents/Quant/K线数据库/期货商品指数_parquet')
OUTPUT_BASE_DIR = project_root / 'data'
MODEL_OUTPUT_DIR = project_root / 'models' / 'rolling'

# 品种配置
SYMBOL_CONFIG = {
    'HC8888.XSGE': {'name': '热卷'},
    'I8888.XDCE': {'name': '铁矿石'},
    'AU8888.XSGE': {'name': '黄金'},
    'CF8888.XZCE': {'name': '郑棉'},
    'IF8888.CCFX': {'name': '股指期货'}
}


def test_single_symbol(symbol):
    """
    测试单个品种的完整流程

    Args:
        symbol: 品种代码

    Returns:
        (data_result, training_result)
    """
    logger.info("\n" + "="*100)
    logger.info(f"开始测试品种: {symbol} ({SYMBOL_CONFIG.get(symbol, {}).get('name', 'Unknown')})")
    logger.info("="*100)

    start_time = datetime.now()

    # Step 1: 数据准备
    logger.info("\n" + "-"*100)
    logger.info("Step 1: 数据准备")
    logger.info("-"*100)

    try:
        pipeline = MultiSymbolDataPipeline(
            source_dir=SOURCE_DATA_DIR,
            output_base_dir=OUTPUT_BASE_DIR,
            symbols=[symbol]
        )

        data_results = pipeline.process_all_symbols(parallel=False, max_workers=1)

        if not data_results or not data_results[0]['success']:
            logger.error(f"数据准备失败: {symbol}")
            return None, None

        logger.info(f"✓ 数据准备完成")

    except Exception as e:
        logger.error(f"数据准备异常: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # Step 2: 模型训练
    logger.info("\n" + "-"*100)
    logger.info("Step 2: 模型训练")
    logger.info("-"*100)

    try:
        data_base_dir = OUTPUT_BASE_DIR / 'multi_symbol'
        model_output_dir = project_root / 'models' / 'rolling'

        trainer = RollingTrainMultiSymbol(
            data_base_dir=data_base_dir,
            model_output_dir=model_output_dir,
            symbols=[symbol]
        )

        training_results = trainer.train_all_symbols(parallel=False, max_workers=1)

        if not training_results or symbol not in training_results:
            logger.error(f"模型训练失败: {symbol}")
            return data_results, None

        logger.info(f"✓ 模型训练完成")

    except Exception as e:
        logger.error(f"模型训练异常: {e}")
        import traceback
        traceback.print_exc()
        return data_results, None

    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "="*100)
    logger.info(f"✓ {symbol} 测试完成！总耗时: {elapsed/60:.1f}分钟")
    logger.info("="*100)

    return data_results, training_results


def test_multiple_symbols(symbols):
    """
    批量测试多个品种

    Args:
        symbols: 品种列表

    Returns:
        所有品种的结果汇总
    """
    logger.info("\n" + "="*100)
    logger.info("批量品种测试")
    logger.info("="*100)
    logger.info(f"待测品种: {symbols}")
    logger.info(f"总计: {len(symbols)} 个品种")

    all_results = {}
    overall_start = datetime.now()

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n{'='*100}")
        logger.info(f"进度: {i}/{len(symbols)}")
        logger.info(f"{'='*100}")

        data_result, training_result = test_single_symbol(symbol)

        all_results[symbol] = {
            'data_result': data_result,
            'training_result': training_result,
            'success': training_result is not None
        }

    total_elapsed = (datetime.now() - overall_start).total_seconds()

    # 汇总结果
    logger.info("\n" + "="*100)
    logger.info("批量测试完成")
    logger.info("="*100)
    logger.info(f"总耗时: {total_elapsed/60:.1f}分钟")

    # 成功统计
    success_count = sum(1 for r in all_results.values() if r['success'])
    logger.info(f"成功: {success_count}/{len(symbols)}")

    for symbol, result in all_results.items():
        status = "✓" if result['success'] else "✗"
        logger.info(f"  {status} {symbol} ({SYMBOL_CONFIG.get(symbol, {}).get('name', symbol)})")

    return all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='品种测试脚本')
    parser.add_argument('--symbol', type=str, help='单个品种代码（如I8888.XDCE）')
    parser.add_argument('--symbols', type=str, nargs='+', help='多个品种代码（空格分隔）')

    args = parser.parse_args()

    # 验证路径
    if not SOURCE_DATA_DIR.exists():
        logger.error(f"源数据目录不存在: {SOURCE_DATA_DIR}")
        return 1

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 确定要测试的品种
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    else:
        # 默认测试未完成的品种
        symbols = ['AU8888.XSGE', 'CF8888.XZCE', 'IF8888.CCFX']
        logger.info(f"未指定品种，默认测试: {symbols}")

    # 验证品种代码
    invalid_symbols = [s for s in symbols if s not in SYMBOL_CONFIG]
    if invalid_symbols:
        logger.error(f"无效的品种代码: {invalid_symbols}")
        logger.error(f"支持的品种: {list(SYMBOL_CONFIG.keys())}")
        return 1

    # 执行测试
    if len(symbols) == 1:
        test_single_symbol(symbols[0])
    else:
        test_multiple_symbols(symbols)

    return 0


if __name__ == '__main__':
    sys.exit(main())
