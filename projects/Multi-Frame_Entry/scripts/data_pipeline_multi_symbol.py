#!/usr/bin/env python3
"""
多品种数据准备管道

功能：
1. 从期货商品指数_parquet读取5个品种的1min数据
2. 进行多周期重采样（5/15/60min/day）
3. 生成Regime标签和趋势标签（10根K线窗口）
4. 计算57个特征
5. 支持并行处理

品种：
- HC8888.XSGE (热卷)
- I8888.XDCE (铁矿石)
- AU8888.XSGE (黄金)
- CF8888.XZCE (郑棉)
- IF8888.CCFX (股指期货)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from labels import volatility_regime, binary_label
from features import trend_features

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 品种配置
SYMBOL_CONFIG = {
    'HC8888.XSGE': {
        'name': '热卷',
        'exchange': 'XSGE',
        'source_file': 'HC8888.XSGE.parquet'
    },
    'I8888.XDCE': {
        'name': '铁矿石',
        'exchange': 'XDCE',
        'source_file': 'I8888.XDCE.parquet'
    },
    'AU8888.XSGE': {
        'name': '黄金',
        'exchange': 'XSGE',
        'source_file': 'AU8888.XSGE.parquet'
    },
    'CF8888.XZCE': {
        'name': '郑棉',
        'exchange': 'XZCE',
        'source_file': 'CF8888.XZCE.parquet'
    },
    'IF8888.CCFX': {
        'name': '股指期货',
        'exchange': 'CCFX',
        'source_file': 'IF8888.CCFX.parquet',
        'handle_missing': True  # 需要特殊处理缺失值
    }
}

# 时间配置
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
TARGET_FREQS = ['5min', '15min', '60min', 'D']

# 重采样规则
RESAMPLE_RULES = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'money': 'sum',  # 成交额
    'open_interest': 'last'
}


class MultiSymbolDataPipeline:
    """多品种数据准备管道"""

    def __init__(
        self,
        source_dir: Path,
        output_base_dir: Path,
        symbols: List[str] = None
    ):
        """
        初始化数据管道

        Args:
            source_dir: 源数据目录（K线数据库/期货商品指数_parquet）
            output_base_dir: 输出基础目录
            symbols: 要处理的品种列表，None则处理所有
        """
        self.source_dir = Path(source_dir)
        self.output_base_dir = Path(output_base_dir)
        self.symbols = symbols or list(SYMBOL_CONFIG.keys())

        # 创建输出目录结构
        self.data_dir = self.output_base_dir / 'multi_symbol'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*80)
        logger.info("多品种数据准备管道初始化")
        logger.info("="*80)
        logger.info(f"源数据目录: {self.source_dir}")
        logger.info(f"输出目录: {self.data_dir}")
        logger.info(f"处理品种: {self.symbols}")
        logger.info(f"时间范围: {START_DATE} 至 {END_DATE}")
        logger.info("="*80 + "\n")

    def read_1min_data(self, symbol: str) -> pd.DataFrame:
        """
        读取品种的1min数据

        Args:
            symbol: 品种代码

        Returns:
            1min数据的DataFrame
        """
        config = SYMBOL_CONFIG[symbol]
        source_file = self.source_dir / config['source_file']

        logger.info(f"读取 {symbol} ({config['name']}) 的1min数据...")

        if not source_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {source_file}")

        # 读取parquet文件
        df = pd.read_parquet(source_file)

        # 确保索引是datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            else:
                df.index = pd.to_datetime(df.index)

        # 筛选时间范围
        df = df.loc[START_DATE:END_DATE]

        # 处理缺失值
        if config.get('handle_missing', False):
            logger.info(f"  ⚠️  检测到需要处理缺失值")
            missing_count = df.isnull().sum().sum()
            missing_pct = missing_count / (len(df) * len(df.columns)) * 100
            logger.info(f"  缺失值: {missing_count:,} ({missing_pct:.2f}%)")

            # 前向填充
            df = df.fillna(method='ffill')

            # 删除连续缺失超过100行的样本
            max_consecutive_missing = 100
            for col in df.columns:
                # 计算连续缺失的数量
                consecutive_missing = df[col].isnull().astype(int).groupby(
                    (df[col].notnull()).cumsum()
                ).cumsum()

                # 标记需要删除的行
                to_drop = consecutive_missing > max_consecutive_missing
                if to_drop.any():
                    logger.info(f"  {col}: 删除 {to_drop.sum()} 行（连续缺失>{max_consecutive_missing}）")
                    df = df[~to_drop]

            # 再次前向填充剩余的缺失值
            df = df.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"  ✓ 数据行数: {len(df):,}")
        logger.info(f"  ✓ 时间范围: {df.index.min()} 至 {df.index.max()}")

        return df

    def resample_data(self, df_1min: pd.DataFrame, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        重采样数据到多个周期

        Args:
            df_1min: 1min数据
            symbol: 品种代码

        Returns:
            各周期数据的字典
        """
        logger.info(f"\n重采样 {symbol}...")

        resampled_data = {'1min': df_1min}

        for freq in TARGET_FREQS:
            logger.info(f"  生成 {freq} 数据...")

            # 重采样
            df_resampled = df_1min.resample(freq).agg(RESAMPLE_RULES)

            # 删除全为NaN的行
            df_resampled = df_resampled.dropna(how='all')

            # 计算VWAP（如果有amount和volume）
            if 'amount' in df_resampled.columns and 'volume' in df_resampled.columns:
                df_resampled['vwap'] = df_resampled['amount'] / df_resampled['volume']
                df_resampled['vwap'] = df_resampled['vwap'].fillna(method='ffill')

            logger.info(f"    ✓ {freq}: {len(df_resampled):,} 行")
            resampled_data[freq] = df_resampled

        return resampled_data

    def save_resampled_data(self, resampled_data: Dict[str, pd.DataFrame], symbol: str):
        """
        保存重采样数据到Qlib格式

        Args:
            resampled_data: 重采样数据字典
            symbol: 品种代码
        """
        output_dir = self.data_dir / symbol / 'qlib_data'
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n保存 {symbol} 的重采样数据...")

        for freq, df in resampled_data.items():
            # 创建目录结构
            freq_dir = output_dir / f'instruments/{freq}'
            freq_dir.mkdir(parents=True, exist_ok=True)

            # 保存每个字段
            for field in df.columns:
                if field not in RESAMPLE_RULES.keys() and field not in ['vwap', 'avg', 'money', 'open_interest']:
                    continue

                field_dir = freq_dir / f'${field}'
                field_dir.mkdir(parents=True, exist_ok=True)

                # 保存字段数据
                df_field = df[[field]].copy()
                df_field = df_field.dropna()

                output_file = field_dir / f'{symbol}.csv'
                df_field.to_csv(output_file)
                logger.info(f"  ✓ 保存 {freq}/${field}: {len(df_field)} 行")

    def generate_labels(
        self,
        df_60min: pd.DataFrame,
        symbol: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成Regime标签和趋势标签

        Args:
            df_60min: 60min数据
            symbol: 品种代码

        Returns:
            (regime_labels, trend_labels)
        """
        logger.info(f"\n生成 {symbol} 的标签...")

        # 创建输出目录
        label_dir = self.data_dir / symbol / 'labels'
        label_dir.mkdir(parents=True, exist_ok=True)

        # 准备收盘价数据
        df_close = df_60min[['close']].copy()

        # 1. 生成波动率Regime标签（使用临时CSV文件）
        logger.info("  生成波动率Regime标签...")
        regime_temp_file = label_dir / 'temp_close_for_regime.csv'
        df_close.to_csv(regime_temp_file)

        regime_output_file = label_dir / f'volatility_regime_labels_{symbol}.csv'
        df_regime = volatility_regime.generate_volatility_regime_labels(
            window=10,
            data_file=regime_temp_file,
            output_file=regime_output_file,
            train_start_year=2020
        )

        # 删除临时文件
        regime_temp_file.unlink()

        # 2. 生成趋势标签（10根K线窗口）
        logger.info("  生成趋势标签（10根K线窗口）...")
        trend_temp_file = label_dir / 'temp_close_for_trend.csv'
        df_close.to_csv(trend_temp_file)

        trend_output_file = label_dir / f'binary_labels_10bars_{symbol}.csv'
        df_trend = binary_label.generate_binary_labels(
            window=10,
            sigma_threshold=1.5,
            data_file=trend_temp_file,
            output_file=trend_output_file
        )

        # 删除临时文件
        trend_temp_file.unlink()

        logger.info(f"  ✓ Regime标签: {len(df_regime)} 行")
        logger.info(f"  ✓ 趋势标签: {len(df_trend)} 行")

        return df_regime, df_trend

    def generate_features(
        self,
        df_60min: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        生成特征

        Args:
            df_60min: 60min数据
            symbol: 品种代码

        Returns:
            特征DataFrame
        """
        logger.info(f"\n生成 {symbol} 的特征...")

        # 创建输出目录
        feature_dir = self.data_dir / symbol / 'features'
        feature_dir.mkdir(parents=True, exist_ok=True)

        # 创建TrendFeatures实例
        feature_generator = trend_features.TrendFeatures()

        # 计算所有特征
        df_features = feature_generator.compute_all_features(df_60min, shift=True)

        # 保存特征到CSV
        output_file = feature_dir / f'trend_features_{symbol}.csv'
        df_features.to_csv(output_file)
        logger.info(f"  ✓ 特征已保存: {output_file}")

        logger.info(f"  ✓ 特征数: {len(df_features.columns)}")
        logger.info(f"  ✓ 样本数: {len(df_features)}")

        return df_features

    def process_single_symbol(self, symbol: str) -> Dict:
        """
        处理单个品种的完整流程

        Args:
            symbol: 品种代码

        Returns:
            处理结果信息
        """
        logger.info("\n" + "="*80)
        logger.info(f"开始处理品种: {symbol} ({SYMBOL_CONFIG[symbol]['name']})")
        logger.info("="*80)

        start_time = datetime.now()
        result = {
            'symbol': symbol,
            'name': SYMBOL_CONFIG[symbol]['name'],
            'success': False,
            'error': None,
            'data_stats': {}
        }

        try:
            # 1. 读取1min数据
            df_1min = self.read_1min_data(symbol)
            result['data_stats']['1min_rows'] = len(df_1min)

            # 2. 重采样
            resampled_data = self.resample_data(df_1min, symbol)
            result['data_stats']['freqs'] = list(resampled_data.keys())

            # 3. 保存重采样数据
            self.save_resampled_data(resampled_data, symbol)

            # 4. 生成标签
            df_regime, df_trend = self.generate_labels(resampled_data['60min'], symbol)
            result['data_stats']['regime_labels'] = len(df_regime)
            result['data_stats']['trend_labels'] = len(df_trend)

            # 5. 生成特征
            df_features = self.generate_features(resampled_data['60min'], symbol)
            result['data_stats']['features'] = len(df_features.columns)
            result['data_stats']['samples'] = len(df_features)

            result['success'] = True

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"\n✓ {symbol} 处理完成！耗时: {elapsed:.1f}秒")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"\n✗ {symbol} 处理失败: {e}")
            import traceback
            traceback.print_exc()

        return result

    def process_all_symbols(self, parallel: bool = False, max_workers: int = 1) -> List[Dict]:
        """
        处理所有品种（优化内存占用）

        Args:
            parallel: 是否并行处理（默认False避免内存溢出）
            max_workers: 最大并行数（建议1-2）

        Returns:
            所有品种的处理结果
        """
        logger.info("\n" + "="*80)
        logger.info("开始批量处理所有品种")
        logger.info("="*80)
        logger.info(f"处理模式: {'串行' if max_workers == 1 else f'并行({max_workers} workers)'}")
        logger.info("⚠️  使用串行模式以降低内存占用")

        results = []

        if parallel and max_workers > 1:
            # 限制并行数量
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=max_workers)(
                delayed(self.process_single_symbol)(symbol)
                for symbol in tqdm(self.symbols, desc="处理品种")
            )
        else:
            # 串行处理（默认）
            for i, symbol in enumerate(self.symbols, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"进度: {i}/{len(self.symbols)} - {symbol}")
                logger.info(f"{'='*80}")

                result = self.process_single_symbol(symbol)
                results.append(result)

                # 显式清理内存
                import gc
                gc.collect()

                logger.info(f"✓ {symbol} 处理完成 ({i}/{len(self.symbols)})")

        # 汇总结果
        self._summarize_results(results)

        return results

    def _summarize_results(self, results: List[Dict]):
        """汇总处理结果"""
        logger.info("\n" + "="*80)
        logger.info("处理结果汇总")
        logger.info("="*80)

        # 统计成功/失败
        success_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - success_count

        logger.info(f"\n总计: {len(results)} 个品种")
        logger.info(f"✓ 成功: {success_count}")
        logger.info(f"✗ 失败: {failed_count}")

        # 详细结果表格
        logger.info(f"\n{'品种':<15} {'名称':<10} {'状态':<8} {'样本数':<10} {'特征数':<8}")
        logger.info("-" * 60)

        for r in results:
            status = "✓ 成功" if r['success'] else "✗ 失败"
            samples = r.get('data_stats', {}).get('samples', 0)
            features = r.get('data_stats', {}).get('features', 0)
            logger.info(f"{r['symbol']:<15} {r['name']:<10} {status:<8} {samples:<10,} {features:<8}")

        # 保存汇总报告
        report_file = self.data_dir / 'data_pipeline_report.csv'
        df_report = pd.DataFrame(results)
        df_report.to_csv(report_file, index=False)
        logger.info(f"\n✓ 汇总报告已保存: {report_file}")

        logger.info("="*80 + "\n")


def main():
    """主函数"""
    # 路径配置（基于脚本位置）
    project_root = Path(__file__).parent.parent
    source_dir = Path('/Users/mystryl/Documents/Quant/K线数据库/期货商品指数_parquet')
    output_base_dir = project_root / 'data'

    # 验证源数据目录
    if not source_dir.exists():
        raise FileNotFoundError(f"源数据目录不存在: {source_dir}")

    # 确保输出目录存在
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据管道
    pipeline = MultiSymbolDataPipeline(
        source_dir=source_dir,
        output_base_dir=output_base_dir
    )

    # 处理所有品种
    results = pipeline.process_all_symbols(parallel=False, max_workers=1)

    return results


if __name__ == '__main__':
    main()
