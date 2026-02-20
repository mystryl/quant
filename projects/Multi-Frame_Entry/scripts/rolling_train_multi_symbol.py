#!/usr/bin/env python3
"""
多品种年度滚动训练框架

功能：
1. Walk Forward训练：2020→2021→2022→2023→2024→2025
2. 只在高波动Regime中训练
3. 使用XGBoost模型和Top30特征选择
4. 支持多品种并行训练
5. 按品种和年份保存模型

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
import pickle
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

# 模型库
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 品种配置
SYMBOL_CONFIG = {
    'HC8888.XSGE': {'name': '热卷'},
    'I8888.XDCE': {'name': '铁矿石'},
    'AU8888.XSGE': {'name': '黄金'},
    'CF8888.XZCE': {'name': '郑棉'},
    'IF8888.CCFX': {'name': '股指期货'}
}

# 训练配置
TRAINING_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_WINDOW_MONTHS = 18  # 训练窗口18个月
TOP_K_FEATURES = 30  # 选择Top30特征

# XGBoost参数
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}


class RollingTrainMultiSymbol:
    """多品种年度滚动训练"""

    def __init__(
        self,
        data_base_dir: Path,
        model_output_dir: Path,
        symbols: List[str] = None
    ):
        """
        初始化训练框架

        Args:
            data_base_dir: 数据基础目录（multi_symbol目录）
            model_output_dir: 模型输出目录
            symbols: 要训练的品种列表
        """
        self.data_base_dir = Path(data_base_dir)
        self.model_output_dir = Path(model_output_dir)
        self.symbols = symbols or list(SYMBOL_CONFIG.keys())

        # 创建输出目录
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.model_output_dir.parent / 'training_results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*80)
        logger.info("多品种年度滚动训练框架")
        logger.info("="*80)
        logger.info(f"数据目录: {self.data_base_dir}")
        logger.info(f"模型输出目录: {self.model_output_dir}")
        logger.info(f"训练品种: {self.symbols}")
        logger.info(f"训练年份: {TRAINING_YEARS}")
        logger.info(f"训练窗口: {TRAIN_WINDOW_MONTHS}个月")
        logger.info(f"Top K特征: {TOP_K_FEATURES}")
        logger.info("="*80 + "\n")

        if not XGBOOST_AVAILABLE:
            logger.warning("⚠️  XGBoost不可用，将跳过训练")

    def load_symbol_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载品种的数据（特征、趋势标签、Regime标签）

        Args:
            symbol: 品种代码

        Returns:
            (features, trend_labels, regime_labels)
        """
        logger.info(f"\n加载 {symbol} 的数据...")

        symbol_dir = self.data_base_dir / symbol

        # 1. 加载特征
        feature_file = symbol_dir / 'features' / f'trend_features_{symbol}.csv'
        if not feature_file.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_file}")

        df_features = pd.read_csv(feature_file, index_col=0, parse_dates=True)
        logger.info(f"  ✓ 特征: {df_features.shape}")

        # 2. 加载趋势标签
        trend_label_file = symbol_dir / 'labels' / f'binary_labels_10bars_{symbol}.csv'
        if not trend_label_file.exists():
            raise FileNotFoundError(f"趋势标签文件不存在: {trend_label_file}")

        df_trend_labels = pd.read_csv(trend_label_file, index_col=0, parse_dates=True)
        logger.info(f"  ✓ 趋势标签: {df_trend_labels.shape}")

        # 3. 加载Regime标签
        regime_label_file = symbol_dir / 'labels' / f'volatility_regime_labels_{symbol}.csv'
        if not regime_label_file.exists():
            raise FileNotFoundError(f"Regime标签文件不存在: {regime_label_file}")

        df_regime_labels = pd.read_csv(regime_label_file, index_col=0, parse_dates=True)
        logger.info(f"  ✓ Regime标签: {df_regime_labels.shape}")

        return df_features, df_trend_labels, df_regime_labels

    def prepare_training_data(
        self,
        df_features: pd.DataFrame,
        df_trend_labels: pd.DataFrame,
        df_regime_labels: pd.DataFrame,
        train_start: str,
        train_end: str,
        test_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备训练和测试数据

        Args:
            df_features: 特征DataFrame
            df_trend_labels: 趋势标签
            df_regime_labels: Regime标签
            train_start: 训练开始日期
            train_end: 训练结束日期
            test_year: 测试年份

        Returns:
            (X_train, y_train), (X_test, y_test)
        """
        # 合并数据
        df = df_features.copy()

        # 添加标签（确保列名不冲突）
        df['trend_label'] = df_trend_labels['trend_label']
        df['regime_label'] = df_regime_labels['regime_label']

        # 删除包含NaN的行
        df = df.dropna(subset=['trend_label', 'regime_label'])

        # 筛选时间范围
        train_data = df.loc[train_start:train_end]
        test_data = df.loc[f"{test_year}-01-01":f"{test_year}-12-31"]

        # 只在高波动Regime中训练
        train_data = train_data[train_data['regime_label'] == 1]

        # 分离特征和标签
        feature_cols = [col for col in df.columns
                       if col not in ['trend_label', 'regime_label']]

        X_train = train_data[feature_cols].copy()
        y_train = train_data['trend_label']

        X_test = test_data[feature_cols].copy()
        y_test = test_data['trend_label']

        # 清理数据：替换inf和NaN
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(X_train.median())

        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(X_test.median())

        logger.info(f"  训练集: {len(X_train)} 样本 (高波动Regime)")
        logger.info(f"  测试集: {len(X_test)} 样本 (全部Regime)")
        logger.info(f"  标签分布 - 训练集: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
        logger.info(f"  标签分布 - 测试集: 0={sum(y_test==0)}, 1={sum(y_test==1)}")

        return (X_train, y_train), (X_test, y_test)

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple:
        """
        训练XGBoost模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            (model, feature_importance, metrics)
        """
        logger.info("\n  训练XGBoost模型...")

        # 选择Top K特征（基于特征重要性）
        # 首先用所有特征训练一个初步模型
        preliminary_model = XGBClassifier(**XGBOOST_PARAMS)
        preliminary_model.fit(X_train, y_train)

        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': preliminary_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 选择Top K特征
        top_features = feature_importance.head(TOP_K_FEATURES)['feature'].tolist()
        logger.info(f"  Top {TOP_K_FEATURES} 特征:")
        for idx, row in feature_importance.head(TOP_K_FEATURES).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")

        # 用Top K特征重新训练
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]

        # 训练最终模型
        model = XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train_top, y_train)

        # 评估
        y_pred = model.predict(X_test_top)
        y_pred_proba = model.predict_proba(X_test_top)[:, 1]

        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # 更新特征重要性（只保留Top K）
        feature_importance = feature_importance.head(TOP_K_FEATURES)

        return model, feature_importance, metrics, top_features

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        计算评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率

        Returns:
            指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }

        # 震荡召回率（label=0的召回率）
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)

        range_mask = y_true_array == 0
        if range_mask.sum() > 0:
            metrics['recall_range'] = recall_score(y_true_array, y_pred_array, labels=[0], average='macro')[0] if hasattr(recall_score(y_true_array, y_pred_array, labels=[0], average='macro'), '__iter__') else \
                                       (y_pred_array[~range_mask].sum() == 0).astype(int).sum() / range_mask.sum() if range_mask.sum() > 0 else 0
        else:
            metrics['recall_range'] = 0.0

        # 有趋势召回率（label=1的召回率）
        trend_mask = y_true_array == 1
        if trend_mask.sum() > 0:
            metrics['recall_trend'] = (y_pred_array[trend_mask] == 1).sum() / trend_mask.sum()
        else:
            metrics['recall_trend'] = 0.0

        return metrics

    def train_single_symbol_single_year(
        self,
        symbol: str,
        year: int,
        df_features: pd.DataFrame,
        df_trend_labels: pd.DataFrame,
        df_regime_labels: pd.DataFrame
    ) -> Dict:
        """
        训练单个品种单个年份

        Args:
            symbol: 品种代码
            year: 测试年份
            df_features: 特征
            df_trend_labels: 趋势标签
            df_regime_labels: Regime标签

        Returns:
            训练结果字典
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"训练 {symbol} - {year}年")
        logger.info(f"{'='*60}")

        result = {
            'symbol': symbol,
            'year': year,
            'success': False,
            'error': None,
            'metrics': {},
            'feature_importance': None,
            'top_features': []
        }

        try:
            # 计算训练时间范围（前18个月）
            train_start = f"{year-2}-01-01"
            train_end = f"{year-1}-12-31"

            # 准备数据
            (X_train, y_train), (X_test, y_test) = self.prepare_training_data(
                df_features, df_trend_labels, df_regime_labels,
                train_start, train_end, year
            )

            # 检查数据是否足够
            if len(X_train) < 100 or len(X_test) < 50:
                raise ValueError(f"数据不足: 训练集={len(X_train)}, 测试集={len(X_test)}")

            # 训练模型
            model, feature_importance, metrics, top_features = self.train_model(
                X_train, y_train, X_test, y_test
            )

            # 保存模型
            model_file = self.model_output_dir / f'{symbol}_{year}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'features': top_features,
                    'feature_importance': feature_importance,
                    'metrics': metrics,
                    'symbol': symbol,
                    'year': year,
                    'train_period': f"{year-2}-{year-1}",
                    'test_period': str(year)
                }, f)

            logger.info(f"\n  ✓ 模型已保存: {model_file}")
            logger.info(f"\n  性能指标:")
            for key, value in metrics.items():
                logger.info(f"    {key}: {value:.4f}")

            result['success'] = True
            result['metrics'] = metrics
            result['feature_importance'] = feature_importance
            result['top_features'] = top_features

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"\n  ✗ 训练失败: {e}")
            import traceback
            traceback.print_exc()

        return result

    def train_single_symbol(self, symbol: str) -> List[Dict]:
        """
        训练单个品种的所有年份

        Args:
            symbol: 品种代码

        Returns:
            所有年份的结果列表
        """
        logger.info("\n" + "="*80)
        logger.info(f"开始训练品种: {symbol} ({SYMBOL_CONFIG[symbol]['name']})")
        logger.info("="*80)

        start_time = datetime.now()
        results = []

        try:
            # 加载数据
            df_features, df_trend_labels, df_regime_labels = self.load_symbol_data(symbol)

            # 逐年训练
            for year in TRAINING_YEARS:
                result = self.train_single_symbol_single_year(
                    symbol, year,
                    df_features, df_trend_labels, df_regime_labels
                )
                results.append(result)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"\n✓ {symbol} 训练完成！耗时: {elapsed:.1f}秒")

        except Exception as e:
            logger.error(f"\n✗ {symbol} 训练失败: {e}")
            import traceback
            traceback.print_exc()

        return results

    def train_all_symbols(self, parallel: bool = False, max_workers: int = 1) -> Dict[str, List[Dict]]:
        """
        训练所有品种（优化内存占用）

        Args:
            parallel: 是否并行训练（默认False避免内存溢出）
            max_workers: 最大并行数（建议1-2）

        Returns:
            所有品种的结果字典
        """
        logger.info("\n" + "="*80)
        logger.info("开始批量训练所有品种")
        logger.info("="*80)
        logger.info(f"训练模式: {'串行' if max_workers == 1 else f'并行({max_workers} workers)'}")
        logger.info("⚠️  使用串行模式以降低内存占用")

        all_results = {}

        if parallel and max_workers > 1 and XGBOOST_AVAILABLE:
            # 限制并行数量
            from joblib import Parallel, delayed

            results_list = Parallel(n_jobs=max_workers)(
                delayed(self.train_single_symbol)(symbol)
                for symbol in tqdm(self.symbols, desc="训练品种")
            )

            # 整理结果
            for symbol, results in zip(self.symbols, results_list):
                all_results[symbol] = results

        else:
            # 串行训练（默认）
            for i, symbol in enumerate(self.symbols, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"进度: {i}/{len(self.symbols)} - {symbol}")
                logger.info(f"{'='*80}")

                results = self.train_single_symbol(symbol)
                all_results[symbol] = results

                # 显式清理内存
                import gc
                gc.collect()

                logger.info(f"✓ {symbol} 训练完成 ({i}/{len(self.symbols)})")

        # 汇总结果
        self._summarize_results(all_results)

        # 保存汇总结果
        self._save_summary(all_results)

        return all_results

    def _summarize_results(self, all_results: Dict[str, List[Dict]]):
        """汇总训练结果"""
        logger.info("\n" + "="*80)
        logger.info("训练结果汇总")
        logger.info("="*80)

        # 生成汇总表格
        summary_data = []

        for symbol, results in all_results.items():
            for result in results:
                if result['success']:
                    row = {
                        '品种': symbol,
                        '名称': SYMBOL_CONFIG[symbol]['name'],
                        '年份': result['year'],
                        '准确率': f"{result['metrics']['accuracy']:.4f}",
                        'AUC': f"{result['metrics']['auc']:.4f}",
                        '震荡召回率': f"{result['metrics'].get('recall_range', 0):.4f}",
                        '趋势召回率': f"{result['metrics'].get('recall_trend', 0):.4f}",
                        'F1': f"{result['metrics']['f1']:.4f}"
                    }
                    summary_data.append(row)

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            logger.info(f"\n{df_summary.to_string(index=False)}")

            # 保存汇总CSV
            summary_file = self.results_dir / 'training_summary.csv'
            df_summary.to_csv(summary_file, index=False)
            logger.info(f"\n✓ 汇总报告已保存: {summary_file}")

        logger.info("="*80 + "\n")

    def _save_summary(self, all_results: Dict[str, List[Dict]]):
        """保存详细结果"""
        import json

        # 保存JSON格式的详细结果
        json_file = self.results_dir / 'training_results.json'

        # 转换为可序列化的格式
        serializable_results = {}
        for symbol, results in all_results.items():
            serializable_results[symbol] = []
            for result in results:
                serializable_result = {
                    'symbol': result['symbol'],
                    'year': result['year'],
                    'success': result['success'],
                    'error': result['error'],
                    'metrics': result['metrics'],
                    'top_features': result['top_features']
                }
                serializable_results[symbol].append(serializable_result)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ 详细结果已保存: {json_file}")


def main():
    """主函数"""
    # 路径配置
    data_base_dir = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/multi_symbol')
    model_output_dir = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/models/rolling')

    # 创建训练框架
    trainer = RollingTrainMultiSymbol(
        data_base_dir=data_base_dir,
        model_output_dir=model_output_dir
    )

    # 训练所有品种
    all_results = trainer.train_all_symbols(parallel=True)

    return all_results


if __name__ == '__main__':
    main()
