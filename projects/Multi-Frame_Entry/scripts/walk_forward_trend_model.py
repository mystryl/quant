"""
Walk Forward 趋势模型训练（高波动Regime）

策略：
1. 筛选高波动Regime数据
2. 使用波动率归一化标签
3. Walk Forward训练：每年用前一年数据训练，测试当年
4. 对比逐年结果
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.trend_features import TrendFeatures

# 模型库
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)


class WalkForwardTrendModel:
    """Walk Forward 趋势模型（只在高波动Regime中训练）"""

    def __init__(
        self,
        model_params: Dict = None,
        top_k_features: int = 30
    ):
        """
        初始化模型

        Args:
            model_params: XGBoost参数
            top_k_features: 特征选择数量
        """
        self.model_params = model_params or {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'eval_metric': 'logloss'
        }
        self.top_k_features = top_k_features
        self.models = {}  # 存储每年的模型
        self.results = {}  # 存储每年的结果

        logger.info("Walk Forward 趋势模型初始化:")
        logger.info(f"  模型参数: {self.model_params}")
        logger.info(f"  Top K 特征: {self.top_k_features}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载OHLCV数据、波动率Regime和趋势标签"""
        logger.info("\n" + "="*60)
        logger.info("加载数据")
        logger.info("="*60)

        # 1. 加载OHLCV数据
        logger.info("\n加载60min OHLCV数据...")
        data_dir = Path('/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq/instruments/60min')
        instrument = 'HC8888.XSGE'

        fields = {
            'open': '$open',
            'high': '$high',
            'low': '$low',
            'close': '$close',
            'volume': '$volume'
        }

        dfs = []
        for field_name, folder_name in fields.items():
            field_file = data_dir / folder_name / f'{instrument}.csv'
            df_field = pd.read_csv(field_file, index_col=0, parse_dates=True)
            dfs.append(df_field)

        df_ohlcv = pd.concat(dfs, axis=1)
        df_ohlcv.columns = fields.keys()
        df_ohlcv = df_ohlcv.reset_index()
        df_ohlcv.columns = ['datetime'] + list(fields.keys())

        logger.info(f"✓ OHLCV数据: {df_ohlcv.shape}")

        # 2. 加载波动率Regime标签
        logger.info("\n加载波动率Regime标签...")
        df_regime = pd.read_csv('data/labels/volatility_regime_labels_2020_2025.csv')
        df_regime['datetime'] = pd.to_datetime(df_regime['datetime'])
        logger.info(f"✓ Regime数据: {df_regime.shape}")

        # 3. 生成波动率归一化趋势标签
        logger.info("\n生成波动率归一化趋势标签...")
        from labels.binary_label import generate_binary_labels

        # 生成全量标签
        df_trend = generate_binary_labels(
            window=10,  # 使用10根K线窗口
            sigma_threshold=1.5,
            output_file=Path('data/labels/binary_labels_10bars.csv')
        )

        # 筛选2020-2025
        df_trend['datetime'] = pd.to_datetime(df_trend['datetime'])
        df_trend_filtered = df_trend[(df_trend['datetime'].dt.year >= 2020) & (df_trend['datetime'].dt.year <= 2025)].copy()

        logger.info(f"✓ 趋势标签数据: {df_trend_filtered.shape}")

        # 4. 合并所有数据
        logger.info("\n合并数据...")

        # 先合并OHLCV和Regime
        df = df_ohlcv.merge(df_regime[['datetime', 'regime_label']], on='datetime', how='inner')

        # 再合并趋势标签
        df = df.merge(df_trend_filtered[['datetime', 'trend_label']], on='datetime', how='inner')

        logger.info(f"✓ 合并后数据: {df.shape}")

        # 5. 筛选高波动Regime
        logger.info("\n筛选高波动Regime...")
        df_high_vol = df[df['regime_label'] == 1].copy()

        logger.info(f"✓ 高波动Regime数据: {df_high_vol.shape}")
        logger.info(f"  占比: {len(df_high_vol)/len(df)*100:.1f}%")

        return df_ohlcv, df_high_vol

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势特征"""
        logger.info("\n计算趋势特征...")

        calculator = TrendFeatures(
            ema_short=20,
            ema_long=60,
            adx_period=14,
            atr_period=14
        )

        df = calculator.compute_all_features(df, shift=True)

        logger.info(f"✓ 特征计算完成: {df.shape}")

        return df

    def prepare_year_data(
        self,
        df: pd.DataFrame,
        train_year: int,
        test_year: int
    ) -> Dict:
        """
        准备单年训练数据

        Args:
            df: 特征数据
            train_year: 训练年份
            test_year: 测试年份

        Returns:
            训练和测试数据
        """
        # 提取年份
        df_copy = df.copy()
        df_copy['year'] = df_copy['datetime'].dt.year

        # 筛选训练集（指定年份）
        train_mask = df_copy['year'] == train_year
        test_mask = df_copy['year'] == test_year

        # 提取特征
        feature_cols = [col for col in df_copy.columns if col not in
                       ['datetime', 'year', 'open', 'high', 'low', 'close', 'volume',
                        'regime_label', 'trend_label']]

        X_train = df_copy.loc[train_mask, feature_cols]
        y_train = df_copy.loc[train_mask, 'trend_label']

        X_test = df_copy.loc[test_mask, feature_cols]
        y_test = df_copy.loc[test_mask, 'trend_label']

        # 删除缺失值
        valid_train = y_train.notna()
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        valid_test = y_test.notna()
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]

        X_train = X_train.dropna()
        y_train = y_train[X_train.index]

        X_test = X_test.dropna()
        y_test = y_test[X_test.index]

        # 转换为整数
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        logger.info(f"\n{train_year} -> {test_year} 数据准备:")
        logger.info(f"  训练集: {len(X_train)} 样本")
        logger.info(f"  测试集: {len(X_test)} 样本")

        # 标签分布
        if len(y_train) > 0:
            logger.info(f"  训练集 - 震荡: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%), "
                       f"有趋势: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

        if len(y_test) > 0:
            logger.info(f"  测试集 - 震荡: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.1f}%), "
                       f"有趋势: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': feature_cols
        }

    def train_year(self, train_year: int, test_year: int, data: Dict):
        """训练单年模型"""
        logger.info("\n" + "="*60)
        logger.info(f"训练模型: {train_year} -> {test_year}")
        logger.info("="*60)

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        feature_names = data['feature_names']

        # 检查数据量
        if len(X_train) < 50 or len(X_test) < 20:
            logger.warning(f"  ⚠️  数据量不足，跳过 ({len(X_train)} 训练, {len(X_test)} 测试)")
            return None

        # 特征选择
        logger.info(f"\n执行特征选择（Top {self.top_k_features}）...")
        temp_model = XGBClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=self.model_params['learning_rate'],
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        temp_model.fit(X_train, y_train)

        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)

        selected_features = importances.head(self.top_k_features)['feature'].tolist()

        logger.info(f"  Top 5 特征:")
        for idx, row in importances.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")

        # 训练最终模型
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # 计算类别权重
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        logger.info(f"\n训练XGBoost模型...")
        logger.info(f"  scale_pos_weight: {scale_pos_weight:.2f}")

        model = XGBClassifier(
            **self.model_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_selected, y_train)

        # 评估
        train_acc = model.score(X_train_selected, y_train)
        test_acc = model.score(X_test_selected, y_test)

        logger.info(f"\n✓ 训练集准确率: {train_acc:.4f}")
        logger.info(f"✓ 测试集准确率: {test_acc:.4f}")

        # 详细评估
        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)[:, 1]

        # AUC-ROC
        try:
            auc = roc_auc_score(y_test, y_proba)
            logger.info(f"AUC-ROC: {auc:.4f}")
        except:
            auc = None
            logger.info("AUC-ROC: 无法计算")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"混淆矩阵:")
        logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

        # 召回率
        recall_0 = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        recall_1 = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        logger.info(f"\n召回率:")
        logger.info(f"  震荡: {recall_0:.4f}")
        logger.info(f"  有趋势: {recall_1:.4f}")

        # 保存结果
        result = {
            'train_year': train_year,
            'test_year': test_year,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'auc': auc,
            'confusion_matrix': cm,
            'recall_range': recall_0,
            'recall_trend': recall_1,
            'model': model,
            'selected_features': selected_features
        }

        return result

    def walk_forward_training(self, df: pd.DataFrame, years: List[Tuple[int, int]]):
        """
        Walk Forward训练

        Args:
            df: 特征数据
            years: [(train_year, test_year), ...] 列表
        """
        logger.info("\n" + "="*60)
        logger.info("开始Walk Forward训练")
        logger.info("="*60)

        results = []

        for train_year, test_year in years:
            # 准备数据
            data = self.prepare_year_data(df, train_year, test_year)

            # 训练模型
            result = self.train_year(train_year, test_year, data)

            if result is not None:
                results.append(result)
                self.models[test_year] = result['model']

        self.results = results

        # 汇总结果
        self.summarize_results()

        return results

    def summarize_results(self):
        """汇总Walk Forward结果"""
        logger.info("\n" + "="*60)
        logger.info("Walk Forward 结果汇总")
        logger.info("="*60)

        if not self.results:
            logger.info("无结果")
            return

        # 创建结果表格
        df_results = pd.DataFrame([{
            'Train→Test': f"{r['train_year']}→{r['test_year']}",
            'Train样本': r['train_samples'],
            'Test样本': r['test_samples'],
            '训练准确率': f"{r['train_acc']:.2%}",
            '测试准确率': f"{r['test_acc']:.2%}",
            'AUC-ROC': f"{r['auc']:.4f}" if r['auc'] else 'N/A',
            '震荡召回率': f"{r['recall_range']:.2%}",
            '有趋势召回率': f"{r['recall_trend']:.2%}"
        } for r in self.results])

        logger.info(f"\n{df_results.to_string(index=False)}")

        # 平均性能
        valid_results = [r for r in self.results if r['test_acc'] > 0]

        if valid_results:
            avg_test_acc = np.mean([r['test_acc'] for r in valid_results])
            avg_auc = np.mean([r['auc'] for r in valid_results if r['auc'] is not None])
            avg_recall_range = np.mean([r['recall_range'] for r in valid_results])
            avg_recall_trend = np.mean([r['recall_trend'] for r in valid_results])

            logger.info(f"\n平均性能:")
            logger.info(f"  测试准确率: {avg_test_acc:.2%}")
            logger.info(f"  AUC-ROC: {avg_auc:.4f}")
            logger.info(f"  震荡召回率: {avg_recall_range:.2%}")
            logger.info(f"  有趋势召回率: {avg_recall_trend:.2%}")


def run_walk_forward_experiment():
    """运行Walk Forward实验"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("="*60)
    logger.info("Walk Forward 趋势模型实验（高波动Regime）")
    logger.info("="*60)

    # 创建模型
    model = WalkForwardTrendModel(
        model_params={
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'eval_metric': 'logloss'
        },
        top_k_features=30
    )

    # 加载数据
    df_ohlcv, df_high_vol = model.load_data()

    # 计算特征
    df_features = model.compute_features(df_high_vol)

    # 定义Walk Forward年对
    # 使用前一年训练，测试下一年
    year_pairs = [
        (2020, 2021),
        (2021, 2022),
        (2022, 2023),
        (2023, 2024),
        (2024, 2025)
    ]

    # Walk Forward训练
    results = model.walk_forward_training(df_features, year_pairs)

    logger.info(f"\n{'='*60}")
    logger.info(f"实验完成！")
    logger.info(f"{'='*60}")

    return model, results


if __name__ == '__main__':
    model, results = run_walk_forward_experiment()
