"""
ADX/DMI Factor Backtest — REAL Rebar Futures Data (RB9999.XSGE)

Uses 3 years of daily bars resampled from 1-minute futures data.
Tests 3 parameter variants of the ADX/DMI trading system as a continuous factor:
  - ADX_N7:  N=7,  M=3 (fast)
  - ADX_N14: N=14, M=6 (standard)
  - ADX_N21: N=21, M=9 (slow)

Note: Single-instrument analysis uses time-series IC (Pearson correlation of
factor vs next-day returns over time) instead of cross-sectional IC.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from datetime import datetime
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('/Users/mystryl/Documents/Quant/projects/multi_factor_analyzer')

from src.core import PerformanceEvaluator, CycleAligner, StrategyAnalyzer
from src.core.reliability import ReliabilityEvaluator
from src.core.config import DEFAULT_WEIGHTS
from src.report import ReportGenerator, Visualizer


# ---------------------------------------------------------------------------
# Helper: bars since a boolean condition was last True
# ---------------------------------------------------------------------------

def _bars_since(condition: pd.Series) -> pd.Series:
    """Count bars since condition was last True. 0 on the signal bar itself."""
    groups = condition.cumsum()
    result = groups.groupby(groups).cumcount()
    result[condition] = 0
    return result


# ---------------------------------------------------------------------------
# ADX / DMI core indicator calculation
# ---------------------------------------------------------------------------

def calculate_adx_dmi(df: pd.DataFrame, n: int = 14, m: int = 6):
    """Calculate ADX/DMI components. Returns: (pdi, mdi, adx, adxr) as pd.Series."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    hd = high - high.shift(1)
    ld = low.shift(1) - low

    dmp = hd.where((hd > 0) & (hd >= ld), 0.0)
    dmm = ld.where((ld > 0) & (ld > hd), 0.0)

    tr_sum = tr.rolling(n, min_periods=n).sum()
    dmp_sum = dmp.rolling(n, min_periods=n).sum()
    dmm_sum = dmm.rolling(n, min_periods=n).sum()

    pdi = (dmp_sum / tr_sum) * 100
    mdi = (dmm_sum / tr_sum) * 100

    dx_sum = pdi + mdi
    dx = pd.Series(0.0, index=df.index)
    nonzero = dx_sum > 0
    dx[nonzero] = ((mdi[nonzero] - pdi[nonzero]).abs() / dx_sum[nonzero]) * 100

    adx = dx.rolling(m, min_periods=m).mean()
    adxr = (adx + adx.shift(m)) / 2

    return pdi, mdi, adx, adxr


# ---------------------------------------------------------------------------
# ADX/DMI factor signal encoder
# ---------------------------------------------------------------------------

def calculate_adx_factor(df: pd.DataFrame, n: int = 14, m: int = 6,
                         lookback: int = 5) -> pd.Series:
    """Generate a continuous factor value from ADX/DMI trading signals. Range ~ [-1, +1]."""
    pdi, mdi, adx, adxr = calculate_adx_dmi(df, n, m)

    golden_cross = (pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))
    death_cross = (mdi > pdi) & (mdi.shift(1) <= pdi.shift(1))

    bars_since_golden = _bars_since(golden_cross)
    bars_since_death = _bars_since(death_cross)

    adx_cross_up = (adx > 20) & (adx.shift(1) <= 20)
    adx_declining = adx < adx.shift(1)

    factor = pd.Series(0.0, index=df.index)

    # Strongest bullish: ADX just crossed 20 upward + DI+ on top
    mask = adx_cross_up & (pdi > mdi)
    factor[mask] = 1.0

    # Standard bullish: recent golden cross + ADX > 20
    mask = (bars_since_golden <= lookback) & (bars_since_golden > 0) & (adx > 20)
    strength = (1 - bars_since_golden[mask] / lookback) * 0.6
    adx_boost = np.clip((adx[mask] - 20) / 20, 0, 0.4)
    factor[mask] = np.maximum(factor[mask], strength + adx_boost)

    # Bullish trend continuation: DI+ > DI- AND ADX > 25
    mask = ((pdi > mdi) & (adx > 25) &
            (bars_since_golden > lookback) & (bars_since_death > lookback))
    factor[mask] = np.clip((adx[mask] - 25) / 30, 0.1, 0.5)

    # Exit signal: pull toward 0 when ADX declining
    mask = (factor > 0) & adx_declining
    factor[mask] *= 0.3

    # Bearish: death cross + ADX > 20 + ADX declining
    mask = ((bars_since_death <= lookback) & (bars_since_death > 0) &
            (adx > 20) & adx_declining)
    strength = (1 - bars_since_death[mask] / lookback) * 0.6
    adx_boost = np.clip((adx[mask] - 20) / 20, 0, 0.4)
    factor[mask] = -(strength + adx_boost)

    # Suppress in choppy market (ADX < 20)
    factor[adx < 20] = 0.0

    return factor


# ---------------------------------------------------------------------------
# Load real rebar futures data
# ---------------------------------------------------------------------------

def load_real_data(parquet_path: str, start_date: str = '2022-01-01'):
    """Load 1-min parquet, resample to daily, filter to date range."""
    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    logger.info(f"  Raw rows: {len(df):,}, range: {df.index[0]} ~ {df.index[-1]}")

    # Resample 1-min bars to daily bars
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()
    logger.info(f"  Daily bars: {len(daily):,}, range: {daily.index[0].date()} ~ {daily.index[-1].date()}")

    # Filter last 3 years
    daily = daily[start_date:]
    logger.info(f"  Filtered to {start_date}: {len(daily):,} bars, {daily.index[0].date()} ~ {daily.index[-1].date()}")
    logger.info(f"  Price range: {daily['close'].min():.0f} ~ {daily['close'].max():.0f}")

    return daily


# ---------------------------------------------------------------------------
# Prepare single-instrument factor data with MultiIndex
# ---------------------------------------------------------------------------

def prepare_single_instrument_factor(daily: pd.DataFrame, factor_func, factor_name: str):
    """Calculate factor for a single instrument and wrap in MultiIndex."""
    instrument = 'RB9999.XSGE'

    factor = factor_func(daily)
    factor = factor.dropna()
    df_aligned = daily.loc[factor.index]

    n = len(factor)
    index = pd.MultiIndex.from_arrays(
        [factor.index, [instrument] * n],
        names=['datetime', 'instrument']
    )

    factor_df = pd.DataFrame({'factor': factor.values}, index=index)
    price_df = pd.DataFrame({'close': df_aligned['close'].values}, index=index)

    logger.info(f"  {factor_name}: {len(factor_df)} data points")
    return factor_df, price_df


# ---------------------------------------------------------------------------
# Time-series IC fallback for single instrument
# ---------------------------------------------------------------------------

def calculate_timeseries_metrics(pred: pd.Series, label: pd.Series):
    """
    For single-instrument analysis, compute time-series metrics:
    - Rolling Pearson IC (factor_t vs return_{t+1} over rolling window)
    - Overall Pearson & Spearman correlation
    - Directional accuracy
    """
    valid = pred.notna() & label.notna()
    p = pred[valid]
    l = label[valid]

    # Rolling IC (60-day window)
    rolling_ic = p.rolling(60, min_periods=20).corr(l)

    # Overall IC
    if len(p) > 2:
        ic_mean = p.corr(l)
        rank_ic_mean = p.corr(l.rank())
    else:
        ic_mean = 0.0
        rank_ic_mean = 0.0

    # Rolling IC statistics
    ic_std = rolling_ic.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0.0

    rank_ic_series = p.rolling(60, min_periods=20).corr(l.rank())
    rank_ic_std = rank_ic_series.std()
    rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0

    # Directional accuracy: sign(factor_t) == sign(return_{t+1})
    sign_agree = ((p > 0) & (l > 0)) | ((p < 0) & (l < 0))
    directional_acc = sign_agree.mean()

    return {
        'ic': rolling_ic.dropna(),
        'rank_ic': rank_ic_series.dropna(),
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'rank_ic_mean': rank_ic_mean,
        'rank_ic_std': rank_ic_std,
        'rank_icir': rank_icir,
        'directional_accuracy': directional_acc,
    }


def calculate_long_short_return_single(pred: pd.Series, label: pd.Series):
    """
    For single instrument: long when factor > 0, short when factor < 0.
    Returns daily PnL series.
    """
    valid = pred.notna() & label.notna()
    p = pred[valid]
    l = label[valid]

    # Position = sign of factor
    position = np.sign(p)
    # Daily return = position * actual return
    daily_pnl = position * l

    return daily_pnl


def calculate_single_instrument_metrics(pred: pd.Series, label: pd.Series):
    """Full metrics dict for single-instrument factor analysis."""
    ts_metrics = calculate_timeseries_metrics(pred, label)
    ls_return = calculate_long_short_return_single(pred, label)

    # Risk metrics
    annual_return = (1 + ls_return.mean()) ** 252 - 1
    sharpe = ls_return.mean() / ls_return.std() * np.sqrt(252) if ls_return.std() > 0 else 0.0

    cum = (1 + ls_return).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_drawdown = drawdown.min()

    win_rate = (ls_return > 0).mean()

    return {
        'ic': ts_metrics['ic'],
        'rank_ic': ts_metrics['rank_ic'],
        'ic_mean': ts_metrics['ic_mean'],
        'ic_std': ts_metrics['ic_std'],
        'icir': ts_metrics['icir'],
        'rank_ic_mean': ts_metrics['rank_ic_mean'],
        'rank_ic_std': ts_metrics['rank_ic_std'],
        'rank_icir': ts_metrics['rank_icir'],
        'long_short_return': ls_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'directional_accuracy': ts_metrics['directional_accuracy'],
    }


# ---------------------------------------------------------------------------
# Full analysis pipeline (single-instrument aware)
# ---------------------------------------------------------------------------

def analyze_factor(factor_name: str, factor_df: pd.DataFrame,
                  price_df: pd.DataFrame,
                  output_dir: str = './output'):
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing factor: {factor_name}")
        logger.info(f"{'='*80}\n")

        # 1. Cycle alignment
        logger.info("1. Cycle alignment...")
        aligner = CycleAligner()
        factor_aligned, returns_aligned = aligner.align(
            factor_df, price_df, method='default'
        )
        logger.info(f"   Factor points: {len(factor_aligned):,}")
        logger.info(f"   Return points: {len(returns_aligned):,}")

        # 2. Intersect
        common = factor_aligned.index.intersection(returns_aligned.index)
        factor_final = factor_aligned.loc[common]
        returns_final = returns_aligned.loc[common]
        logger.info(f"   Common points: {len(common):,}")

        # Extract Series
        pred = factor_final.iloc[:, 0] if isinstance(factor_final, pd.DataFrame) else factor_final
        label = returns_final.iloc[:, 0] if isinstance(returns_final, pd.DataFrame) else returns_final

        # 3. Performance evaluation — use single-instrument path
        logger.info("\n3. Performance evaluation (time-series IC for single instrument)...")
        metrics = calculate_single_instrument_metrics(pred, label)

        logger.info(f"   IC mean   : {metrics['ic_mean']:.4f}")
        logger.info(f"   ICIR      : {metrics['icir']:.4f}")
        logger.info(f"   Rank IC   : {metrics['rank_ic_mean']:.4f}")
        logger.info(f"   Rank ICIR : {metrics['rank_icir']:.4f}")
        logger.info(f"   Direction : {metrics['directional_accuracy']:.2%}")
        logger.info(f"   Ann Return: {metrics['annual_return']:.2%}")
        logger.info(f"   Sharpe    : {metrics['sharpe_ratio']:.2f}")
        logger.info(f"   Max DD    : {metrics['max_drawdown']:.2%}")
        logger.info(f"   Win Rate  : {metrics['win_rate']:.2%}")

        # 4. Strategy scenario analysis
        logger.info("\n4. Strategy scenario analysis...")
        sa = StrategyAnalyzer()

        try:
            bull = sa.analyze_bull_strategy(factor_final, returns_final, top_pct=0.2)
            logger.info(f"   Bull annual return : {bull['annual_return']:.2%}")
            logger.info(f"   Bull sharpe        : {bull['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.warning(f"   Bull strategy failed: {e}")
            bull = {'annual_return': metrics['annual_return'], 'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'], 'win_rate': metrics['win_rate'],
                    'total_return': (1 + metrics['long_short_return']).prod() - 1}

        try:
            bear = sa.analyze_bear_strategy(factor_final, returns_final, bottom_pct=0.2)
            logger.info(f"   Bear annual return : {bear['annual_return']:.2%}")
            logger.info(f"   Bear sharpe        : {bear['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.warning(f"   Bear strategy failed: {e}")
            bear = {'annual_return': metrics['annual_return'], 'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'], 'win_rate': metrics['win_rate'],
                    'total_return': (1 + metrics['long_short_return']).prod() - 1}

        try:
            ls = sa.analyze_long_short_strategy(factor_final, returns_final,
                                               top_pct=0.2, bottom_pct=0.2)
            logger.info(f"   L/S  annual return : {ls['annual_return']:.2%}")
            logger.info(f"   L/S  sharpe        : {ls['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.warning(f"   L/S strategy failed (single instrument): {e}")
            ls = {'annual_return': metrics['annual_return'], 'sharpe_ratio': metrics['sharpe_ratio'],
                  'max_drawdown': metrics['max_drawdown'], 'win_rate': metrics['win_rate'],
                  'total_return': (1 + metrics['long_short_return']).prod() - 1}

        scenario_results = {'bull': bull, 'bear': bear, 'long_short': ls}

        # 5. Reliability evaluation
        logger.info("\n5. Reliability evaluation...")
        rel_eval = ReliabilityEvaluator(weights=DEFAULT_WEIGHTS)
        evaluation = rel_eval.evaluate(metrics, scenario_results)

        logger.info(f"   Total score  : {evaluation['total_score']:.2f}")
        logger.info(f"   Reliability  : {evaluation['reliability']}")

        # 6. Report + charts
        logger.info("\n6. Generating report & charts...")
        os.makedirs(output_dir, exist_ok=True)

        report_gen = ReportGenerator(output_dir=output_dir)
        visualizer = Visualizer(output_dir=output_dir)

        reports = report_gen.generate_full_report(
            factor_name=factor_name,
            metrics=metrics,
            scenario_results=scenario_results,
            reliability_result=evaluation
        )
        report_gen.save_markdown(reports['markdown'], f'{factor_name}_report.md')
        report_gen.save_html(reports['html'], f'{factor_name}_report.html')
        charts = visualizer.save_all_charts(metrics, scenario_results,
                                            factor_name)
        logger.info(f"   Saved report + {len(charts)} charts to {output_dir}")

        return metrics, scenario_results, evaluation

    except Exception as e:
        logger.error(f"Factor analysis failed ({factor_name}): {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ---------------------------------------------------------------------------
# Parameter variants
# ---------------------------------------------------------------------------

VARIANTS = [
    {'name': 'ADX_Real_N7',  'n': 7,  'm': 3, 'output': './output/adx_real_n7'},
    {'name': 'ADX_Real_N14', 'n': 14, 'm': 6, 'output': './output/adx_real_n14'},
    {'name': 'ADX_Real_N21', 'n': 21, 'm': 9, 'output': './output/adx_real_n21'},
]


def main():
    logger.info("\n" + "=" * 80)
    logger.info("ADX/DMI Factor Backtest — REAL Rebar Futures (RB9999.XSGE)")
    logger.info("=" * 80)

    # 1. Load real data
    parquet_path = '/Users/mystryl/Documents/Quant/K线数据库/期货主力连续_parquet/RB9999.XSGE.parquet'
    daily = load_real_data(parquet_path, start_date='2022-01-01')

    # 2. Run each variant
    results = {}
    for v in VARIANTS:
        name, n, m, out_dir = v['name'], v['n'], v['m'], v['output']
        logger.info(f"\n{'='*80}")
        logger.info(f"Variant: {name}  (N={n}, M={m})")
        logger.info(f"{'='*80}")

        try:
            factor_df, price_df = prepare_single_instrument_factor(
                daily,
                lambda df, _n=n, _m=m: calculate_adx_factor(df, n=_n, m=_m),
                name
            )
            metrics, scenarios, evaluation = analyze_factor(
                name, factor_df, price_df, output_dir=out_dir
            )
            if metrics is not None:
                results[name] = {
                    'metrics': metrics,
                    'scenarios': scenarios,
                    'evaluation': evaluation,
                }
            else:
                logger.error(f"{name} analysis returned None")
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            import traceback
            traceback.print_exc()

    # 3. Comparison table
    if len(results) < 1:
        logger.error("No variants completed successfully")
        return

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 80)

    completed = [v for v in VARIANTS if v['name'] in results]
    rows = {
        'IC Mean':    [results[v['name']]['metrics']['ic_mean']           for v in completed],
        'ICIR':       [results[v['name']]['metrics']['icir']              for v in completed],
        'Rank IC':    [results[v['name']]['metrics']['rank_ic_mean']      for v in completed],
        'Rank ICIR':  [results[v['name']]['metrics']['rank_icir']         for v in completed],
        'Dir Acc':    [results[v['name']]['metrics']['directional_accuracy'] for v in completed],
        'Ann Return': [results[v['name']]['metrics']['annual_return']     for v in completed],
        'Sharpe':     [results[v['name']]['metrics']['sharpe_ratio']      for v in completed],
        'Max DD':     [results[v['name']]['metrics']['max_drawdown']      for v in completed],
        'L/S Sharpe': [results[v['name']]['scenarios']['long_short']['sharpe_ratio'] for v in completed],
        'Score':      [results[v['name']]['evaluation']['total_score']    for v in completed],
    }
    cols = [v['name'] for v in completed]

    comparison = pd.DataFrame(rows, index=cols).T
    logger.info(f"\n{comparison.round(4).to_string()}")

    # 4. Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    best_name = max(results, key=lambda k: results[k]['evaluation']['total_score'])
    for name, r in results.items():
        ev = r['evaluation']
        m = r['metrics']
        tag = " <-- BEST" if name == best_name else ""
        logger.info(f"  {name}: score={ev['total_score']:.2f}  IC={m['ic_mean']:.4f}  