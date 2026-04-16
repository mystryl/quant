"""
ADX/DMI Factor Test Script

Tests the ADX (Average Directional Index) / DMI (Directional Movement Index)
trading system as a continuous factor across 3 parameter variants:
  - ADX_N7:  N=7,  M=3 (fast)
  - ADX_N14: N=14, M=6 (standard)
  - ADX_N21: N=21, M=9 (slow)

Signal encoding:
  Strongest bullish  (factor ~ 1.0)  : ADX crosses 20 upward + DI+ > DI-
  Standard bullish   (factor ~ 0.6-0.8): recent golden cross + ADX > 20
  Trend continuation (factor ~ 0.3-0.5): DI+ > DI- + ADX > 25
  Neutral / choppy   (factor ~ 0)    : ADX < 20
  Bearish            (factor ~ -0.6--1.0): death cross + ADX > 20 + declining
  Exit signal        (factor → 0)    : ADX declining from bullish position
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
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
    """
    Calculate ADX/DMI components.

    Returns: (pdi, mdi, adx, adxr) as pd.Series
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # --- True Range ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # --- Directional Movement ---
    hd = high - high.shift(1)   # +DM directional component
    ld = low.shift(1) - low     # -DM directional component

    dmp = pd.Series(np.nan, index=df.index, dtype=float)
    dmm = pd.Series(np.nan, index=df.index, dtype=float)

    # Default: only positive moves count
    dmp = hd.where((hd > 0) & (hd >= ld), 0.0)
    dmm = ld.where((ld > 0) & (ld > hd), 0.0)

    # --- Wilder's smoothing via rolling SUM (first N bars) ---
    tr_sum = tr.rolling(n, min_periods=n).sum()
    dmp_sum = dmp.rolling(n, min_periods=n).sum()
    dmm_sum = dmm.rolling(n, min_periods=n).sum()

    # --- DI+ and DI- ---
    pdi = (dmp_sum / tr_sum) * 100
    mdi = (dmm_sum / tr_sum) * 100

    # --- DX and ADX ---
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
    """
    Generate a continuous factor value from ADX/DMI trading signals.

    Factor range approximately [-1, +1].
    """
    pdi, mdi, adx, adxr = calculate_adx_dmi(df, n, m)

    # Cross detections
    golden_cross = (pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))
    death_cross = (mdi > pdi) & (mdi.shift(1) <= pdi.shift(1))

    bars_since_golden = _bars_since(golden_cross)
    bars_since_death = _bars_since(death_cross)

    adx_cross_up = (adx > 20) & (adx.shift(1) <= 20)
    adx_declining = adx < adx.shift(1)

    factor = pd.Series(0.0, index=df.index)

    # ---- Strongest bullish: ADX just crossed 20 upward + DI+ on top ----
    mask = adx_cross_up & (pdi > mdi)
    factor[mask] = 1.0

    # ---- Standard bullish: recent golden cross + ADX > 20 ----
    mask = (bars_since_golden <= lookback) & (bars_since_golden > 0) & (adx > 20)
    strength = (1 - bars_since_golden[mask] / lookback) * 0.6
    adx_boost = np.clip((adx[mask] - 20) / 20, 0, 0.4)
    factor[mask] = np.maximum(factor[mask], strength + adx_boost)

    # ---- Bullish trend continuation: DI+ > DI- AND ADX > 25 ----
    mask = ((pdi > mdi) & (adx > 25) &
            (bars_since_golden > lookback) & (bars_since_death > lookback))
    factor[mask] = np.clip((adx[mask] - 25) / 30, 0.1, 0.5)

    # ---- Exit signal: pull toward 0 when ADX declining ----
    mask = (factor > 0) & adx_declining
    factor[mask] *= 0.3

    # ---- Bearish: death cross + ADX > 20 + ADX declining ----
    mask = ((bars_since_death <= lookback) & (bars_since_death > 0) &
            (adx > 20) & adx_declining)
    strength = (1 - bars_since_death[mask] / lookback) * 0.6
    adx_boost = np.clip((adx[mask] - 20) / 20, 0, 0.4)
    factor[mask] = -(strength + adx_boost)

    # ---- Suppress in choppy market (ADX < 20) ----
    factor[adx < 20] = 0.0

    return factor


# ---------------------------------------------------------------------------
# Mock data generator (same as KDJ script)
# ---------------------------------------------------------------------------

def create_mock_data(instruments=None, start_date='2020-01-01',
                     end_date='2023-12-31') -> Dict:
    if instruments is None:
        instruments = ['STOCK001', 'STOCK002', 'STOCK003', 'STOCK004', 'STOCK005']

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]

    np.random.seed(42)
    data = {}

    for instrument in instruments:
        returns = np.random.normal(0.0005, 0.02, len(dates))
        price = 100 * np.cumprod(1 + returns)

        high = price * (1 + np.abs(np.random.uniform(0, 0.02, len(dates))))
        low = price * (1 - np.abs(np.random.uniform(0, 0.02, len(dates))))

        df = pd.DataFrame({
            'open': price * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)

        df['high'] = df[['open', 'close']].max(axis=1) * (
            1 + np.abs(np.random.uniform(0, 0.01, len(dates))))
        df['low'] = df[['open', 'close']].min(axis=1) * (
            1 - np.abs(np.random.uniform(0, 0.01, len(dates))))

        data[instrument] = df

    logger.info(f"Created mock data for {len(data)} instruments, "
                f"{dates[0].date()} to {dates[-1].date()}")
    return data


# ---------------------------------------------------------------------------
# Factor data preparation (MultiIndex, from_arrays)
# ---------------------------------------------------------------------------

def prepare_factor_data(data_dict: Dict, factor_func, factor_name: str):
    factor_data = []
    price_data = []

    for instrument, df in data_dict.items():
        try:
            factor = factor_func(df)
            factor = factor.dropna()

            n = len(factor)
            index = pd.MultiIndex.from_arrays(
                [factor.index, [instrument] * n],
                names=['datetime', 'instrument']
            )

            factor_data.append(
                pd.DataFrame({'factor': factor.values}, index=index))
            price_data.append(
                pd.DataFrame({'close': df.loc[factor.index, 'close'].values},
                             index=index))
        except Exception as e:
            logger.warning(f"{instrument} factor calc failed: {e}")
            continue

    if not factor_data:
        raise ValueError(f"No factor data computed for {factor_name}")

    factor_df = pd.concat(factor_data).dropna()
    price_df = pd.concat(price_data).dropna()

    logger.info(f"{factor_name}: {len(factor_df)} data points ready")
    return factor_df, price_df


# ---------------------------------------------------------------------------
# Full analysis pipeline (same pattern as KDJ script)
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

        # 3. Performance evaluation
        logger.info("\n3. Performance evaluation...")
        evaluator = PerformanceEvaluator()
        pred = factor_final.iloc[:, 0] if isinstance(factor_final, pd.DataFrame) else factor_final
        label = returns_final.iloc[:, 0] if isinstance(returns_final, pd.DataFrame) else returns_final
        metrics = evaluator.calculate_all(pred, label)

        logger.info(f"   IC mean   : {metrics['ic_mean']:.4f}")
        logger.info(f"   ICIR      : {metrics['icir']:.4f}")
        logger.info(f"   Rank IC   : {metrics['rank_ic_mean']:.4f}")
        logger.info(f"   Rank ICIR : {metrics['rank_icir']:.4f}")

        # 4. Strategy scenario analysis
        logger.info("\n4. Strategy scenario analysis...")
        sa = StrategyAnalyzer()

        bull = sa.analyze_bull_strategy(factor_final, returns_final, top_pct=0.2)
        bear = sa.analyze_bear_strategy(factor_final, returns_final, bottom_pct=0.2)
        ls = sa.analyze_long_short_strategy(factor_final, returns_final,
                                           top_pct=0.2, bottom_pct=0.2)

        logger.info(f"   Bull annual return : {bull['annual_return']:.2%}")
        logger.info(f"   Bull sharpe        : {bull['sharpe_ratio']:.2f}")
        logger.info(f"   Bear annual return : {bear['annual_return']:.2%}")
        logger.info(f"   Bear sharpe        : {bear['sharpe_ratio']:.2f}")
        logger.info(f"   L/S  annual return : {ls['annual_return']:.2%}")
        logger.info(f"   L/S  sharpe        : {ls['sharpe_ratio']:.2f}")

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
# Main: run 3 parameter variants and compare
# ---------------------------------------------------------------------------

VARIANTS = [
    {'name': 'ADX_N7',  'n': 7,  'm': 3, 'output': './output/adx_n7'},
    {'name': 'ADX_N14', 'n': 14, 'm': 6, 'output': './output/adx_n14'},
    {'name': 'ADX_N21', 'n': 21, 'm': 9, 'output': './output/adx_n21'},
]


def main():
    logger.info("\n" + "=" * 80)
    logger.info("ADX/DMI Factor Test — 3 Parameter Variants")
    logger.info("=" * 80)

    # 1. Create mock data
    logger.info("\nCreating mock data...")
    data_dict = create_mock_data(
        instruments=['STOCK001', 'STOCK002', 'STOCK003', 'STOCK004', 'STOCK005'],
        start_date='2020-01-01',
        end_date='2023-12-31'
    )

    # 2. Run each variant
    results = {}
    for v in VARIANTS:
        name, n, m, out_dir = v['name'], v['n'], v['m'], v['output']
        logger.info(f"\n{'='*80}")
        logger.info(f"Variant: {name}  (N={n}, M={m})")
        logger.info(f"{'='*80}")

        try:
            factor_df, price_df = prepare_factor_data(
                data_dict,
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

    rows = {
        'IC Mean':    [results[v['name']]['metrics']['ic_mean']    for v in VARIANTS if v['name'] in results],
        'ICIR':       [results[v['name']]['metrics']['icir']       for v in VARIANTS if v['name'] in results],
        'Rank IC':    [results[v['name']]['metrics']['rank_ic_mean'] for v in VARIANTS if v['name'] in results],
        'Rank ICIR':  [results[v['name']]['metrics']['rank_icir']  for v in VARIANTS if v['name'] in results],
        'Bull Return':[results[v['name']]['scenarios']['bull']['annual_return'] for v in VARIANTS if v['name'] in results],
        'L/S Return': [results[v['name']]['scenarios']['long_short']['annual_return'] for v in VARIANTS if v['name'] in results],
        'L/S Sharpe': [results[v['name']]['scenarios']['long_short']['sharpe_ratio'] for v in VARIANTS if v['name'] in results],
        'Score':      [results[v['name']]['evaluation']['total_score'] for v in VARIANTS if v['name'] in results],
    }
    cols = [v['name'] for v in VARIANTS if v['name'] in results]

    comparison = pd.DataFrame(rows, index=cols).T
    logger.info(f"\n{comparison.round(4).to_string()}")

    # 4. Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    best_name = max(results, key=lambda k: results[k]['evaluation']['total_score'])
    for name, r in results.items():
        ev = r['evaluation']
        tag = " <-- BEST" if name == best_name else ""
        logger.info(f"  {name}: score={ev['total_score']:.2f}  "
                    f"reliability={ev['reliability']}{tag}")

    logger.info(f"\nBest variant: {best_name} "
                f"(score={results[best_name]['evaluation']['total_score']:.2f})")
    logger.info("\n" + "=" * 80)
    logger.info("Done. Reports saved to ./output/adx_n*/")
    logger.info("=" * 80 + "\n")


if __name__ == '__main__':
    main()
