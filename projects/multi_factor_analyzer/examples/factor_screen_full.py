"""
Full factor screening for rebar futures hourly data.
Computes IC, Rank IC, ICIR, best T, t-stat for 58 candidate factors.
Outputs top 20 results as JSON for document generation.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load data ───
DATA_PATH = "data/RB9999.XSGE.parquet"
df = pd.read_parquet(DATA_PATH)
print(f"Raw data: {df.shape}")

# Filter last 6 years (2020-2025)
df = df[df.index >= '2020-01-01'].copy()
print(f"After filter: {df.shape}")

# ─── 2. Resample to hourly ───
hourly = df.resample('1h').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
}).dropna()

# Manual open_interest: last value per hour
oi_hourly = df['open_interest'].resample('1h').last()
hourly['open_interest'] = oi_hourly

# VWAP approximation (money / volume)
hourly['vwap'] = (df['money'].resample('1h').sum() / df['volume'].resample('1h').sum())
hourly['vwap'] = hourly['vwap'].replace([np.inf, -np.inf], np.nan)

print(f"Hourly data: {hourly.shape}, range: {hourly.index.min()} ~ {hourly.index.max()}")
print(f"Columns: {hourly.columns.tolist()}")

# ─── 3. Factor computation ───
c = hourly['close'].astype('float64')
h = hourly['high'].astype('float64')
l_ = hourly['low'].astype('float64')
o = hourly['open'].astype('float64')
v = hourly['volume'].astype('float64')
oi = hourly['open_interest'].astype('float64')
vwap = hourly['vwap'].astype('float64')

ret = c.pct_change()

def rolling_skew(series, window):
    return series.rolling(window).skew()

def rolling_kurt(series, window):
    return series.rolling(window).kurt()

factors = {}

# === Momentum (4) ===
for w in [4, 8, 20, 44]:
    factors[f'mom_{w}'] = c / c.shift(w) - 1

# === Reversal (2) ===
factors['rev_4_44'] = (c / c.shift(4) - 1) - (c / c.shift(44) - 1)
factors['rev_8_44'] = (c / c.shift(8) - 1) - (c / c.shift(44) - 1)

# === Volatility (3) ===
realized_vol_20 = ret.rolling(20).std() * np.sqrt(44)  # annualized-ish
realized_vol_60 = ret.rolling(60).std() * np.sqrt(44)
factors['realized_vol_20'] = realized_vol_20
factors['realized_vol_60'] = realized_vol_60
factors['volz_20'] = (realized_vol_20 - realized_vol_20.rolling(60).mean()) / realized_vol_20.rolling(60).std()
factors['volz_60'] = (realized_vol_60 - realized_vol_60.rolling(60).mean()) / realized_vol_60.rolling(60).std()

# === Volume (2) ===
factors['vwap_dev'] = c / vwap - 1
factors['vol_ratio'] = v.rolling(20).mean() / v.rolling(60).mean()

# === MA deviation / crossover (4) ===
for w in [5, 10, 20, 44]:
    factors[f'ma_dev_{w}'] = (c - c.rolling(w).mean()) / c.rolling(w).mean()
factors['ma_cross_5_20'] = c.rolling(5).mean() - c.rolling(20).mean()
factors['ma_cross_10_44'] = c.rolling(10).mean() - c.rolling(44).mean()
factors['ma_cross_20_44'] = c.rolling(20).mean() - c.rolling(44).mean()
factors['ma_cross_5_44'] = c.rolling(5).mean() - c.rolling(44).mean()

# === Amplitude (1) ===
factors['amp_20'] = ((h - l_) / c).rolling(20).mean()
factors['amp_60'] = ((h - l_) / c).rolling(60).mean()

# === Higher-order stats (2) ===
factors['skew_20'] = rolling_skew(ret, 20)
factors['skew_60'] = rolling_skew(ret, 60)
factors['kurt_20'] = rolling_kurt(ret, 20)
factors['kurt_60'] = rolling_kurt(ret, 60)

# === Open Interest (2) ===
oi_change = oi.diff()
price_change = c.diff()
factors['oi_change'] = oi_change
factors['oi_change_pct'] = oi.pct_change()
factors['oi_price_corr'] = oi_change.rolling(20).corr(price_change)
factors['oi_price_corr_44'] = oi_change.rolling(44).corr(price_change)
factors['oi_momentum'] = oi / oi.shift(20) - 1
factors['oi_accel'] = oi.diff().diff()

# === Additional momentum variants ===
for w in [2, 12, 24, 36]:
    factors[f'mom_{w}'] = c / c.shift(w) - 1

# === Overnight / intraday ===
factors['overnight_gap'] = o / c.shift(1) - 1
factors['intraday_ret'] = c / o - 1

# === Price position in range ===
factors['close_loc_20'] = (c - l_.rolling(20).min()) / (h.rolling(20).max() - l_.rolling(20).min())
factors['close_loc_44'] = (c - l_.rolling(44).min()) / (h.rolling(44).max() - l_.rolling(44).min())

# === Volume-price relationship ===
factors['vp_corr_20'] = v.rolling(20).corr(ret)
factors['vp_corr_44'] = v.rolling(44).corr(ret)

# === Return acceleration ===
factors['ret_accel_4'] = ret - ret.shift(4)
factors['ret_accel_20'] = ret - ret.shift(20)

# === Price speed ===
factors['price_speed_4'] = (c - c.shift(4)) / 4
factors['price_speed_20'] = (c - c.shift(20)) / 20

# === Max drawdown in window ===
def rolling_max_dd(close, window):
    roll_max = close.rolling(window).max()
    dd = close / roll_max - 1
    return dd.rolling(window).min()

factors['max_dd_20'] = rolling_max_dd(c, 20)
factors['max_dd_44'] = rolling_max_dd(c, 44)

# === Range expansion ===
factors['range_exp_20'] = (h - l_).rolling(20).mean() / (h - l_).rolling(60).mean() - 1

# === Cumulative return variants ===
factors['cum_ret_4'] = ret.rolling(4).sum()
factors['cum_ret_20'] = ret.rolling(20).sum()

# === Normalized returns ===
factors['norm_ret_20'] = ret.rolling(20).mean() / ret.rolling(20).std()
factors['norm_ret_60'] = ret.rolling(60).mean() / ret.rolling(60).std()

print(f"\nTotal factors computed: {len(factors)}")
print(f"Factor names: {sorted(factors.keys())}")

# ─── 4. IC Decay computation ───
MAX_HORIZON = 44
IC_WINDOW = 120  # rolling IC window for ICIR

results = []

for fname, fval in factors.items():
    # Drop NaN to avoid spurious correlations
    valid = pd.DataFrame({'factor': fval, 'close': c}).dropna()
    if len(valid) < IC_WINDOW + MAX_HORIZON + 10:
        continue

    best_ic = 0
    best_rank_ic = 0
    best_icir = 0
    best_t = 1
    best_ic_val = 0
    best_rank_ic_val = 0

    for hh in range(1, MAX_HORIZON + 1):
        fwd_ret = valid['close'].shift(-hh) / valid['close'] - 1
        pair = pd.DataFrame({'factor': valid['factor'], 'fwd_ret': fwd_ret}).dropna()
        if len(pair) < 100:
            continue

        ic_val = pair['factor'].corr(pair['fwd_ret'])
        rank_ic_val = pair['factor'].rank().corr(pair['fwd_ret'].rank())

        # Rolling ICIR
        rolling_ic = pair['factor'].rolling(IC_WINDOW).corr(pair['fwd_ret']).dropna()
        if len(rolling_ic) > 20:
            icir_val = rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() > 0 else 0
        else:
            icir_val = 0

        # t-stat for raw IC
        n = len(pair)
        if n > 2:
            t_stat = abs(ic_val) * np.sqrt((n - 2) / (1 - ic_val**2)) if abs(ic_val) < 1 else 99
        else:
            t_stat = 0

        # Best by |IC| * sign(ICIR) — prefer high IC with consistent direction
        score = abs(ic_val) * (1 + 0.3 * abs(icir_val))
        if score > best_ic:
            best_ic = score
            best_ic_val = ic_val
            best_rank_ic_val = rank_ic_val
            best_icir = icir_val
            best_t = hh
            best_t_stat = t_stat

    results.append({
        'name': fname,
        'ic': round(best_ic_val, 4),
        'rank_ic': round(best_rank_ic_val, 4),
        'icir': round(best_icir, 2),
        'best_t': best_t,
        't_stat': round(best_t_stat, 1) if 'best_t_stat' in dir() else 0,
        'direction': '正' if best_ic_val > 0 else '负',
    })

# ─── 5. Rank and output ───
# Sort by: ICIR > 0 first, then by |IC|
results.sort(key=lambda x: (abs(x['icir']) if x['icir'] > 0 else 0, abs(x['ic'])), reverse=True)

# Also compute a composite score for final ranking
# Composite = |IC| * (1 + |ICIR|) weighted
for r in results:
    r['composite'] = round(abs(r['ic']) * (1 + abs(r['icir'])), 4)

results.sort(key=lambda x: x['composite'], reverse=True)

print("\n" + "="*80)
print(f"{'RANK':>4} | {'FACTOR':<22} | {'IC':>7} | {'RankIC':>7} | {'ICIR':>6} | {'T':>3} | {'t-stat':>6} | {'DIR':>3}")
print("-"*80)
for i, r in enumerate(results[:20], 1):
    print(f"{i:>4} | {r['name']:<22} | {r['ic']:>+7.4f} | {r['rank_ic']:>+7.4f} | {r['icir']:>+6.2f} | {r['best_t']:>3} | {r['t_stat']:>6.1f} | {r['direction']:>3}")

# Save to JSON
with open('/tmp/factor_top20.json', 'w') as f:
    json.dump(results[:20], f, indent=2, ensure_ascii=False)

print(f"\nTotal valid factors: {len(results)}")
print("Top 20 saved to /tmp/factor_top20.json")
