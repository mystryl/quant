"""
DMI 和 ADXR 指标监控工具
- 使用简单 SUM 和 MA（非 Wilder's Smoothing）
- DMI 使用 14 周期 (N=14)
- ADXR 使用 6 周期 (M=6)
- 同时计算 1 小时和 30 分钟数据
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import sys


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                   volume: pd.Series, N: int) -> pd.Series:
    """
    计算 VWAP 指标

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        N: VWAP 计算周期

    Returns:
        VWAP 值序列
    """
    # 计算典型价格
    typical_price = (high + low + close) / 3

    # 计算 VWAP = SUM(典型价格 * 成交量, N) / SUM(成交量, N)
    tp_volume = typical_price * volume
    vwaps = tp_volume.rolling(window=N).sum() / volume.rolling(window=N).sum()

    return vwaps


def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    计算 SuperTrend 指标

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: ATR 周期
        multiplier: ATR 倍数

    Returns:
        包含 SuperTrend 和趋势方向的 DataFrame
    """
    # 转换为 numpy 数组
    high_arr = high.values.astype(float)
    low_arr = low.values.astype(float)
    close_arr = close.values.astype(float)

    # 计算 True Range
    tr = np.maximum(high_arr - low_arr,
                    np.maximum(abs(high_arr - np.roll(close_arr, 1)),
                              abs(low_arr - np.roll(close_arr, 1))))
    tr[0] = high_arr[0] - low_arr[0]  # 第一个TR用高-低

    # 计算 ATR (简单移动平均)
    atr = np.zeros_like(close_arr)
    for i in range(len(close_arr)):
        start_idx = max(0, i - period + 1)
        atr[i] = np.mean(tr[start_idx:i+1])

    # 计算 HL2
    hl2 = (high_arr + low_arr) / 2

    # 计算基本带
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # 计算最终带（考虑前一周期状态）
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(close_arr)):
        if upper_band[i] < final_upper[i-1] or close_arr[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]

        if lower_band[i] > final_lower[i-1] or close_arr[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]

    # 计算 SuperTrend 线
    supertrend = np.zeros(len(close_arr))
    trend = np.zeros(len(close_arr))

    # 初始化
    supertrend[0] = final_upper[0]
    trend[0] = -1  # 初始假设下降趋势

    for i in range(1, len(close_arr)):
        if supertrend[i-1] == final_upper[i-1]:
            if close_arr[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
                trend[i] = -1  # 下降趋势
            else:
                supertrend[i] = final_lower[i]
                trend[i] = 1   # 上升趋势
        else:
            if close_arr[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
                trend[i] = 1   # 上升趋势
            else:
                supertrend[i] = final_upper[i]
                trend[i] = -1  # 下降趋势

    return pd.DataFrame({
        'supertrend': supertrend,
        'trend': trend,
        'upper_band': final_upper,
        'lower_band': final_lower
    }, index=close.index)


def calculate_dmi_adxr(high: pd.Series, low: pd.Series, close: pd.Series,
                      N: int = 14, M: int = 6) -> pd.DataFrame:
    """
    计算 DMI 和 ADXR 指标（使用简单 SUM 和 MA）

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        N: DMI 计算周期（默认 14）
        M: ADXR 计算周期（默认 6）

    Returns:
        包含 +DI, -DI, ADX, ADXR 的 DataFrame
    """
    n = len(high)

    # TR := SUM(MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(LOW-REF(CLOSE,1))),N)
    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        )
    )

    # 计算 TR 的 N 周期简单求和
    TR = tr.rolling(window=N).sum()

    # HD := HIGH-REF(HIGH,1)
    HD = high - high.shift(1)

    # LD := REF(LOW,1)-LOW
    LD = low.shift(1) - low

    # DMP := SUM(IFELSE(HD>0 && HD>LD,HD,0),N)
    DMP_cond = (HD > 0) & (HD > LD)
    DMP = np.where(DMP_cond, HD, 0)
    DMP = pd.Series(DMP).rolling(window=N).sum()

    # DMM := SUM(IFELSE(LD>0 && LD>HD,LD,0),N)
    DMM_cond = (LD > 0) & (LD > HD)
    DMM = np.where(DMM_cond, LD, 0)
    DMM = pd.Series(DMM).rolling(window=N).sum()

    # PDI: DMP*100/TR
    PDI = DMP * 100 / TR

    # MDI: DMM*100/TR
    MDI = DMM * 100 / TR

    # ADX: MA(ABS(MDI-PDI)/(MDI+PDI)*100,M)
    DX = abs(MDI - PDI) / (MDI + PDI) * 100
    ADX = DX.rolling(window=M).mean()

    # ADXR: (ADX+REF(ADX,M))/2
    ADXR = (ADX + ADX.shift(M)) / 2

    return pd.DataFrame({
        'plus_di': PDI,
        'minus_di': MDI,
        'dx': DX,
        'adx': ADX,
        'adxr': ADXR
    }, index=close.index)


def get_futures_data(symbol: str, period: str = '60', days: int = 60) -> pd.DataFrame:
    """
    使用 akshare 获取期货数据

    Args:
        symbol: 期货合约代码
        period: 周期类型（'30' = 30分钟，'60' = 1小时）
        days: 获取天数

    Returns:
        K线数据 DataFrame
    """
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)

        if df is None or df.empty:
            return pd.DataFrame()

        # 处理日期列
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            df['date'] = pd.to_datetime(df.index)

        # 筛选指定天数的数据
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['date'] >= cutoff_date].copy()

        # 选择需要的列
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols].copy()

        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"获取期货数据失败: {e}")
        return pd.DataFrame()


def analyze_indicator(symbol: str, period: str = '60', days: int = 60,
                     N: int = 14, M: int = 6, st_period: int = 10, st_multiplier: float = 3.0,
                     vwap_periods: list = None) -> Dict:
    """
    分析指定期货合约的 DMI、ADXR 和 SuperTrend 指标

    Args:
        symbol: 期货合约代码
        period: 周期类型
        days: 获取天数
        N: DMI 计算周期
        M: ADXR 计算周期
        st_period: SuperTrend ATR 周期
        st_multiplier: SuperTrend 倍数

    Returns:
        包含所有指标的分析结果
    """
    print(f"\n{'=' * 80}")
    print(f"正在分析 {symbol.upper()} {period}分钟 K线...")
    print(f"参数: DMI(N={N}), ADXR(M={M}), SuperTrend({st_period}, {st_multiplier}), VWAP({vwap_periods})")
    print(f"{'=' * 80}")

    df = get_futures_data(symbol, period, days)

    # 计算需要的最小数据量
    min_required = N + M * 2 + st_period
    if vwap_periods:
        min_required = max(min_required, max(vwap_periods))

    if df.empty or len(df) < min_required:
        print(f"错误: 数据不足，需要至少 {min_required} 条数据")
        print(f"当前数据条数: {len(df) if not df.empty else 0}")
        return None

    # 计算 DMI 和 ADXR
    dmi_df = calculate_dmi_adxr(df['high'], df['low'], df['close'], N=N, M=M)

    # 计算 SuperTrend
    st_df = calculate_supertrend(df['high'], df['low'], df['close'], st_period, st_multiplier)

    # 计算 VWAP（多周期）
    vwap_data = {}
    if vwap_periods:
        for vwap_n in vwap_periods:
            vwap_col = f'vwap_{vwap_n}'
            vwap_data[vwap_col] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'], vwap_n)

    # 合并指标
    df = pd.concat([df, dmi_df, st_df], axis=1)
    if vwap_data:
        df = pd.concat([df, pd.DataFrame(vwap_data, index=df.index)], axis=1)

    # 获取最新数据（跳过前 N + M * 2 条数据）
    min_periods = N + M * 2
    df_valid = df.iloc[min_periods:].copy()

    if len(df_valid) == 0:
        print(f"错误: 有效数据不足")
        return None

    # 获取最新数据
    latest = df_valid.iloc[-1]
    latest_datetime = df_valid['date'].iloc[-1]

    # 判断趋势强度（基于 ADX）
    adx_value = latest['adx']
    if pd.isna(adx_value) or adx_value < 20:
        trend_strength = "弱"
        trend_note = "市场缺乏明确趋势，建议观望"
    elif adx_value < 25:
        trend_strength = "中"
        trend_note = "趋势正在形成中"
    elif adx_value < 40:
        trend_strength = "强"
        trend_note = "趋势明确，可考虑跟随趋势"
    else:
        trend_strength = "极强"
        trend_note = "趋势非常强劲"

    # ADXR 判断
    adxr_value = latest['adxr']
    if pd.isna(adxr_value):
        adxr_note = "数据不足"
    elif adxr_value > 30:
        adxr_note = "长期趋势强劲"
    elif adxr_value > 20:
        adxr_note = "长期趋势明确"
    else:
        adxr_note = "长期趋势不明"

    # DMI 信号判断
    plus_di = latest['plus_di']
    minus_di = latest['minus_di']
    if plus_di > minus_di:
        dmi_signal = "看涨 ↑"
        di_diff = plus_di - minus_di
    else:
        dmi_signal = "看跌 ↓"
        di_diff = minus_di - plus_di

    # SuperTrend 信号判断
    st_trend = "上升" if latest['trend'] == 1 else "下降"
    st_distance = latest['close'] - latest['supertrend']
    st_distance_pct = (st_distance / latest['close']) * 100

    # VWAP 信号判断
    vwap_results = {}
    if vwap_periods:
        for vwap_n in vwap_periods:
            vwap_col = f'vwap_{vwap_n}'
            if vwap_col in latest and not pd.isna(latest[vwap_col]):
                vwap_value = latest[vwap_col]
                vwap_distance = latest['close'] - vwap_value
                vwap_distance_pct = (vwap_distance / latest['close']) * 100
                vwap_results[vwap_n] = {
                    'value': vwap_value,
                    'distance': vwap_distance,
                    'distance_pct': vwap_distance_pct,
                    'position': '上方' if vwap_distance > 0 else '下方'
                }

    return {
        'symbol': symbol,
        'period': period,
        'datetime': latest_datetime,
        'price': latest['close'],
        'volume': latest['volume'],
        'dmi': {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx_value,
            'adxr': adxr_value,
            'dx': latest['dx'],
            'trend_strength': trend_strength,
            'trend_note': trend_note,
            'adxr_note': adxr_note,
            'signal': dmi_signal,
            'di_diff': di_diff
        },
        'supertrend': {
            'value': latest['supertrend'],
            'trend': st_trend,
            'distance': st_distance,
            'distance_pct': st_distance_pct,
            'upper_band': latest['upper_band'],
            'lower_band': latest['lower_band']
        },
        'vwap': vwap_results,
        'dataframe': df_valid
    }


def print_analysis(analysis: Dict, detailed: bool = False):
    """
    打印分析结果

    Args:
        analysis: 分析结果字典
        detailed: 是否显示详细信息
    """
    if analysis is None:
        print("分析失败")
        return

    # 简洁模式：只显示基本信息
    if not detailed:
        return

    # 详细模式：显示完整分析报告
    print(f"\n{'=' * 80}")
    print(f"指标分析报告 - {analysis['symbol'].upper()} ({analysis['period']}分钟)")
    print(f"{'=' * 80}")

    print(f"\n【时间】{analysis['datetime']}")
    print(f"【当前价格】{analysis['price']:.2f}")
    print(f"【成交量】{int(analysis['volume']):,}")

    # DMI & ADXR 指标
    print(f"\n{'-' * 80}")
    print("【DMI & ADXR 指标】")
    print(f"{'-' * 80}")
    dmi = analysis['dmi']
    print(f"ADX: {dmi['adx']:.2f}  ({dmi['trend_strength']})")
    print(f"ADXR: {dmi['adxr']:.2f}  ({dmi['adxr_note']})")

    # SuperTrend 指标
    print(f"\n{'-' * 80}")
    print("【SuperTrend 指标】")
    print(f"{'-' * 80}")
    st = analysis['supertrend']
    print(f"SuperTrend: {st['value']:.2f}")
    print(f"趋势方向: {st['trend']}")
    print(f"距离 ST: {st['distance']:+.2f} ({st['distance_pct']:+.2f}%)")

    # VWAP 指标
    if 'vwap' in analysis and analysis['vwap']:
        print(f"\n{'-' * 80}")
        print("【VWAP 指标】")
        print(f"{'-' * 80}")
        for period_n, vwap_data in analysis['vwap'].items():
            print(f"VWAP({period_n}): {vwap_data['value']:.2f}")
            print(f"  位置: 价格在 VWAP {vwap_data['position']}")
            print(f"  距离: {vwap_data['distance']:+.2f} ({vwap_data['distance_pct']:+.2f}%)")

    # 综合信号
    print(f"\n{'-' * 80}")
    print("【综合信号】")
    print(f"{'-' * 80}")
    print(f"DMI: {dmi['signal']}  (ADX={dmi['adx']:.2f}, ADXR={dmi['adxr']:.2f})")
    print(f"SuperTrend: {st['trend']}  (ST={st['value']:.2f})")

    # 交易建议
    print(f"\n{'-' * 80}")
    print("【分析与建议】")
    print(f"{'-' * 80}")

    adx = dmi['adx']
    adxr = dmi['adxr']
    st_up = st['trend'] == "上升"
    dmi_up = dmi['signal'] == "看涨 ↑"

    # 综合判断
    if pd.isna(adx) or adx < 20:
        print(f"⚠️  ADX < 20，市场震荡")
        print(f"  建议: 观望等待")
    elif adx < 25:
        print(f"⚡  ADX 在 20-25，趋势形成中")
        if st_up and dmi_up:
            print(f"  ST 和 DMI 均看涨，可逢低做多")
        else:
            print(f"  信号不一致，观望")
    else:
        print(f"✓  ADX > 25，趋势明确")
        if st_up and dmi_up:
            print(f"  ST 和 DMI 双重看涨，可积极做多")
        elif not st_up and not dmi_up:
            print(f"  ST 和 DMI 双重看跌，可积极做空")
        else:
            print(f"  ST 和 DMI 信号冲突，观望")

    if not pd.isna(adxr):
        if adxr > 20:
            print(f"\n✓  ADXR > 20，长期趋势明确，趋势持续性较好")
        else:
            print(f"\n⚠️  ADXR < 20，长期趋势不明")

    print(f"{'=' * 80}")


def main():
    """
    主函数
    """
    # 配置参数
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'HC0'  # 默认使用热卷主力连续
    periods = ['60', '15']  # 1小时、15分钟
    days = 60  # 获取60天数据

    # DMI 和 ADXR 参数
    N = 14  # DMI 周期
    M = 6   # ADXR 周期

    # SuperTrend 参数
    st_params = {
        '60': {'period': 7, 'multiplier': 3.0},
        '15': {'period': 10, 'multiplier': 3.0}
    }

    # VWAP 参数（监控40和60周期）
    vwap_periods = [40, 60]

    print("=" * 80)
    print("多周期指标监控系统 (DMI / ADXR / SuperTrend / VWAP)")
    print("=" * 80)
    print(f"监控合约: {symbol.upper()}")
    print(f"监控周期: {', '.join(periods)} 分钟")
    print(f"数据天数: {days} 天")
    print(f"参数设置: N={N} (DMI), M={M} (ADX平均, ADXR)")
    print(f"SuperTrend: 60min(7,3.0), 15min(10,3.0)")
    print(f"VWAP 周期: {', '.join(map(str, vwap_periods))}")
    print("=" * 80)

    results = {}

    # 分析每个周期
    for period in periods:
        try:
            st_period = st_params[period]['period']
            st_multiplier = st_params[period]['multiplier']

            analysis = analyze_indicator(
                symbol=symbol,
                period=period,
                days=days,
                N=N,
                M=M,
                st_period=st_period,
                st_multiplier=st_multiplier,
                vwap_periods=vwap_periods
            )

            if analysis:
                results[period] = analysis
                print_analysis(analysis)

        except Exception as e:
            print(f"\n错误: 分析 {period} 分钟周期失败: {e}")
            import traceback
            traceback.print_exc()

    # 多周期综合分析
    if len(results) > 1:
        print(f"\n{'=' * 80}")
        print("【多周期指标对比】")
        print(f"{'=' * 80}")

        periods_list = list(results.keys())
        period_names = ['15分钟', '60分钟']

        # 准备数据
        data_15 = results.get('15')
        data_60 = results.get('60')

        if data_15 and data_60:
            # 构建表格
            print(f"\n{'指标':<20} {'15分钟':<25} {'60分钟':<25}")
            print("-" * 70)

            # 当前价格和时间
            print(f"{'时间':<20} {str(data_15['datetime']):<25} {str(data_60['datetime']):<25}")
            print(f"{'当前价格':<20} {data_15['price']:>10.2f}{'':<14} {data_60['price']:>10.2f}")

            # ADX 和 ADXR
            print(f"{'ADX':<20} {data_15['dmi']['adx']:>10.2f}{'':<14} {data_60['dmi']['adx']:>10.2f}")
            print(f"{'ADXR':<20} {data_15['dmi']['adxr']:>10.2f}{'':<14} {data_60['dmi']['adxr']:>10.2f}")

            # SuperTrend (阻力或支撑)
            st_type_15 = '支撑' if data_15['supertrend']['distance'] > 0 else '阻力'
            st_type_60 = '支撑' if data_60['supertrend']['distance'] > 0 else '阻力'
            print(f"{'SuperTrend':<20} {data_15['supertrend']['value']:>10.2f} ({st_type_15}){'':<4} {data_60['supertrend']['value']:>10.2f} ({st_type_60})")

            # VWAP 40 和 60
            vwap_15 = data_15.get('vwap', {})
            vwap_60 = data_60.get('vwap', {})

            if 40 in vwap_15 and 40 in vwap_60:
                vwap40_pos_15 = '上方' if vwap_15[40]['distance'] > 0 else '下方'
                vwap40_pos_60 = '上方' if vwap_60[40]['distance'] > 0 else '下方'
                print(f"{'VWAP(40)':<20} {vwap_15[40]['value']:>10.2f} ({vwap40_pos_15}){'':<4} {vwap_60[40]['value']:>10.2f} ({vwap40_pos_60})")

            if 60 in vwap_15 and 60 in vwap_60:
                vwap60_pos_15 = '上方' if vwap_15[60]['distance'] > 0 else '下方'
                vwap60_pos_60 = '上方' if vwap_60[60]['distance'] > 0 else '下方'
                print(f"{'VWAP(60)':<20} {vwap_15[60]['value']:>10.2f} ({vwap60_pos_15}){'':<4} {vwap_60[60]['value']:>10.2f} ({vwap60_pos_60})")

            # SuperTrend 趋势
            print(f"{'SuperTrend趋势':<20} {data_15['supertrend']['trend']:<25} {data_60['supertrend']['trend']:<25}")

            # DMI 信号
            print(f"{'DMI信号':<20} {data_15['dmi']['signal']:<25} {data_60['dmi']['signal']:<25}")

            # 综合分析
            print("\n" + "-" * 70)
            print("【综合分析】")
            print("-" * 70)

            # ADX 判断
            adx_15 = data_15['dmi']['adx']
            adx_60 = data_60['dmi']['adx']

            if adx_15 < 20 and adx_60 < 20:
                print("⚠️  双周期 ADX < 20，市场震荡，建议观望")
            elif adx_15 > 25 and adx_60 > 25:
                print("✓  双周期 ADX > 25，趋势明确")
            else:
                print(f"⚡  15分钟 ADX: {adx_15:.2f}, 60分钟 ADX: {adx_60:.2f}")

            # 趋势一致性
            st_trend_15 = data_15['supertrend']['trend']
            st_trend_60 = data_60['supertrend']['trend']
            dmi_signal_15 = data_15['dmi']['signal']
            dmi_signal_60 = data_60['dmi']['signal']

            if st_trend_15 == st_trend_60 and '看涨' in dmi_signal_15 and '看涨' in dmi_signal_60:
                print("✓  趋势一致向上，可积极做多")
            elif st_trend_15 == st_trend_60 and '看跌' in dmi_signal_15 and '看跌' in dmi_signal_60:
                print("✓  趋势一致向下，可积极做空")
            else:
                print("⚠️  双周期趋势不一致，建议观望")

        print("=" * 80)

        # 输出详细报告
        print(f"\n{'=' * 80}")
        print("【详细分析报告】")
        print(f"{'=' * 80}")
        for period in periods_list:
            print_analysis(results[period], detailed=True)

    print("\n分析完成！")


if __name__ == '__main__':
    main()
