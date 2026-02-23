#!/usr/bin/env python3
"""
批量生成所有品种的K线图+信号标记

功能：
1. 为每个品种生成独立的K线图
2. 标记信号类型（上涨=绿色、下跌=红色、震荡=灰色）
3. 显示MACD指标
4. 支持交互式缩放和查看

品种：
- HC8888.XSGE (热卷)
- I8888.XDCE (铁矿石)
- AU8888.XSGE (黄金)
- CF8888.XZCE (郑棉)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'plotly'])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置
project_root = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry')
data_dir = Path('/Users/mystryl/Library/CloudStorage/Dropbox/润富/钢铁/code/期货相关代码/futures_data_fetcher/futures_data/60min')
signal_file = project_root / 'predictions' / '2026_signals' / '所有K线信号_2026.xlsx'
output_dir = project_root / 'predictions' / '2026_signals' / 'charts'

SYMBOL_CONFIG = {
    'HC888': {'full_code': 'HC8888.XSGE', 'name': '热卷', 'file': 'HC888.parquet'},
    'I888': {'full_code': 'I8888.XDCE', 'name': '铁矿石', 'file': 'I888.parquet'},
    'AU888': {'full_code': 'AU8888.XSGE', 'name': '黄金', 'file': 'AU888.parquet'},
    'CF888': {'full_code': 'CF8888.XZCE', 'name': '郑棉', 'file': 'CF888.parquet'},
}


def visualize_symbol(symbol, config):
    """可视化单个品种"""
    logger.info(f"\n{'='*60}")
    logger.info(f"处理品种: {config['name']} ({symbol})")
    logger.info(f"{'='*60}")

    # 1. 加载K线数据
    kline_file = data_dir / config['file']
    if not kline_file.exists():
        logger.warning(f"K线文件不存在: {kline_file}")
        return None

    df_kline = pd.read_parquet(kline_file)
    df_kline['datetime'] = pd.to_datetime(df_kline['date'])
    df_kline = df_kline[df_kline['datetime'] >= '2025-10-01']
    df_kline = df_kline.sort_values('datetime')

    logger.info(f"  K线数据: {len(df_kline)} 条")

    # 2. 加载信号数据
    df_signals = pd.read_excel(signal_file)
    df_signals['时间'] = pd.to_datetime(df_signals['时间'])
    df_signals_symbol = df_signals[df_signals['合约代码'] == config['full_code']].copy()
    df_signals_symbol = df_signals_symbol.sort_values('时间')

    logger.info(f"  信号数据: {len(df_signals_symbol)} 条")

    if len(df_signals_symbol) == 0:
        logger.warning(f"没有找到信号数据")
        return None

    # 3. 合并数据
    df_signals_clean = df_signals_symbol[['时间', '信号类型', 'P(震荡)', 'P(上涨)', 'P(下跌)', '最高概率']].copy()
    df_signals_clean.columns = ['datetime', 'signal', 'p_range', 'p_up', 'p_down', 'p_max']

    df_merged = pd.merge(df_kline, df_signals_clean, on='datetime', how='left')

    # 4. 添加颜色
    def get_color(signal):
        if signal == '上涨':
            return 'green'
        elif signal == '下跌':
            return 'red'
        elif signal == '震荡':
            return 'gray'
        return 'white'

    df_merged['signal_color'] = df_merged['signal'].apply(get_color)

    # 5. 创建图表
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{config["name"]} - K线图+信号标记', 'MACD指标')
    )

    # K线图（红涨绿跌）
    fig.add_trace(
        go.Candlestick(
            x=df_merged['datetime'],
            open=df_merged['open'],
            high=df_merged['high'],
            low=df_merged['low'],
            close=df_merged['close'],
            name='K线',
            increasing_line_color='red',      # 上涨红色
            decreasing_line_color='green',    # 下跌绿色
            increasing_fillcolor='red',
            decreasing_fillcolor='green'
        ),
        row=1, col=1
    )

    # 信号标记（只显示上涨和下跌，不显示震荡，都在收盘价）
    for signal_type in ['上涨', '下跌']:
        df_signal = df_merged[df_merged['signal'] == signal_type]

        if len(df_signal) > 0:
            # 红涨绿跌，使用三角标记
            color_map = {'上涨': 'red', '下跌': 'green'}
            marker_symbol = 'triangle-up' if signal_type == '上涨' else 'triangle-down'

            fig.add_trace(
                go.Scatter(
                    x=df_signal['datetime'],
                    y=df_signal['close'],  # 所有标记都在收盘价
                    mode='markers',
                    name=f'{signal_type}信号',
                    marker=dict(
                        symbol=marker_symbol,
                        size=12,
                        color=color_map[signal_type],
                        opacity=0.9,
                        line=dict(width=2, color='white')
                    ),
                    text=df_signal['p_max'].apply(lambda x: f'P={x:.2f}' if pd.notna(x) else ''),
                    hovertemplate='<b>%{x}</b><br>信号: %{fullData.name}<br>收盘: %{y:.2f}<br>开:%{customdata[0]:.2f} 高:%{customdata[1]:.2f} 低:%{customdata[2]:.2f}<br>%{text}<extra></extra>',
                    customdata=df_signal[['open', 'high', 'low']].values
                ),
                row=1, col=1
            )

    # MACD
    if 'macd' in df_merged.columns:
        fig.add_trace(
            go.Scatter(
                x=df_merged['datetime'],
                y=df_merged['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )

    if 'macd_signal' in df_merged.columns:
        fig.add_trace(
            go.Scatter(
                x=df_merged['datetime'],
                y=df_merged['macd_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='orange', width=1)
            ),
            row=2, col=1
        )

    if 'macd_histogram' in df_merged.columns:
        fig.add_trace(
            go.Bar(
                x=df_merged['datetime'],
                y=df_merged['macd_histogram'],
                name='Histogram',
                marker_color=['red' if x < 0 else 'green' for x in df_merged['macd_histogram']],
                opacity=0.5
            ),
            row=2, col=1
        )

    # 布局
    fig.update_layout(
        title={
            'text': f'{config["name"]} - 趋势信号可视化 (2025-10-01 至 2026-02-13)',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_rangeslider_visible=False,
        height=1000,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    fig.update_xaxes(title_text='时间')
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='MACD', row=2, col=1)

    # 6. 保存图表
    output_file = output_dir / f'{config["name"]}_K线图_信号标记.html'
    fig.write_html(output_file)

    # 7. 统计
    stats = {
        '总K线数': len(df_merged),
        '上涨信号': (df_merged['signal'] == '上涨').sum(),
        '下跌信号': (df_merged['signal'] == '下跌').sum(),
        '震荡信号': (df_merged['signal'] == '震荡').sum(),
        '趋势信号占比': ((df_merged['signal'] == '上涨').sum() + (df_merged['signal'] == '下跌').sum()) / len(df_merged) * 100,
        '最高价': df_merged['close'].max(),
        '最低价': df_merged['close'].min(),
    }

    logger.info(f"  ✓ 图表已保存: {output_file}")
    logger.info(f"  信号分布: 上涨={stats['上涨信号']}, 下跌={stats['下跌信号']}, 震荡={stats['震荡信号']}")
    logger.info(f"  趋势信号占比: {stats['趋势信号占比']:.1f}%")

    return fig, stats


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("批量生成K线图+信号标记")
    logger.info("="*60)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    # 处理每个品种
    for symbol, config in SYMBOL_CONFIG.items():
        try:
            fig, stats = visualize_symbol(symbol, config)
            if stats:
                all_stats[config['name']] = stats
        except Exception as e:
            logger.error(f"处理 {symbol} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 汇总统计
    logger.info("\n" + "="*60)
    logger.info("汇总统计")
    logger.info("="*60)

    for name, stats in all_stats.items():
        logger.info(f"\n{name}:")
        logger.info(f"  总K线: {stats['总K线数']}")
        logger.info(f"  趋势信号占比: {stats['趋势信号占比']:.1f}%")
        logger.info(f"    - 上涨: {stats['上涨信号']} ({stats['上涨信号']/stats['总K线数']*100:.1f}%)")
        logger.info(f"    - 下跌: {stats['下跌信号']} ({stats['下跌信号']/stats['总K线数']*100:.1f}%)")
        logger.info(f"    - 震荡: {stats['震荡信号']} ({stats['震荡信号']/stats['总K线数']*100:.1f}%)")
        logger.info(f"  价格区间: {stats['最低价']:.2f} ~ {stats['最高价']:.2f}")

    logger.info(f"\n输出目录: {output_dir}")

    return all_stats


if __name__ == '__main__':
    stats = main()
