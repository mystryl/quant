#!/usr/bin/env python3
"""
K线图+信号标记可视化

功能：
1. 绘制K线图（蜡烛图）
2. 标记信号类型（上涨=绿色、下跌=红色、震荡=灰色）
3. 显示MACD指标
4. 支持交互式缩放和查看

品种：热卷（HC8888.XSGE）
时间范围：2025-10-01 至 2026-02-13
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 尝试导入plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly未安装，正在安装...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'plotly'])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 路径配置
project_root = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry')
data_dir = Path('/Users/mystryl/Library/CloudStorage/Dropbox/润富/钢铁/code/期货相关代码/futures_data_fetcher/futures_data/60min')
signal_file = project_root / 'predictions' / '2026_signals' / '所有K线信号_2026.xlsx'
output_file = project_root / 'predictions' / '2026_signals' / '热卷K线图_信号标记.html'


def load_data():
    """加载K线数据和信号数据"""
    logger.info("加载数据...")

    # 1. 加载K线数据
    kline_file = data_dir / 'HC888.parquet'
    df_kline = pd.read_parquet(kline_file)
    df_kline['datetime'] = pd.to_datetime(df_kline['date'])

    # 筛选时间范围
    df_kline = df_kline[df_kline['datetime'] >= '2025-10-01']
    df_kline = df_kline.sort_values('datetime')

    logger.info(f"  K线数据: {len(df_kline)} 条")
    logger.info(f"  时间范围: {df_kline['datetime'].min()} ~ {df_kline['datetime'].max()}")

    # 2. 加载信号数据
    df_signals = pd.read_excel(signal_file)
    df_signals['时间'] = pd.to_datetime(df_signals['时间'])

    # 筛选热卷
    df_signals = df_signals[df_signals['合约代码'] == 'HC8888.XSGE']
    df_signals = df_signals.sort_values('时间')

    logger.info(f"  信号数据: {len(df_signals)} 条")

    return df_kline, df_signals


def merge_signals(df_kline, df_signals):
    """合并K线数据和信号"""
    logger.info("合并数据...")

    # 保留需要的列
    df_signals_clean = df_signals[['时间', '信号类型', 'P(震荡)', 'P(上涨)', 'P(下跌)', '最高概率']].copy()
    df_signals_clean.columns = ['datetime', 'signal', 'p_range', 'p_up', 'p_down', 'p_max']

    # 合并
    df_merged = pd.merge(df_kline, df_signals_clean, on='datetime', how='left')

    logger.info(f"  合并后: {len(df_merged)} 条")
    logger.info(f"  有信号的K线: {df_merged['signal'].notna().sum()} 条")

    return df_merged


def add_signal_colors(df):
    """添加信号颜色"""
    def get_color(signal):
        if signal == '上涨':
            return 'green'
        elif signal == '下跌':
            return 'red'
        elif signal == '震荡':
            return 'gray'
        else:
            return 'white'

    df['signal_color'] = df['signal'].apply(get_color)
    return df


def plot_kline_with_signals(df):
    """绘制K线图+信号标记"""
    logger.info("生成K线图...")

    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('热卷HC888 - K线图+信号标记', 'MACD指标')
    )

    # 1. 绘制K线图
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='red',
            decreasing_line_color='green'
        ),
        row=1, col=1
    )

    # 2. 标记信号点
    for signal_type in ['上涨', '下跌', '震荡']:
        df_signal = df[df['signal'] == signal_type]

        if len(df_signal) > 0:
            color_map = {'上涨': 'green', '下跌': 'red', '震荡': 'gray'}

            fig.add_trace(
                go.Scatter(
                    x=df_signal['datetime'],
                    y=df_signal['close'],
                    mode='markers',
                    name=f'{signal_type}信号',
                    marker=dict(
                        symbol='circle' if signal_type != '震荡' else 'diamond',
                        size=8 if signal_type != '震荡' else 6,
                        color=color_map[signal_type],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    text=df_signal['p_max'].apply(lambda x: f'P={x:.2f}'),
                    hovertemplate='<b>%{x}</b><br>信号: %{fullData.name}<br>价格: %{y:.2f}<br>%{text}<extra></extra>'
                ),
                row=1, col=1
            )

    # 3. 添加信号背景色（可选）
    # 使用矩形标记不同信号期间
    for i in range(len(df)):
        if i == 0 or pd.isna(df.iloc[i]['signal']) or df.iloc[i]['signal'] != df.iloc[i-1]['signal']:
            continue

        signal = df.iloc[i]['signal']
        color_map = {'上涨': 'rgba(0,255,0,0.05)', '下跌': 'rgba(255,0,0,0.05)', '震荡': 'rgba(128,128,128,0.03)'}

        if signal in color_map:
            # 添加背景色带
            fig.add_vrect(
                x0=df.iloc[i]['datetime'],
                x1=df.iloc[i]['datetime'] + pd.Timedelta(hours=1),
                fillcolor=color_map[signal],
                layer='below',
                row=1, col=1
            )

    # 4. 绘制MACD
    colors = df['signal_color'].tolist()

    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df.get('macd', [0]*len(df)),
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=1)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df.get('macd_signal', [0]*len(df)),
            mode='lines',
            name='Signal',
            line=dict(color='orange', width=1)
        ),
        row=2, col=1
    )

    # MACD柱状图
    if 'macd_histogram' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df['datetime'],
                y=df['macd_histogram'],
                name='Histogram',
                marker_color=['red' if x < 0 else 'green' for x in df['macd_histogram']],
                opacity=0.5
            ),
            row=2, col=1
        )

    # 5. 设置布局
    fig.update_layout(
        title={
            'text': '热卷HC888 - 趋势信号可视化 (2025-10-01 至 2026-02-13)',
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

    fig.update_xaxes(
        title_text='时间',
        rangeslider_visible=False
    )

    fig.update_yaxes(
        title_text='价格',
        row=1, col=1
    )

    fig.update_yaxes(
        title_text='MACD',
        row=2, col=1
    )

    return fig


def generate_statistics(df):
    """生成信号统计"""
    logger.info("生成信号统计...")

    stats = {
        '总K线数': len(df),
        '有信号K线数': df['signal'].notna().sum(),
        '上涨信号数': (df['signal'] == '上涨').sum(),
        '下跌信号数': (df['signal'] == '下跌').sum(),
        '震荡信号数': (df['signal'] == '震荡').sum(),
        '最高价格': df['close'].max(),
        '最低价格': df['close'].min(),
        '平均价格': df['close'].mean(),
        '价格波动率': df['close'].std() / df['close'].mean() * 100
    }

    logger.info("\n" + "="*60)
    logger.info("信号统计汇总")
    logger.info("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")

    return stats


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("热卷K线图+信号标记可视化")
    logger.info("="*60)

    # 1. 加载数据
    df_kline, df_signals = load_data()

    # 2. 合并数据
    df_merged = merge_signals(df_kline, df_signals)

    # 3. 添加颜色
    df_merged = add_signal_colors(df_merged)

    # 4. 生成统计
    stats = generate_statistics(df_merged)

    # 5. 绘制图表
    fig = plot_kline_with_signals(df_merged)

    # 6. 保存图表
    logger.info(f"\n保存图表: {output_file}")
    fig.write_html(output_file)
    logger.info("✓ 图表已保存")

    # 7. 显示统计
    logger.info("\n" + "="*60)
    logger.info("完成！")
    logger.info("="*60)
    logger.info(f"K线数据: {stats['总K线数']} 条")
    logger.info(f"信号分布:")
    logger.info(f"  上涨: {stats['上涨信号数']} 条")
    logger.info(f"  下跌: {stats['下跌信号数']} 条")
    logger.info(f"  震荡: {stats['震荡信号数']} 条")
    logger.info(f"\n价格统计:")
    logger.info(f"  最高: {stats['最高价格']:.2f}")
    logger.info(f"  最低: {stats['最低价格']:.2f}")
    logger.info(f"  平均: {stats['平均价格']:.2f}")
    logger.info(f"  波动率: {stats['价格波动率']:.2f}%")
    logger.info(f"\n输出文件: {output_file}")
    logger.info("\n请在浏览器中打开HTML文件查看交互式图表")

    return fig, stats


if __name__ == '__main__':
    fig, stats = main()
