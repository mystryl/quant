/**
 * ZigZag 指标实现
 *
 * 算法原理：
 * - 识别显著的高点和低点（波峰波谷）
 * - 过滤微小的价格波动（基于 deviation 参数）
 * - 连接重要的转折点形成折线
 *
 * 参数：
 * - depth: 查看历史K线数量（默认12）
 * - deviation: 最小价格变化百分比（默认5）
 * - backstep: 回溯确认条数（默认3）
 */

(function() {
    if (typeof klinecharts === 'undefined') {
        console.error('KLineChart library not loaded');
        return;
    }

    // ZigZag 计算函数
    function calcZigZag(dataList, calcParams) {
        const depth = calcParams[0] || 12;
        const deviation = calcParams[1] || 5;
        const backstep = calcParams[2] || 3;

        const results = [];
        const pivots = [];

        const STATE = { START: 0, RISING: 1, FALLING: 2 };
        let currentState = STATE.START;
        let lastPivotIndex = -1;
        let lastPivotPrice = 0;
        let lastPivotType = null;

        // 检查局部高点
        function isLocalHigh(dataList, index, depth) {
            if (index < depth || index >= dataList.length - depth) return false;
            const currentHigh = dataList[index].high;
            for (let i = 1; i <= depth; i++) {
                if (dataList[index - i].high >= currentHigh ||
                    dataList[index + i].high >= currentHigh) {
                    return false;
                }
            }
            return true;
        }

        // 检查局部低点
        function isLocalLow(dataList, index, depth) {
            if (index < depth || index >= dataList.length - depth) return false;
            const currentLow = dataList[index].low;
            for (let i = 1; i <= depth; i++) {
                if (dataList[index - i].low <= currentLow ||
                    dataList[index + i].low <= currentLow) {
                    return false;
                }
            }
            return true;
        }

        // 计算价格变化百分比
        function getPriceChangePercent(price1, price2) {
            return Math.abs((price2 - price1) / price1 * 100);
        }

        // 主循环
        for (let i = depth; i < dataList.length - depth; i++) {
            const isHigh = isLocalHigh(dataList, i, depth);
            const isLow = isLocalLow(dataList, i, depth);

            if (isHigh || isLow) {
                const price = isHigh ? dataList[i].high : dataList[i].low;
                const type = isHigh ? 'high' : 'low';

                if (lastPivotIndex === -1) {
                    lastPivotIndex = i;
                    lastPivotPrice = price;
                    lastPivotType = type;
                    pivots.push({ index: i, price: price, type: type });
                    results.push({ zigzag: price });
                } else {
                    const priceChange = getPriceChangePercent(lastPivotPrice, price);

                    if (priceChange >= deviation && lastPivotType !== type) {
                        lastPivotIndex = i;
                        lastPivotPrice = price;
                        lastPivotType = type;
                        pivots.push({ index: i, price: price, type: type });
                        results.push({ zigzag: price });
                    } else {
                        results.push({});
                    }
                }
            } else {
                results.push({});
            }
        }

        // 输出 ZigZag 统计信息
        const validPoints = results.filter(r => r.zigzag !== undefined);
        console.log('📊 ZigZag 指标统计:');
        console.log(`   输入数据: ${dataList.length} 条`);
        console.log(`   输出数据: ${results.length} 条`);
        console.log(`   有效转折点: ${validPoints.length} 个`);
        console.log(`   转折点详情:`, pivots);

        return results;
    }

    // 注册指标到 KLineChart
    klinecharts.registerIndicator({
        name: 'ZIGZAG',
        shortName: 'ZigZag',
        series: 'normal',
        calcParams: [12, 5, 3],
        figures: [
            {
                key: 'zigzag',
                title: 'ZigZag: ',
                type: 'line'
            }
        ],
        calc: (dataList, indicator) => {
            return calcZigZag(dataList, indicator.calcParams);
        },
        shouldFormatBigNumber: false,
        precision: 2,
        styles: {
            lines: [
                {
                    style: 'line',
                    smooth: false,
                    size: 2,
                    color: '#FF9800'
                }
            ]
        }
    });

    console.log('✅ ZigZag indicator registered successfully');
})();
