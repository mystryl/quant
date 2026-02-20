<template>
  <div class="kline-chart-container" :style="{ height }">
    <div ref="chartRef" class="kline-chart" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { init, dispose, type Chart } from 'klinecharts'
import type { KLineData } from '~/types/kline'

interface Props {
  data: KLineData[]
  height?: string
}

const props = withDefaults(defineProps<Props>(), {
  height: '600px'
})

const chartRef = ref<HTMLDivElement>()
let chart: Chart | null = null

// 初始化图表
onMounted(() => {
  console.log('KLineChart: onMounted, chartRef.value:', chartRef.value)
  if (!chartRef.value) {
    console.error('KLineChart: chartRef.value 为空')
    return
  }

  chart = init(chartRef.value, {
    styles: {
      candle: {
        type: 'candle_solid',
        bar: {
          upColor: '#26A69A',
          downColor: '#EF5350',
          noChangeColor: '#888888'
        },
        tooltip: {
          showRule: 'always',
          showType: 'standard',
          labels: ['时间: ', '开: ', '收: ', '高: ', '低: ', '成交量: '],
          text: {
            size: 12,
            color: '#D9D9D9'
          }
        },
        priceMark: {
          show: true,
          high: {
            show: true,
            color: '#26A69A',
            textSize: 10
          },
          low: {
            show: true,
            color: '#EF5350',
            textSize: 10
          },
          last: {
            show: true,
            upColor: '#26A69A',
            downColor: '#EF5350',
            noChangeColor: '#888888',
            text: {
              show: true,
              size: 12
            }
          }
        }
      },
      indicator: {
        tooltip: {
          showRule: 'crosshair',
          showType: 'standard',
          text: {
            size: 12,
            color: '#D9D9D9'
          }
        }
      },
      xAxis: {
        show: true,
        axisLine: {
          show: true,
          color: '#888888'
        },
        tickLine: {
          show: true,
          length: 5,
          color: '#888888'
        },
        tickText: {
          show: true,
          color: '#D9D9D9',
          size: 12
        }
      },
      yAxis: {
        show: true,
        position: 'right',
        axisLine: {
          show: true,
          color: '#888888'
        },
        tickLine: {
          show: true,
          length: 5,
          color: '#888888'
        },
        tickText: {
          show: true,
          color: '#D9D9D9',
          size: 12
        }
      },
      grid: {
        show: true,
        horizontal: {
          show: true,
          size: 1,
          color: '#292929',
          style: 'dashed'
        },
        vertical: {
          show: true,
          size: 1,
          color: '#292929',
          style: 'dashed'
        }
      },
      crosshair: {
        show: true,
        horizontal: {
          show: true,
          line: {
            show: true,
            style: 'dashed',
            dashValue: [4, 2],
            size: 1,
            color: '#888888'
          },
          text: {
            show: true,
            color: '#D9D9D9',
            size: 12,
            backgroundColor: '#505050'
          }
        },
        vertical: {
          show: true,
          line: {
            show: true,
            style: 'dashed',
            dashValue: [4, 2],
            size: 1,
            color: '#888888'
          },
          text: {
            show: true,
            color: '#D9D9D9',
            size: 12,
            backgroundColor: '#505050'
          }
        }
      }
    }
  })

  // 添加默认指标
  chart?.createIndicator('MA', true, { id: 'candle_pane' })
  chart?.createIndicator('VOL', false, { id: 'volume_pane' })

  console.log('KLineChart: 图表初始化完成, chart:', !!chart)
})

// 监听数据变化
watch(
  () => props.data,
  (newData) => {
    console.log('KLineChart: watch 触发, 数据长度:', newData?.length || 0, 'chart 是否存在:', !!chart)
    if (chart && newData && newData.length > 0) {
      console.log('KLineChart: 应用数据到图表, 第1条:', newData[0])
      chart.applyNewData(newData)
    } else {
      if (!chart) {
        console.log('KLineChart: 图表未初始化')
      }
      if (!newData || newData.length === 0) {
        console.log('KLineChart: 数据为空')
      }
    }
  },
  { deep: true }
)

// 销毁图表
onUnmounted(() => {
  if (chart) {
    dispose(chartRef.value!)
  }
})

// 暴露图表实例
defineExpose({
  chart
})
</script>

<style scoped>
.kline-chart-container {
  width: 100%;
  height: 100%;
  background-color: #0f0f0f;
}

.kline-chart {
  width: 100%;
  padding: 20px;
}
</style>
