/**
 * 模拟数据生成器
 */
import type { KLineData, Contract } from '~/types/kline'

// 模拟合约列表
export const mockContracts: Contract[] = [
  { symbol: 'CU9999.XSGE', name: '铜主力连续', exchange: 'XSGE' },
  { symbol: 'AL9999.XSGE', name: '铝主力连续', exchange: 'XSGE' },
  { symbol: 'AU9999.XSGE', name: '黄金主力连续', exchange: 'XSGE' },
  { symbol: 'AG9999.XSGE', name: '白银主力连续', exchange: 'XSGE' },
  { symbol: 'ZN9999.XSGE', name: '锌主力连续', exchange: 'XSGE' },
  { symbol: 'RB9999.XSGE', name: '螺纹钢主力连续', exchange: 'XSGE' },
  { symbol: 'A9999.XDCE', name: '豆一主力连续', exchange: 'XDCE' },
  { symbol: 'M9999.XDCE', name: '豆粕主力连续', exchange: 'XDCE' },
  { symbol: 'Y9999.XDCE', name: '豆油主力连续', exchange: 'XDCE' },
  { symbol: 'SR9999.XZCE', name: '白糖主力连续', exchange: 'XZCE' },
  { symbol: 'CF9999.XZCE', name: '棉花主力连续', exchange: 'XZCE' },
  { symbol: 'MA9999.XZCE', name: '甲醇主力连续', exchange: 'XZCE' }
]

/**
 * 生成模拟K线数据
 * @param basePrice 基础价格
 * @param count 数据条数
 * @param startTime 开始时间戳
 */
export function generateMockKLineData(
  basePrice: number = 68500,
  count: number = 500,
  startTime?: number
): KLineData[] {
  const data: KLineData[] = []
  let price = basePrice
  let timestamp = startTime || Date.now() - count * 60 * 1000

  for (let i = 0; i < count; i++) {
    // 模拟价格波动
    const volatility = price * 0.002 // 0.2% 波动
    const open = price
    const change = (Math.random() - 0.5) * 2 * volatility
    const close = price + change

    // 生成最高价和最低价
    const high = Math.max(open, close) + Math.random() * volatility * 0.5
    const low = Math.min(open, close) - Math.random() * volatility * 0.5

    // 生成成交量
    const volume = Math.floor(Math.random() * 10000 + 5000)

    data.push({
      timestamp: timestamp + i * 60 * 1000, // 每分钟
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume
    })

    price = close
  }

  return data
}

/**
 * 计算移动平均线
 */
export function calculateMA(data: KLineData[], period: number): number[] {
  const result: number[] = []

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(NaN)
      continue
    }

    let sum = 0
    for (let j = 0; j < period; j++) {
      sum += data[i - j].close
    }
    result.push(Number((sum / period).toFixed(2)))
  }

  return result
}

/**
 * 格式化时间戳为日期字符串
 */
export function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}
