/**
 * 真实K线数据加载器
 */
import type { KLineData, Contract } from '~/types/kline'

/**
 * 从JSON文件加载合约列表
 */
export async function loadContracts(): Promise<Contract[]> {
  try {
    console.log('🔄 fetch /data/contracts.json')
    const response = await fetch('/data/contracts.json')
    const data = await response.json()
    console.log('✅ 合约列表加载成功:', data.contracts.length, '个')
    return data.contracts
  } catch (error) {
    console.error('❌ 加载合约列表失败:', error)
    return []
  }
}

/**
 * 从JSON文件加载K线数据
 */
export async function loadKLineData(symbol: string): Promise<{
  data: KLineData[]
  stats: {
    count: number
    start_date: number
    end_date: number
    price_range: {
      min: number
      max: number
    }
  }
} | null> {
  try {
    const url = `/data/${symbol}.json`
    console.log(`🔄 fetch ${url}`)
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`加载失败: ${response.statusText}`)
    }
    const result = await response.json()
    console.log(`✅ ${symbol} 数据加载成功:`, result.data.length, '条')
    return result
  } catch (error) {
    console.error(`❌ 加载 ${symbol} 数据失败:`, error)
    return null
  }
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
