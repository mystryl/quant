/**
 * K线数据类型定义
 */

export interface KLineData {
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Contract {
  symbol: string
  name: string
  exchange: string
}

export interface IndicatorConfig {
  name: string
  params?: Record<string, number>
  visible: boolean
}

export type PeriodType = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d'

export type DataSourceType = 'parquet' | 'smart-provider'

export interface ChartState {
  contract: Contract | null
  startDate: string
  endDate: string
  period: PeriodType
  dataSource: DataSourceType
  indicators: IndicatorConfig[]
}
