<template>
  <div class="indicator-panel">
    <div class="panel-header">
      <h3>技术指标</h3>
      <button class="refresh-btn" @click="handleRefresh">
        重新生成数据
      </button>
    </div>

    <div class="indicators-grid">
      <div
        v-for="indicator in indicators"
        :key="indicator.id"
        class="indicator-item"
      >
        <label class="indicator-checkbox">
          <input
            type="checkbox"
            v-model="indicator.visible"
            @change="handleToggle(indicator)"
          />
          <span class="indicator-name">{{ indicator.name }}</span>
        </label>

        <div v-if="indicator.visible" class="indicator-params">
          <span v-for="(value, key) in indicator.params" :key="key" class="param-item">
            {{ key }}: {{ value }}
          </span>
        </div>
      </div>
    </div>

    <div class="data-source-section">
      <label class="section-label">数据源</label>
      <div class="data-source-toggles">
        <button
          v-for="source in dataSources"
          :key="source.value"
          :class="['source-btn', { active: dataSource === source.value }]"
          @click="handleDataSourceChange(source.value)"
        >
          {{ source.label }}
        </button>
      </div>
    </div>

    <div class="period-section">
      <label class="section-label">周期</label>
      <select v-model="period" class="period-select" @change="handlePeriodChange">
        <option value="1m">1分钟</option>
        <option value="5m">5分钟</option>
        <option value="15m">15分钟</option>
        <option value="30m">30分钟</option>
        <option value="1h">1小时</option>
        <option value="4h">4小时</option>
        <option value="1d">日线</option>
      </select>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { IndicatorConfig, PeriodType, DataSourceType } from '~/types/kline'

interface DataSourceOption {
  label: string
  value: DataSourceType
}

const emit = defineEmits<{
  'refresh': []
  'toggle-indicator': [indicator: IndicatorConfig]
  'change-data-source': [source: DataSourceType]
  'change-period': [period: PeriodType]
}>()

const indicators = ref<IndicatorConfig[]>([
  { id: 'ma5', name: 'MA5', params: { period: 5 }, visible: true },
  { id: 'ma10', name: 'MA10', params: { period: 10 }, visible: true },
  { id: 'ma20', name: 'MA20', params: { period: 20 }, visible: true },
  { id: 'ma30', name: 'MA30', params: { period: 30 }, visible: false },
  { id: 'macd', name: 'MACD', params: {}, visible: false }
])

const dataSources: DataSourceOption[] = [
  { label: 'Parquet 文件', value: 'parquet' },
  { label: 'SmartDataProvider', value: 'smart-provider' }
]

const dataSource = ref<DataSourceType>('parquet')
const period = ref<PeriodType>('1m')

const handleRefresh = () => {
  emit('refresh')
}

const handleToggle = (indicator: IndicatorConfig) => {
  emit('toggle-indicator', indicator)
}

const handleDataSourceChange = (source: DataSourceType) => {
  dataSource.value = source
  emit('change-data-source', source)
}

const handlePeriodChange = () => {
  emit('change-period', period.value)
}
</script>

<style scoped>
.indicator-panel {
  padding: 20px;
  background-color: #1a1a1a;
  border-radius: 8px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.panel-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: #e0e0e0;
}

.refresh-btn {
  padding: 8px 16px;
  font-size: 14px;
  background-color: #26A69A;
  border: none;
  border-radius: 6px;
  color: #ffffff;
  cursor: pointer;
  transition: background-color 0.2s;
}

.refresh-btn:hover {
  background-color: #2bbd9f;
}

.indicators-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}

.indicator-item {
  padding: 12px;
  background-color: #2a2a2a;
  border-radius: 6px;
  border: 1px solid #3a3a3a;
}

.indicator-checkbox {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.indicator-checkbox input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.indicator-name {
  font-size: 14px;
  color: #e0e0e0;
}

.indicator-params {
  margin-top: 8px;
  padding-left: 24px;
}

.param-item {
  font-size: 12px;
  color: #888;
}

.data-source-section,
.period-section {
  margin-bottom: 20px;
}

.section-label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: #b0b0b0;
  margin-bottom: 8px;
}

.data-source-toggles {
  display: flex;
  gap: 8px;
}

.source-btn {
  flex: 1;
  padding: 10px 16px;
  font-size: 14px;
  background-color: #2a2a2a;
  border: 1px solid #3a3a3a;
  border-radius: 6px;
  color: #b0b0b0;
  cursor: pointer;
  transition: all 0.2s;
}

.source-btn:hover {
  background-color: #3a3a3a;
}

.source-btn.active {
  background-color: #26A69A;
  border-color: #26A69A;
  color: #ffffff;
}

.period-select {
  width: 100%;
  padding: 10px 12px;
  font-size: 14px;
  background-color: #2a2a2a;
  border: 1px solid #3a3a3a;
  border-radius: 6px;
  color: #e0e0e0;
  cursor: pointer;
}

.period-select:focus {
  outline: none;
  border-color: #26A69A;
}
</style>
