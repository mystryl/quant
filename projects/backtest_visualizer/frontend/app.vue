<template>
  <div class="app-container">
    <header class="app-header">
      <h1>📈 K线回测数据可视化系统</h1>
      <p class="subtitle">策略研究辅助工具 - Phase 1 静态页面（真实数据）</p>
    </header>

    <main class="app-main">
      <div class="control-panel">
        <ContractSelector
          :contracts="contracts"
          v-model="selectedContract"
          @change="handleContractChange"
          :loading="contractsLoading"
        />

        <div v-if="currentContract" class="contract-stats">
          <div class="stat-item">
            <span class="stat-label">交易所:</span>
            <span class="stat-value">{{ currentContract.exchange }}</span>
          </div>
          <div v-if="dataStats" class="stat-item">
            <span class="stat-label">数据条数:</span>
            <span class="stat-value">{{ dataStats.count }}</span>
          </div>
          <div v-if="dataStats" class="stat-item">
            <span class="stat-label">价格区间:</span>
            <span class="stat-value">
              {{ dataStats.price_range.min.toFixed(2) }} - {{ dataStats.price_range.max.toFixed(2) }}
            </span>
          </div>
        </div>

        <div class="date-range-section">
          <label class="section-label">时间段（自动加载数据）</label>
          <div v-if="dataStats" class="date-display">
            <div class="date-item">
              <span class="date-label">开始:</span>
              <span>{{ formatDate(dataStats.start_date) }}</span>
            </div>
            <div class="date-item">
              <span class="date-label">结束:</span>
              <span>{{ formatDate(dataStats.end_date) }}</span>
            </div>
          </div>
        </div>

        <IndicatorPanel
          @refresh="handleRefreshData"
          @toggle-indicator="handleToggleIndicator"
        />
      </div>

      <div class="chart-panel">
        <div v-if="!selectedContract" class="empty-state">
          <p>👈 请选择合约开始查看 K线数据</p>
        </div>
        <div v-else-if="loading" class="loading-state">
          <div class="spinner"></div>
          <p>加载数据中...</p>
        </div>
        <div v-else class="chart-wrapper">
          <div class="chart-header">
            <h2>{{ currentContract?.name }}</h2>
            <span class="chart-info">
              {{ dataStats?.count || 0 }} 条数据 | 真实历史数据
            </span>
          </div>
          <KLineChart
            ref="chartRef"
            :data="klineData"
            height="650px"
          />
        </div>
      </div>
    </main>

    <footer class="app-footer">
      <p>基于 KLineChart 9.8 | Nuxt 3 + TypeScript</p>
      <p class="footer-note">使用真实期货历史数据 - 数据来源: K线数据库</p>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import KLineChart from '~/components/KLineChart.vue'
import ContractSelector from '~/components/ContractSelector.vue'
import IndicatorPanel from '~/components/IndicatorPanel.vue'
import { loadContracts, loadKLineData, formatTimestamp } from '~/composables/useRealData'
import type { Contract, KLineData, IndicatorConfig } from '~/types/kline'

// 状态管理
const contracts = ref<Contract[]>([])
const contractsLoading = ref(true)
const selectedContract = ref('')
const klineData = ref<KLineData[]>([])
const dataStats = ref<any>(null)
const loading = ref(false)

const chartRef = ref<InstanceType<typeof KLineChart>>()

// 当前选择的合约
const currentContract = computed(() => {
  return contracts.value.find(c => c.symbol === selectedContract.value)
})

// 格式化日期
const formatDate = (timestamp: number) => {
  return formatTimestamp(timestamp)
}

// 初始化：加载合约列表
onMounted(async () => {
  console.log('🔄 页面已挂载，开始加载合约列表...')
  contracts.value = await loadContracts()
  contractsLoading.value = false
  console.log('✅ 已加载合约列表:', contracts.value.length, '个', contracts.value)
})

// 处理合约变化
const handleContractChange = async (contract: Contract) => {
  console.log('合约切换:', contract)
  selectedContract.value = contract.symbol

  // 清空现有数据
  klineData.value = []
  dataStats.value = null

  // 加载新数据
  await loadContractData(contract.symbol)
}

// 加载合约数据
const loadContractData = async (symbol: string) => {
  loading.value = true
  console.log(`开始加载 ${symbol} 数据...`)
  try {
    const result = await loadKLineData(symbol)

    if (result) {
      klineData.value = result.data
      dataStats.value = result.stats
      console.log(`✅ 加载 ${symbol} 数据成功:`, result.stats.count, '条')
      console.log('数据示例:', result.data[0])
      console.log('klineData.value 长度:', klineData.value.length)
    } else {
      console.error('❌ 加载数据失败: result 为 null')
      alert('加载数据失败')
    }
  } catch (error) {
    console.error('❌ 加载数据异常:', error)
    alert('加载数据失败: ' + error)
  } finally {
    loading.value = false
  }
}

// 刷新数据
const handleRefreshData = () => {
  if (selectedContract.value) {
    loadContractData(selectedContract.value)
  }
}

// 切换指标
const handleToggleIndicator = (indicator: IndicatorConfig) => {
  console.log('切换指标:', indicator)
}
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #0f0f0f;
}

.app-header {
  padding: 24px 32px;
  background-color: #1a1a1a;
  border-bottom: 1px solid #2a2a2a;
}

.app-header h1 {
  font-size: 28px;
  font-weight: 700;
  color: #e0e0e0;
  margin-bottom: 8px;
}

.subtitle {
  font-size: 14px;
  color: #888;
}

.app-main {
  flex: 1;
  display: grid;
  grid-template-columns: 350px 1fr;
  gap: 20px;
  padding: 20px 32px;
}

.control-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.contract-stats {
  padding: 16px;
  background-color: #1a1a1a;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
}

.stat-label {
  color: #888;
}

.stat-value {
  color: #e0e0e0;
  font-weight: 500;
}

.date-range-section {
  padding: 16px;
  background-color: #1a1a1a;
  border-radius: 8px;
}

.section-label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: #b0b0b0;
  margin-bottom: 12px;
}

.date-display {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.date-item {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: #888;
}

.date-label {
  color: #888;
}

.chart-panel {
  background-color: #1a1a1a;
  border-radius: 8px;
  overflow: hidden;
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 700px;
  color: #888;
  font-size: 16px;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 700px;
  color: #888;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #2a2a2a;
  border-top: 4px solid #26A69A;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.chart-wrapper {
  padding: 20px;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #2a2a2a;
}

.chart-header h2 {
  font-size: 20px;
  font-weight: 600;
  color: #e0e0e0;
}

.chart-info {
  font-size: 14px;
  color: #888;
}

.app-footer {
  padding: 16px 32px;
  background-color: #1a1a1a;
  border-top: 1px solid #2a2a2a;
  text-align: center;
  font-size: 14px;
  color: #888;
}

.footer-note {
  margin-top: 4px;
  font-size: 12px;
  color: #666;
}

@media (max-width: 1200px) {
  .app-main {
    grid-template-columns: 1fr;
  }

  .control-panel {
    max-width: 100%;
  }
}
</style>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background-color: #0f0f0f;
  color: #e0e0e0;
}

.kline-container {
  width: 100%;
  height: 100%;
  background-color: #0f0f0f;
}

#kline-chart {
  width: 100%;
  height: 600px;
}
</style>
