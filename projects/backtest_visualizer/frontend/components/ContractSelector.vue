<template>
  <div class="contract-selector">
    <label class="selector-label">选择合约</label>
    <select :value="modelValue" @change="handleChange" class="contract-select" :disabled="loading">
      <option value="">-- 请选择合约 --</option>
      <option v-for="contract in contracts" :key="contract.symbol" :value="contract.symbol">
        {{ contract.name }} ({{ contract.symbol }})
      </option>
    </select>
    <span v-if="selectedContractData" class="contract-info">
      {{ selectedContractData.name }} - {{ selectedContractData.exchange }}
    </span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Contract } from '~/types/kline'

interface Props {
  contracts: Contract[]
  modelValue?: string
  loading?: boolean
}

const props = defineProps<Props>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
  'change': [contract: Contract]
}>()

const selectedContractData = computed(() => {
  return props.contracts.find(c => c.symbol === props.modelValue)
})

const handleChange = (event: Event) => {
  const target = event.target as HTMLSelectElement
  const value = target.value
  emit('update:modelValue', value)

  const contract = props.contracts.find(c => c.symbol === value)
  if (contract) {
    emit('change', contract)
  }
}
</script>

<style scoped>
.contract-selector {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background-color: #1a1a1a;
  border-radius: 8px;
}

.selector-label {
  font-size: 14px;
  font-weight: 500;
  color: #b0b0b0;
}

.contract-select {
  flex: 1;
  max-width: 400px;
  padding: 8px 12px;
  font-size: 14px;
  background-color: #2a2a2a;
  border: 1px solid #3a3a3a;
  border-radius: 6px;
  color: #e0e0e0;
  cursor: pointer;
  transition: border-color 0.2s;
}

.contract-select:hover:not(:disabled) {
  border-color: #4a4a4a;
}

.contract-select:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.contract-select:focus {
  outline: none;
  border-color: #26A69A;
}

.contract-info {
  font-size: 12px;
  color: #888;
}
</style>
