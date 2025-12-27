<!-- src/esper/karn/overwatch/web/src/components/HealthGauges.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { SystemVitals, TamiyoState } from '../types/sanctum'

const props = defineProps<{
  vitals: SystemVitals
  tamiyo: TamiyoState
}>()

// SVG circle geometry
const RADIUS = 40
const STROKE_WIDTH = 8
const CIRCUMFERENCE = 2 * Math.PI * RADIUS
const SIZE = (RADIUS + STROKE_WIDTH) * 2

// Thresholds for health indicators
const TEMP_WARNING_THRESHOLD = 80 // Celsius
const ENTROPY_WARNING_THRESHOLD = 0.1 // Entropy collapse warning
const CLIP_FRACTION_WARNING_THRESHOLD = 0.2 // High clip fraction warning
const EXPLAINED_VARIANCE_GOOD_THRESHOLD = 0.5 // Good explained variance
const EXPLAINED_VARIANCE_WARNING_THRESHOLD = 0.3 // Warning threshold

type HealthStatus = 'good' | 'warning' | 'critical'

interface GaugeConfig {
  id: string
  label: string
  percent: number
  health: HealthStatus
}

// Compute GPU utilization percentage and health
const gpuUtilPercent = computed(() => props.vitals.gpu_utilization)
const gpuMemoryPercent = computed(() => {
  if (props.vitals.gpu_memory_total_gb <= 0) return 0
  return Math.round((props.vitals.gpu_memory_used_gb / props.vitals.gpu_memory_total_gb) * 100)
})

// GPU health considers temperature
const gpuHealth = computed<HealthStatus>(() => {
  if (props.vitals.gpu_temperature > TEMP_WARNING_THRESHOLD) return 'warning'
  if (props.vitals.gpu_utilization > 95) return 'warning'
  return 'good'
})

const gpuMemoryHealth = computed<HealthStatus>(() => {
  const percent = gpuMemoryPercent.value
  if (percent > 95) return 'critical'
  if (percent > 85) return 'warning'
  return 'good'
})

// Entropy: higher is better, warning if below 0.1
const entropyDisplay = computed(() => {
  // Normalize entropy for display (0-100 scale, capped for display purposes)
  // Typical entropy values are 0-2 for PPO, we show as percentage of "healthy" range
  return Math.min(100, Math.round(props.tamiyo.entropy * 100))
})

const entropyHealth = computed<HealthStatus>(() => {
  if (props.tamiyo.entropy < ENTROPY_WARNING_THRESHOLD) return 'critical'
  if (props.tamiyo.entropy < 0.3) return 'warning'
  return 'good'
})

// Clip fraction: lower is better, warning if above 0.2, critical if above 0.4
const CLIP_FRACTION_CRITICAL_THRESHOLD = 0.4
const clipFractionPercent = computed(() => Math.round(props.tamiyo.clip_fraction * 100))
const clipFractionHealth = computed<HealthStatus>(() => {
  if (props.tamiyo.clip_fraction > CLIP_FRACTION_CRITICAL_THRESHOLD) return 'critical'
  if (props.tamiyo.clip_fraction > CLIP_FRACTION_WARNING_THRESHOLD) return 'warning'
  return 'good'
})

// Explained variance: higher is better (range -inf to 1, but typically 0-1)
const explainedVariancePercent = computed(() => {
  // Clamp to 0-100 for display
  const ev = Math.max(0, Math.min(1, props.tamiyo.explained_variance))
  return Math.round(ev * 100)
})

const explainedVarianceHealth = computed<HealthStatus>(() => {
  if (props.tamiyo.explained_variance >= EXPLAINED_VARIANCE_GOOD_THRESHOLD) return 'good'
  if (props.tamiyo.explained_variance >= EXPLAINED_VARIANCE_WARNING_THRESHOLD) return 'warning'
  return 'critical'
})

// All gauge configurations
const gauges = computed<GaugeConfig[]>(() => [
  {
    id: 'gpu-util',
    label: 'GPU',
    percent: gpuUtilPercent.value,
    health: gpuHealth.value
  },
  {
    id: 'gpu-memory',
    label: 'VRAM',
    percent: gpuMemoryPercent.value,
    health: gpuMemoryHealth.value
  },
  {
    id: 'entropy',
    label: 'Entropy',
    percent: entropyDisplay.value,
    health: entropyHealth.value
  },
  {
    id: 'clip-fraction',
    label: 'Clip',
    percent: clipFractionPercent.value,
    health: clipFractionHealth.value
  },
  {
    id: 'explained-variance',
    label: 'ExpVar',
    percent: explainedVariancePercent.value,
    health: explainedVarianceHealth.value
  }
])

// Calculate stroke-dasharray for progress arc
function getStrokeDasharray(percent: number): string {
  const progress = (percent / 100) * CIRCUMFERENCE
  return `${progress} ${CIRCUMFERENCE}`
}

// Temperature warning display
const showTempWarning = computed(() => props.vitals.gpu_temperature > TEMP_WARNING_THRESHOLD)
</script>

<template>
  <div class="health-gauges">
    <div
      v-for="gauge in gauges"
      :key="gauge.id"
      :data-testid="`gauge-${gauge.id}`"
      class="gauge"
      :class="`health-${gauge.health}`"
    >
      <svg
        :width="SIZE"
        :height="SIZE"
        :viewBox="`0 0 ${SIZE} ${SIZE}`"
        class="gauge-svg"
      >
        <!-- Background circle -->
        <circle
          :cx="SIZE / 2"
          :cy="SIZE / 2"
          :r="RADIUS"
          :stroke-width="STROKE_WIDTH"
          class="gauge-bg"
          fill="none"
        />
        <!-- Progress arc -->
        <circle
          :cx="SIZE / 2"
          :cy="SIZE / 2"
          :r="RADIUS"
          :stroke-width="STROKE_WIDTH"
          :stroke-dasharray="getStrokeDasharray(gauge.percent)"
          class="gauge-progress"
          fill="none"
          :transform="`rotate(-90 ${SIZE / 2} ${SIZE / 2})`"
        />
      </svg>
      <div class="gauge-content">
        <span data-testid="gauge-value" class="gauge-value">{{ gauge.percent }}%</span>
        <span data-testid="gauge-label" class="gauge-label">{{ gauge.label }}</span>
      </div>
    </div>

    <!-- Temperature warning indicator -->
    <div
      v-if="showTempWarning"
      data-testid="temp-warning"
      class="temp-warning"
    >
      {{ vitals.gpu_temperature }}°C
    </div>
  </div>
</template>

<style scoped>
.health-gauges {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  padding: var(--space-sm);
}

.gauge {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.gauge-svg {
  display: block;
}

.gauge-bg {
  stroke: var(--bg-elevated);
}

.gauge-progress {
  stroke: var(--status-win);
  stroke-linecap: round;
  transition: stroke-dasharray 0.3s ease;
}

.gauge.health-good .gauge-progress {
  stroke: var(--status-win);
}

.gauge.health-warning .gauge-progress {
  stroke: var(--status-warn);
}

.gauge.health-critical .gauge-progress {
  stroke: var(--status-loss);
}

.gauge-content {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.gauge-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-bright);
}

.gauge-label {
  font-size: 10px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.temp-warning {
  display: flex;
  align-items: center;
  padding: var(--space-xs) var(--space-sm);
  background: var(--status-warn-glow);
  border: 1px solid var(--status-warn);
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  color: var(--status-warn);
}
</style>
