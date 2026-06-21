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
  // null percent + pending=true => no data sampled yet, render "—" not "0%".
  percent: number | null
  health: HealthStatus
  pending: boolean
}

// Presence flags distinguish "not measured yet" from "measured zero".
// GPU vitals arrive from system-stats polling; PPO health arrives on PPO updates.
const gpuPresent = computed(() => props.vitals.gpu_data_present)
const ppoPresent = computed(() => props.tamiyo.ppo_data_received)

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

// Explained variance: higher is better (range -inf to 1, but typically 0-1).
// EV-telemetry-robustness (no-bug-hiding): do NOT clamp the DISPLAYED value to [0,1]
// — a blown-out negative EV must read as its true (negative) value, not a masking "0%".
// The progress-arc geometry is clamped separately in getStrokeDasharray (a negative
// dash length is invalid SVG); only the rendered number is honest about the collapse.
const explainedVariancePercent = computed(() => {
  return Math.round(props.tamiyo.explained_variance * 100)
})

const explainedVarianceHealth = computed<HealthStatus>(() => {
  if (props.tamiyo.ev_low_return_variance) return 'warning'
  if (props.tamiyo.explained_variance >= EXPLAINED_VARIANCE_GOOD_THRESHOLD) return 'good'
  if (props.tamiyo.explained_variance >= EXPLAINED_VARIANCE_WARNING_THRESHOLD) return 'warning'
  return 'critical'
})

// All gauge configurations. When the relevant data group has not been measured
// yet (gpuPresent / ppoPresent false), the gauge is marked pending: percent is
// null and health forced 'good' (neutral) so nothing renders as a false alarm.
const gauges = computed<GaugeConfig[]>(() => [
  {
    id: 'gpu-util',
    label: 'GPU',
    percent: gpuPresent.value ? gpuUtilPercent.value : null,
    health: gpuPresent.value ? gpuHealth.value : 'good',
    pending: !gpuPresent.value
  },
  {
    id: 'gpu-memory',
    label: 'VRAM',
    percent: gpuPresent.value ? gpuMemoryPercent.value : null,
    health: gpuPresent.value ? gpuMemoryHealth.value : 'good',
    pending: !gpuPresent.value
  },
  {
    id: 'entropy',
    label: 'Entropy',
    percent: ppoPresent.value ? entropyDisplay.value : null,
    health: ppoPresent.value ? entropyHealth.value : 'good',
    pending: !ppoPresent.value
  },
  {
    id: 'clip-fraction',
    label: 'Clip',
    percent: ppoPresent.value ? clipFractionPercent.value : null,
    health: ppoPresent.value ? clipFractionHealth.value : 'good',
    pending: !ppoPresent.value
  },
  {
    id: 'explained-variance',
    label: 'ExpVar',
    percent: ppoPresent.value ? explainedVariancePercent.value : null,
    health: ppoPresent.value ? explainedVarianceHealth.value : 'good',
    pending: !ppoPresent.value
  }
])

// Calculate stroke-dasharray for progress arc. Pending gauges have no arc.
// The arc geometry is clamped to [0, CIRCUMFERENCE]: a negative dash length is invalid
// SVG, and an arc cannot exceed a full circle. This clamps the GEOMETRY only — the
// displayed numeric value (e.g. a negative EV) is rendered honestly elsewhere.
function getStrokeDasharray(percent: number | null): string {
  if (percent === null) return `0 ${CIRCUMFERENCE}`
  const fraction = Math.max(0, Math.min(1, percent / 100))
  const progress = fraction * CIRCUMFERENCE
  return `${progress} ${CIRCUMFERENCE}`
}

// Temperature warning display (only meaningful once GPU stats are sampled)
const showTempWarning = computed(
  () => gpuPresent.value && props.vitals.gpu_temperature > TEMP_WARNING_THRESHOLD
)
</script>

<template>
  <div class="health-gauges">
    <div
      v-for="gauge in gauges"
      :key="gauge.id"
      :data-testid="`gauge-${gauge.id}`"
      class="gauge"
      :class="[`health-${gauge.health}`, { 'gauge-pending': gauge.pending }]"
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
        <span data-testid="gauge-value" class="gauge-value">
          {{ gauge.pending ? '—' : `${gauge.percent}%` }}
        </span>
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

    <!-- EV-telemetry-robustness: low-return-variance badge. When the EV denominator is
         ill-conditioned, a deeply negative EV gauge is a denominator artifact, not a collapse;
         this badge tells the reader the EV reading is not trustworthy on this batch. -->
    <div
      v-if="tamiyo.ev_low_return_variance"
      data-testid="ev-low-variance-badge"
      class="ev-low-variance-badge"
      title="EV denominator ill-conditioned (low return variance): the explained-variance gauge is artifact-prone on this batch."
    >
      low return var
    </div>
  </div>
</template>

<style scoped>
.health-gauges {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(88px, 1fr));
  align-items: center;
  justify-items: center;
  gap: var(--space-md);
  padding: var(--space-sm);
}

.gauge {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 96px;
  min-width: 0;
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

/* Pending: no data sampled yet — neutral, no progress arc, dimmed value. */
.gauge.gauge-pending .gauge-progress {
  stroke: var(--bg-elevated);
}

.gauge.gauge-pending .gauge-value {
  color: var(--text-dim);
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
  justify-content: center;
  padding: var(--space-xs) var(--space-sm);
  background: var(--status-warn-glow);
  border: 1px solid var(--status-warn);
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  color: var(--status-warn);
}

.ev-low-variance-badge {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-xs) var(--space-sm);
  background: var(--bg-elevated);
  border: 1px solid var(--text-dim);
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
}

@media (max-width: 640px) {
  .health-gauges {
    grid-template-columns: repeat(3, minmax(88px, 1fr));
    gap: var(--space-sm);
  }
}
</style>
