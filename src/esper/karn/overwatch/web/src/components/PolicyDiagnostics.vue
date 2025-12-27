<!-- src/esper/karn/overwatch/web/src/components/PolicyDiagnostics.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { TamiyoState } from '../types/sanctum'

const props = defineProps<{
  tamiyo: TamiyoState
}>()

// Thresholds for health indicators
const ENTROPY_WARNING_THRESHOLD = 0.1
const HEAD_ENTROPY_WARNING_THRESHOLD = 0.05

type HealthStatus = 'good' | 'warning' | 'critical'

// --- Formatting Helpers ---
function formatLoss(value: number): string {
  return value.toFixed(3)
}

function formatRatio(value: number): string {
  return value.toFixed(2)
}

function formatAdvantage(value: number): string {
  return value.toFixed(2)
}

function formatEntropy(value: number): string {
  return value.toFixed(2)
}

function formatGradNorm(value: number): string {
  return value.toFixed(2)
}

// --- Health Status Computations ---
const deadLayersHealth = computed<HealthStatus>(() => {
  return props.tamiyo.dead_layers > 0 ? 'warning' : 'good'
})

const explodingLayersHealth = computed<HealthStatus>(() => {
  return props.tamiyo.exploding_layers > 0 ? 'critical' : 'good'
})

const nanGradCountHealth = computed<HealthStatus>(() => {
  return props.tamiyo.nan_grad_count > 0 ? 'critical' : 'good'
})

const entropyCollapsedHealth = computed<HealthStatus>(() => {
  return props.tamiyo.entropy_collapsed ? 'critical' : 'good'
})

const globalEntropyHealth = computed<HealthStatus>(() => {
  return props.tamiyo.entropy < ENTROPY_WARNING_THRESHOLD ? 'warning' : 'good'
})

// Head entropy health checks
function getHeadEntropyHealth(value: number): HealthStatus {
  return value < HEAD_ENTROPY_WARNING_THRESHOLD ? 'warning' : 'good'
}

// --- Head Data for Grid Display ---
interface HeadMetric {
  id: string
  label: string
  value: number
  health: HealthStatus
}

const headEntropies = computed<HeadMetric[]>(() => [
  { id: 'slot', label: 'Slot', value: props.tamiyo.head_slot_entropy, health: getHeadEntropyHealth(props.tamiyo.head_slot_entropy) },
  { id: 'blueprint', label: 'Blueprint', value: props.tamiyo.head_blueprint_entropy, health: getHeadEntropyHealth(props.tamiyo.head_blueprint_entropy) },
  { id: 'style', label: 'Style', value: props.tamiyo.head_style_entropy, health: getHeadEntropyHealth(props.tamiyo.head_style_entropy) },
  { id: 'tempo', label: 'Tempo', value: props.tamiyo.head_tempo_entropy, health: getHeadEntropyHealth(props.tamiyo.head_tempo_entropy) },
  { id: 'alpha-target', label: 'Alpha Target', value: props.tamiyo.head_alpha_target_entropy, health: getHeadEntropyHealth(props.tamiyo.head_alpha_target_entropy) },
  { id: 'alpha-speed', label: 'Alpha Speed', value: props.tamiyo.head_alpha_speed_entropy, health: getHeadEntropyHealth(props.tamiyo.head_alpha_speed_entropy) },
  { id: 'alpha-curve', label: 'Alpha Curve', value: props.tamiyo.head_alpha_curve_entropy, health: getHeadEntropyHealth(props.tamiyo.head_alpha_curve_entropy) },
  { id: 'op', label: 'Op', value: props.tamiyo.head_op_entropy, health: getHeadEntropyHealth(props.tamiyo.head_op_entropy) }
])

const headGradNorms = computed<HeadMetric[]>(() => [
  { id: 'slot', label: 'Slot', value: props.tamiyo.head_slot_grad_norm, health: 'good' },
  { id: 'blueprint', label: 'Blueprint', value: props.tamiyo.head_blueprint_grad_norm, health: 'good' },
  { id: 'style', label: 'Style', value: props.tamiyo.head_style_grad_norm, health: 'good' },
  { id: 'tempo', label: 'Tempo', value: props.tamiyo.head_tempo_grad_norm, health: 'good' },
  { id: 'alpha-target', label: 'Alpha Target', value: props.tamiyo.head_alpha_target_grad_norm, health: 'good' },
  { id: 'alpha-speed', label: 'Alpha Speed', value: props.tamiyo.head_alpha_speed_grad_norm, health: 'good' },
  { id: 'alpha-curve', label: 'Alpha Curve', value: props.tamiyo.head_alpha_curve_grad_norm, health: 'good' },
  { id: 'op', label: 'Op', value: props.tamiyo.head_op_grad_norm, health: 'good' }
])

const hasEarlyStop = computed(() => props.tamiyo.early_stop_epoch !== null)
</script>

<template>
  <div class="policy-diagnostics">
    <!-- Loss Display Section -->
    <section data-testid="loss-section" class="section loss-section">
      <h3 class="section-title">Losses</h3>
      <div class="stat-row">
        <div data-testid="policy-loss" class="stat-item">
          <span class="stat-label">Policy</span>
          <span class="stat-value">{{ formatLoss(tamiyo.policy_loss) }}</span>
        </div>
        <div data-testid="value-loss" class="stat-item">
          <span class="stat-label">Value</span>
          <span class="stat-value">{{ formatLoss(tamiyo.value_loss) }}</span>
        </div>
        <div data-testid="entropy-loss" class="stat-item">
          <span class="stat-label">Entropy</span>
          <span class="stat-value">{{ formatLoss(tamiyo.entropy_loss) }}</span>
        </div>
      </div>
    </section>

    <!-- Health Indicators Section -->
    <section data-testid="health-section" class="section health-section">
      <h3 class="section-title">Health</h3>
      <div class="health-indicators">
        <div
          data-testid="dead-layers"
          class="health-item"
          :class="`health-${deadLayersHealth}`"
        >
          <span class="health-label">Dead Layers</span>
          <span class="health-value">{{ tamiyo.dead_layers }}</span>
        </div>
        <div
          data-testid="exploding-layers"
          class="health-item"
          :class="`health-${explodingLayersHealth}`"
        >
          <span class="health-label">Exploding</span>
          <span class="health-value">{{ tamiyo.exploding_layers }}</span>
        </div>
        <div
          data-testid="nan-grad-count"
          class="health-item"
          :class="`health-${nanGradCountHealth}`"
        >
          <span class="health-label">NaN Grads</span>
          <span class="health-value">{{ tamiyo.nan_grad_count }}</span>
        </div>
        <div
          data-testid="entropy-collapsed"
          class="health-item"
          :class="[`health-${entropyCollapsedHealth}`, { prominent: tamiyo.entropy_collapsed }]"
        >
          <span class="health-label">Entropy Collapsed</span>
          <span class="health-value">{{ tamiyo.entropy_collapsed ? 'YES' : 'OK' }}</span>
        </div>
        <div
          data-testid="global-entropy"
          class="health-item"
          :class="`health-${globalEntropyHealth}`"
        >
          <span class="health-label">Entropy</span>
          <span class="health-value">{{ formatEntropy(tamiyo.entropy) }}</span>
        </div>
      </div>
    </section>

    <!-- Ratio Statistics Section -->
    <section data-testid="ratio-section" class="section">
      <h3 class="section-title">Ratio Stats</h3>
      <div data-testid="ratio-stats-group" class="stat-grid">
        <div data-testid="ratio-mean" class="stat-item">
          <span class="stat-label">Mean</span>
          <span class="stat-value">{{ formatRatio(tamiyo.ratio_mean) }}</span>
        </div>
        <div data-testid="ratio-min" class="stat-item">
          <span class="stat-label">Min</span>
          <span class="stat-value">{{ formatRatio(tamiyo.ratio_min) }}</span>
        </div>
        <div data-testid="ratio-max" class="stat-item">
          <span class="stat-label">Max</span>
          <span class="stat-value">{{ formatRatio(tamiyo.ratio_max) }}</span>
        </div>
        <div data-testid="ratio-std" class="stat-item">
          <span class="stat-label">Std</span>
          <span class="stat-value">{{ formatRatio(tamiyo.ratio_std) }}</span>
        </div>
      </div>
    </section>

    <!-- Advantage Statistics Section -->
    <section data-testid="advantage-section" class="section">
      <h3 class="section-title">Advantage Stats</h3>
      <div data-testid="advantage-stats-group" class="stat-grid">
        <div data-testid="advantage-mean" class="stat-item">
          <span class="stat-label">Mean</span>
          <span class="stat-value">{{ formatAdvantage(tamiyo.advantage_mean) }}</span>
        </div>
        <div data-testid="advantage-std" class="stat-item">
          <span class="stat-label">Std</span>
          <span class="stat-value">{{ formatAdvantage(tamiyo.advantage_std) }}</span>
        </div>
        <div data-testid="advantage-min" class="stat-item">
          <span class="stat-label">Min</span>
          <span class="stat-value">{{ formatAdvantage(tamiyo.advantage_min) }}</span>
        </div>
        <div data-testid="advantage-max" class="stat-item">
          <span class="stat-label">Max</span>
          <span class="stat-value">{{ formatAdvantage(tamiyo.advantage_max) }}</span>
        </div>
      </div>
    </section>

    <!-- Per-Head Entropy Grid -->
    <section data-testid="head-entropy-section" class="section">
      <h3 class="section-title">Head Entropies</h3>
      <div data-testid="head-entropy-grid" class="head-grid">
        <div
          v-for="head in headEntropies"
          :key="head.id"
          :data-testid="`head-entropy-${head.id}`"
          class="head-item"
          :class="`health-${head.health}`"
        >
          <span class="head-label">{{ head.label }}</span>
          <span class="head-value">{{ formatEntropy(head.value) }}</span>
        </div>
      </div>
    </section>

    <!-- Per-Head Gradient Norms Grid -->
    <section data-testid="head-grad-section" class="section">
      <h3 class="section-title">Head Grad Norms</h3>
      <div data-testid="head-grad-grid" class="head-grid">
        <div
          v-for="head in headGradNorms"
          :key="head.id"
          :data-testid="`head-grad-${head.id}`"
          class="head-item"
          :class="`health-${head.health}`"
        >
          <span class="head-label">{{ head.label }}</span>
          <span class="head-value">{{ formatGradNorm(head.value) }}</span>
        </div>
      </div>
    </section>

    <!-- Update Timing Section -->
    <section data-testid="timing-section" class="section timing-section">
      <h3 class="section-title">Timing</h3>
      <div class="timing-row">
        <div data-testid="update-time" class="timing-item">
          <span class="timing-label">Update Time</span>
          <span class="timing-value">{{ tamiyo.update_time_ms }} ms</span>
        </div>
        <div
          v-if="hasEarlyStop"
          data-testid="early-stop-epoch"
          class="timing-item"
        >
          <span class="timing-label">Early Stop</span>
          <span class="timing-value">Epoch {{ tamiyo.early_stop_epoch }}</span>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.policy-diagnostics {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  padding: var(--space-md);
  background: var(--bg-panel);
  border-radius: 4px;
}

.section {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
  margin-bottom: var(--space-xs);
}

/* Stat Row for Losses */
.stat-row {
  display: flex;
  gap: var(--space-md);
}

/* Stat Grid for Ratio/Advantage stats */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-sm);
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-xs) var(--space-sm);
  background: var(--bg-elevated);
  border-radius: 4px;
}

.stat-label {
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-bright);
  font-family: var(--font-mono);
}

/* Health Indicators */
.health-indicators {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
}

.health-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-xs) var(--space-sm);
  background: var(--bg-elevated);
  border-radius: 4px;
  border: 1px solid transparent;
}

.health-label {
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.health-value {
  font-size: 14px;
  font-weight: 600;
  font-family: var(--font-mono);
}

/* Health Status Classes */
.health-good {
  border-color: var(--status-win);
}

.health-good .health-value {
  color: var(--status-win);
}

.health-warning {
  border-color: var(--status-warn);
  background: var(--status-warn-glow);
}

.health-warning .health-value {
  color: var(--status-warn);
}

.health-critical {
  border-color: var(--status-loss);
  background: var(--status-loss-glow);
}

.health-critical .health-value {
  color: var(--status-loss);
}

/* Prominent styling for entropy collapsed */
.health-item.prominent {
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 var(--status-loss-glow);
  }
  50% {
    box-shadow: 0 0 12px 4px var(--status-loss-glow);
  }
}

/* Head Grid for Entropy/Grad Norms */
.head-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-xs);
}

.head-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: var(--space-xs);
  background: var(--bg-elevated);
  border-radius: 4px;
  border: 1px solid transparent;
}

.head-label {
  font-size: 9px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.3px;
  text-align: center;
}

.head-value {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-bright);
  font-family: var(--font-mono);
}

.head-item.health-good {
  border-color: var(--border-subtle);
}

.head-item.health-good .head-value {
  color: var(--text-bright);
}

.head-item.health-warning {
  border-color: var(--status-warn);
  background: var(--status-warn-glow);
}

.head-item.health-warning .head-value {
  color: var(--status-warn);
}

/* Timing Section */
.timing-row {
  display: flex;
  gap: var(--space-md);
}

.timing-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-xs) var(--space-sm);
  background: var(--bg-elevated);
  border-radius: 4px;
}

.timing-label {
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.timing-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-bright);
  font-family: var(--font-mono);
}
</style>
