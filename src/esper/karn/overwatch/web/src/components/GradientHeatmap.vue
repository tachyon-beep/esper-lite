<!-- src/esper/karn/overwatch/web/src/components/GradientHeatmap.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { TamiyoState } from '../types/sanctum'

// Named threshold constants
const HEALTH_GOOD_THRESHOLD = 0.8 // Health > 0.8 is normal
const HEALTH_WARN_THRESHOLD = 0.3 // Health > 0.3 is weakening, <= 0.3 is critical

type HealthStatus = 'good' | 'warn' | 'loss'

interface LayerHealth {
  name: string
  health: number
  status: HealthStatus
}

const props = withDefaults(
  defineProps<{
    tamiyo: TamiyoState
    compact?: boolean
  }>(),
  {
    compact: false
  }
)

// Determine health status from numeric value
function getHealthStatus(health: number): HealthStatus {
  if (health > HEALTH_GOOD_THRESHOLD) return 'good'
  if (health > HEALTH_WARN_THRESHOLD) return 'warn'
  return 'loss'
}

// Format health value to 2 decimal places
function formatHealth(health: number): string {
  return health.toFixed(2)
}

// Check if we have gradient data to display
const hasGradientData = computed(() => {
  const data = props.tamiyo.layer_gradient_health
  return data !== null && Object.keys(data).length > 0
})

// Convert layer_gradient_health to sorted array of LayerHealth objects
const layers = computed<LayerHealth[]>(() => {
  const data = props.tamiyo.layer_gradient_health
  if (!data) return []

  return Object.entries(data)
    .map(([name, health]) => ({
      name,
      health,
      status: getHealthStatus(health)
    }))
    .sort((a, b) => a.name.localeCompare(b.name))
})

// Summary counts
const deadLayers = computed(() => props.tamiyo.dead_layers)
const explodingLayers = computed(() => props.tamiyo.exploding_layers)
const hasIssues = computed(() => deadLayers.value > 0 || explodingLayers.value > 0)
</script>

<template>
  <div
    data-testid="gradient-heatmap"
    class="gradient-heatmap"
    :class="{ compact }"
  >
    <!-- Empty state -->
    <div
      v-if="!hasGradientData"
      data-testid="gradient-heatmap-empty"
      class="empty-state"
    >
      No gradient data
    </div>

    <!-- Grid of layers -->
    <template v-else>
      <div data-testid="gradient-heatmap-grid" class="heatmap-grid">
        <div
          v-for="layer in layers"
          :key="layer.name"
          :data-testid="`layer-row-${layer.name}`"
          class="layer-row"
        >
          <span data-testid="layer-name" class="layer-name">{{ layer.name }}</span>
          <div
            data-testid="health-cell"
            class="health-cell"
            :class="`status-${layer.status}`"
          >
            <span data-testid="health-value" class="health-value">{{ formatHealth(layer.health) }}</span>
          </div>
        </div>
      </div>

      <!-- Summary row -->
      <div
        data-testid="gradient-summary"
        class="gradient-summary"
        :class="{ 'has-issues': hasIssues }"
      >
        Dead: {{ deadLayers }} | Exploding: {{ explodingLayers }}
      </div>
    </template>
  </div>
</template>

<style scoped>
.gradient-heatmap {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
  padding: var(--space-sm);
  background: var(--bg-panel);
  border-radius: 4px;
  border: 1px solid var(--border-subtle);
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 60px;
  color: var(--text-secondary);
  font-size: 12px;
  font-style: italic;
}

.heatmap-grid {
  display: grid;
  gap: var(--space-xs);
}

.layer-row {
  display: grid;
  grid-template-columns: minmax(80px, 1fr) 60px;
  gap: var(--space-sm);
  align-items: center;
}

.compact .layer-row {
  grid-template-columns: minmax(60px, 1fr) 48px;
  gap: var(--space-xs);
}

.layer-name {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.compact .layer-name {
  font-size: 10px;
}

.health-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-xs) var(--space-sm);
  border-radius: 3px;
  transition: background-color 0.2s ease;
}

.compact .health-cell {
  padding: 2px var(--space-xs);
}

.health-value {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--text-bright);
}

.compact .health-value {
  font-size: 10px;
}

/* Status colors */
.status-good {
  background: var(--status-win-glow);
  border: 1px solid var(--status-win);
}

.status-warn {
  background: var(--status-warn-glow);
  border: 1px solid var(--status-warn);
}

.status-loss {
  background: var(--status-loss-glow);
  border: 1px solid var(--status-loss);
}

/* Summary row */
.gradient-summary {
  display: flex;
  justify-content: center;
  padding: var(--space-xs) var(--space-sm);
  border-top: 1px solid var(--border-subtle);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
}

.compact .gradient-summary {
  font-size: 10px;
  padding: 2px var(--space-xs);
}

.gradient-summary.has-issues {
  color: var(--status-warn);
  background: var(--status-warn-glow);
  border-radius: 3px;
}
</style>
