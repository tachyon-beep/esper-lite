<!-- src/esper/karn/overwatch/web/src/components/StatusBar.vue -->
<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  connectionState: 'connecting' | 'connected' | 'disconnected'
  staleness: number
  episode: number
  epoch: number
  batch: number
}>()

const stalenessText = computed(() => {
  if (props.staleness < 1000) return 'Just now'
  if (props.staleness < 10000) return `${(props.staleness / 1000).toFixed(1)}s ago`
  return 'STALE'
})

const stalenessClass = computed(() => {
  if (props.staleness < 2000) return 'fresh'
  if (props.staleness < 10000) return 'warning'
  return 'stale'
})
</script>

<template>
  <header class="status-bar">
    <div class="status-section">
      <span
        class="status-indicator"
        :class="connectionState"
        data-testid="connection-status"
      >
        <span class="status-icon" aria-hidden="true">
          <template v-if="connectionState === 'connecting'">&#8987;</template>
          <template v-else-if="connectionState === 'connected'">&#10003;</template>
          <template v-else>&#10007;</template>
        </span>
        {{ connectionState.toUpperCase() }}
      </span>
      <span
        class="staleness"
        :class="stalenessClass"
        data-testid="staleness"
      >
        {{ stalenessText }}
      </span>
    </div>

    <span class="title">OVERWATCH</span>

    <div class="metrics-section">
      <span class="metric">Ep {{ episode }}</span>
      <span class="separator">|</span>
      <span class="metric">Epoch {{ epoch }}</span>
      <span class="separator">|</span>
      <span class="metric">Batch {{ batch }}</span>
    </div>
  </header>
</template>

<style scoped>
.status-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-panel);
  border-bottom: 1px solid var(--border-subtle);
  font-size: 12px;
}

.status-section {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.status-indicator {
  display: inline-flex;
  align-items: center;
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 2px;
}

.status-icon {
  margin-right: 0.25rem;
}

.status-indicator.connecting {
  background: var(--status-warn);
  color: var(--bg-void);
}

.status-indicator.connected {
  background: var(--status-win);
  color: var(--bg-void);
}

.status-indicator.disconnected {
  background: var(--status-loss);
  color: var(--bg-void);
}

.staleness {
  color: var(--text-secondary);
}

.staleness.warning {
  color: var(--status-warn);
}

.staleness.stale {
  color: var(--status-loss);
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.title {
  font-family: var(--font-display);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 3px;
  color: var(--glow-cyan);
}

.metrics-section {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}

.metric {
  color: var(--text-primary);
}

.separator {
  color: var(--text-dim);
}
</style>
