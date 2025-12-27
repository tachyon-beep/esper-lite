<!-- src/esper/karn/overwatch/web/src/components/AnomalySidebar.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { EventLogEntry } from '../types/sanctum'

const props = withDefaults(
  defineProps<{
    events: EventLogEntry[]
    maxItems?: number
  }>(),
  {
    maxItems: 20
  }
)

// Anomaly event type patterns
const ANOMALY_PATTERNS = ['anomaly', 'nan', 'explosion', 'collapse', 'warning']

// Critical event types (red indicators)
const CRITICAL_PATTERNS = ['nan', 'explosion']

const filteredAnomalies = computed(() => {
  return props.events
    .filter((event) =>
      ANOMALY_PATTERNS.some((pattern) =>
        event.event_type.toLowerCase().includes(pattern)
      )
    )
    .slice(0, props.maxItems)
})

const anomalyCount = computed(() => filteredAnomalies.value.length)

const isCritical = (eventType: string): boolean => {
  return CRITICAL_PATTERNS.some((pattern) =>
    eventType.toLowerCase().includes(pattern)
  )
}
</script>

<template>
  <aside class="anomaly-sidebar">
    <header class="sidebar-header" data-testid="sidebar-header">
      Anomalies ({{ anomalyCount }})
    </header>

    <div class="anomaly-list">
      <div
        v-if="anomalyCount === 0"
        class="empty-state"
        data-testid="empty-state"
      >
        No anomalies detected
      </div>

      <div
        v-for="(event, index) in filteredAnomalies"
        :key="`${event.timestamp}-${index}`"
        class="anomaly-item"
        :class="{ critical: isCritical(event.event_type) }"
        data-testid="anomaly-item"
      >
        <span
          class="severity-icon"
          :class="{ critical: isCritical(event.event_type) }"
          data-testid="severity-icon"
        >
          {{ isCritical(event.event_type) ? '!!' : '!' }}
        </span>

        <div class="anomaly-content">
          <span class="anomaly-message">{{ event.message }}</span>
          <div class="anomaly-meta">
            <span class="relative-time" data-testid="relative-time">
              {{ event.relative_time }}
            </span>
            <span
              v-if="event.env_id !== null"
              class="env-badge"
              data-testid="env-badge"
            >
              Env {{ event.env_id }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.anomaly-sidebar {
  display: flex;
  flex-direction: column;
  background: var(--bg-panel);
  border-right: 1px solid var(--border-subtle);
  width: 280px;
  max-height: 100%;
}

.sidebar-header {
  padding: var(--space-md);
  font-family: var(--font-display);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-subtle);
}

.anomaly-list {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-sm);
}

.empty-state {
  padding: var(--space-lg);
  text-align: center;
  color: var(--text-dim);
  font-size: 12px;
}

.anomaly-item {
  display: flex;
  align-items: flex-start;
  gap: var(--space-sm);
  padding: var(--space-sm);
  border-radius: 4px;
  margin-bottom: var(--space-xs);
  background: var(--bg-elevated);
  border-left: 3px solid var(--status-warn);
}

.anomaly-item.critical {
  border-left-color: var(--status-loss);
}

.severity-icon {
  flex-shrink: 0;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 700;
  border-radius: 50%;
  background: var(--status-warn);
  color: var(--bg-void);
}

.severity-icon.critical {
  background: var(--status-loss);
}

.anomaly-content {
  flex: 1;
  min-width: 0;
}

.anomaly-message {
  display: block;
  font-size: 11px;
  color: var(--text-primary);
  line-height: 1.4;
  word-wrap: break-word;
}

.anomaly-meta {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-top: var(--space-xs);
}

.relative-time {
  font-size: 10px;
  color: var(--text-dim);
}

.env-badge {
  font-size: 9px;
  padding: 1px 6px;
  border-radius: 2px;
  background: var(--glow-cyan-dim);
  color: var(--glow-cyan);
  font-weight: 600;
}
</style>
