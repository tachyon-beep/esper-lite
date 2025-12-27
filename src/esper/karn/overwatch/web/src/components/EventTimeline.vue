<!-- src/esper/karn/overwatch/web/src/components/EventTimeline.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { EventLogEntry } from '../types/sanctum'

const props = withDefaults(
  defineProps<{
    events: EventLogEntry[]
    filter?: string[]
    maxHeight?: string
  }>(),
  {
    maxHeight: '400px'
  }
)

// Event type to icon/color mapping
const EVENT_CONFIG: Record<string, { icon: string; colorClass: string }> = {
  seed_germinated: { icon: '🌱', colorClass: 'event-type-seed_germinated' },
  seed_fossilized: { icon: '🪨', colorClass: 'event-type-seed_fossilized' },
  seed_pruned: { icon: '✂️', colorClass: 'event-type-seed_pruned' },
  anomaly_detected: { icon: '⚠️', colorClass: 'event-type-anomaly_detected' },
  episode_complete: { icon: '🏁', colorClass: 'event-type-episode_complete' },
  ppo_update: { icon: '🧠', colorClass: 'event-type-ppo_update' }
}

const DEFAULT_CONFIG = { icon: '📋', colorClass: 'event-type-default' }

const filteredEvents = computed(() => {
  if (!props.filter || props.filter.length === 0) {
    return props.events
  }
  return props.events.filter((event) =>
    props.filter!.includes(event.event_type)
  )
})

const eventCount = computed(() => filteredEvents.value.length)

const getEventConfig = (eventType: string) => {
  return EVENT_CONFIG[eventType] ?? DEFAULT_CONFIG
}

const containerStyle = computed(() => ({
  maxHeight: props.maxHeight
}))
</script>

<template>
  <div class="event-timeline" data-testid="event-timeline">
    <header class="timeline-header" data-testid="timeline-header">
      Events ({{ eventCount }})
    </header>

    <div
      class="timeline-container timeline-scroll"
      :style="containerStyle"
      data-testid="timeline-container"
    >
      <div class="timeline-line" data-testid="timeline-line"></div>

      <div
        v-if="eventCount === 0"
        class="empty-state"
        data-testid="empty-state"
      >
        No events yet
      </div>

      <ul v-else class="timeline-list" data-testid="timeline-list">
        <li
          v-for="(event, index) in filteredEvents"
          :key="`${event.timestamp}-${index}`"
          class="timeline-item"
          data-testid="timeline-item"
        >
          <span
            class="event-icon"
            :class="getEventConfig(event.event_type).colorClass"
            data-testid="event-icon"
          >
            {{ getEventConfig(event.event_type).icon }}
          </span>

          <div class="event-card">
            <div class="event-header">
              <span class="relative-time" data-testid="relative-time">
                {{ event.relative_time }}
              </span>
              <span
                class="event-type-badge"
                :class="getEventConfig(event.event_type).colorClass"
                data-testid="event-type-badge"
              >
                {{ event.event_type }}
              </span>
            </div>

            <p class="event-message" data-testid="event-message">
              {{ event.message }}
            </p>

            <div class="event-meta">
              <span
                v-if="event.env_id !== null"
                class="env-chip"
                data-testid="env-chip"
              >
                Env {{ event.env_id }}
              </span>
            </div>
          </div>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.event-timeline {
  display: flex;
  flex-direction: column;
  background: var(--bg-panel);
  border-radius: 8px;
  border: 1px solid var(--border-subtle);
}

.timeline-header {
  padding: var(--space-md);
  font-family: var(--font-display);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-subtle);
}

.timeline-container {
  position: relative;
  padding: var(--space-md);
  padding-left: var(--space-xl);
}

.timeline-scroll {
  overflow-y: auto;
}

.timeline-line {
  position: absolute;
  left: 24px;
  top: var(--space-md);
  bottom: var(--space-md);
  width: 2px;
  background: var(--border-subtle);
}

.empty-state {
  padding: var(--space-lg);
  text-align: center;
  color: var(--text-dim);
  font-size: 12px;
}

.timeline-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.timeline-item {
  display: flex;
  align-items: flex-start;
  gap: var(--space-md);
  margin-bottom: var(--space-md);
  position: relative;
}

.timeline-item:last-child {
  margin-bottom: 0;
}

.event-icon {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  border-radius: 50%;
  background: var(--bg-elevated);
  border: 2px solid var(--border-subtle);
  z-index: 1;
  margin-left: -14px;
}

/* Event type colors for icons */
.event-icon.event-type-seed_germinated {
  border-color: var(--status-win);
  background: rgba(0, 255, 157, 0.1);
}

.event-icon.event-type-seed_fossilized {
  border-color: var(--glow-cyan);
  background: rgba(0, 229, 255, 0.1);
}

.event-icon.event-type-seed_pruned {
  border-color: var(--status-loss);
  background: rgba(255, 92, 92, 0.1);
}

.event-icon.event-type-anomaly_detected {
  border-color: var(--status-warn);
  background: rgba(255, 179, 71, 0.1);
}

.event-icon.event-type-episode_complete {
  border-color: var(--text-dim);
  background: rgba(61, 74, 102, 0.2);
}

.event-icon.event-type-ppo_update {
  border-color: var(--stage-germinated);
  background: rgba(124, 77, 255, 0.1);
}

.event-icon.event-type-default {
  border-color: var(--text-secondary);
  background: var(--bg-elevated);
}

.event-card {
  flex: 1;
  min-width: 0;
  background: var(--bg-elevated);
  border-radius: 6px;
  padding: var(--space-sm) var(--space-md);
  border: 1px solid var(--border-subtle);
}

.event-header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-xs);
}

.relative-time {
  font-size: 10px;
  color: var(--text-dim);
  font-family: var(--font-mono);
}

.event-type-badge {
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 600;
  text-transform: lowercase;
  letter-spacing: 0.5px;
  background: var(--bg-panel);
  color: var(--text-secondary);
}

/* Event type colors for badges */
.event-type-badge.event-type-seed_germinated {
  background: rgba(0, 255, 157, 0.15);
  color: var(--status-win);
}

.event-type-badge.event-type-seed_fossilized {
  background: var(--glow-cyan-dim);
  color: var(--glow-cyan);
}

.event-type-badge.event-type-seed_pruned {
  background: rgba(255, 92, 92, 0.15);
  color: var(--status-loss);
}

.event-type-badge.event-type-anomaly_detected {
  background: rgba(255, 179, 71, 0.15);
  color: var(--status-warn);
}

.event-type-badge.event-type-episode_complete {
  background: rgba(61, 74, 102, 0.3);
  color: var(--text-secondary);
}

.event-type-badge.event-type-ppo_update {
  background: rgba(124, 77, 255, 0.15);
  color: var(--stage-germinated);
}

.event-message {
  font-size: 11px;
  color: var(--text-primary);
  line-height: 1.4;
  margin: 0;
  word-wrap: break-word;
}

.event-meta {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-top: var(--space-xs);
}

.env-chip {
  font-size: 9px;
  padding: 1px 6px;
  border-radius: 2px;
  background: var(--glow-cyan-dim);
  color: var(--glow-cyan);
  font-weight: 600;
}
</style>
