<!-- src/esper/karn/overwatch/web/src/components/EnvironmentGrid.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { EnvState } from '../types/sanctum'

const props = defineProps<{
  envs: Record<number, EnvState>
  focusedEnvId: number
}>()

const emit = defineEmits<{
  select: [envId: number]
}>()

const sortedEnvs = computed(() => {
  return Object.values(props.envs).sort((a, b) => a.env_id - b.env_id)
})

function formatAccuracy(accuracy: number): string {
  return `${(accuracy * 100).toFixed(1)}%`
}

function getStatusClass(status: string): string {
  switch (status) {
    case 'healthy':
      return 'status-healthy'
    case 'stalled':
      return 'status-stalled'
    case 'degraded':
      return 'status-degraded'
    default:
      return 'status-neutral'
  }
}

function handleCardClick(envId: number) {
  emit('select', envId)
}
</script>

<template>
  <div class="environment-grid">
    <div
      v-for="env in sortedEnvs"
      :key="env.env_id"
      class="env-card"
      :class="{ focused: env.env_id === focusedEnvId }"
      :data-testid="`env-card-${env.env_id}`"
      @click="handleCardClick(env.env_id)"
    >
      <div class="env-badge" :data-testid="`env-badge-${env.env_id}`">
        {{ env.env_id }}
      </div>

      <div class="env-content">
        <div class="env-accuracy" :data-testid="`env-accuracy-${env.env_id}`">
          {{ formatAccuracy(env.host_accuracy) }}
        </div>

        <div class="env-epoch" :data-testid="`env-epoch-${env.env_id}`">
          Epoch {{ env.current_epoch }}
        </div>

        <div class="env-seed-counts" :data-testid="`env-seed-counts-${env.env_id}`">
          <span class="seed-active">{{ env.active_seed_count }}</span>
          <span class="seed-separator">/</span>
          <span class="seed-fossilized">{{ env.fossilized_count }}</span>
          <span class="seed-separator">/</span>
          <span class="seed-pruned">{{ env.pruned_count }}</span>
        </div>

        <div
          class="env-status"
          :class="getStatusClass(env.status)"
          :data-testid="`env-status-${env.env_id}`"
        >
          {{ env.status.toUpperCase() }}
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.environment-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-md);
  padding: var(--space-md);
}

@media (min-width: 768px) {
  .environment-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1024px) {
  .environment-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

.env-card {
  position: relative;
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: var(--space-md);
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.env-card:hover {
  border-color: var(--glow-cyan-dim);
}

.env-card.focused {
  border-color: var(--glow-cyan);
  box-shadow: 0 0 12px var(--glow-cyan-dim), 0 0 24px var(--glow-cyan-dim);
}

.env-badge {
  position: absolute;
  top: var(--space-sm);
  right: var(--space-sm);
  background: var(--bg-elevated);
  color: var(--text-secondary);
  font-size: 10px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 2px;
}

.env-content {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}

.env-accuracy {
  font-family: var(--font-display);
  font-size: 24px;
  font-weight: 600;
  color: var(--text-bright);
}

.env-epoch {
  font-size: 12px;
  color: var(--text-secondary);
}

.env-seed-counts {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  font-size: 11px;
}

.seed-active {
  color: var(--stage-training);
}

.seed-fossilized {
  color: var(--stage-fossilized);
}

.seed-pruned {
  color: var(--stage-pruned);
}

.seed-separator {
  color: var(--text-dim);
}

.env-status {
  font-size: 10px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 2px;
  width: fit-content;
}

.status-healthy {
  background: var(--status-win);
  color: var(--bg-void);
}

.status-stalled {
  background: var(--status-warn);
  color: var(--bg-void);
}

.status-degraded {
  background: var(--status-loss);
  color: var(--bg-void);
}

.status-neutral {
  background: var(--status-neutral);
  color: var(--bg-void);
}
</style>
