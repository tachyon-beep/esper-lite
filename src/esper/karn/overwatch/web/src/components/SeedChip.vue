<!-- src/esper/karn/overwatch/web/src/components/SeedChip.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { SeedStage } from '../types/sanctum'

const props = defineProps<{
  slotId: string
  stage: SeedStage
  alpha?: number
  hasWarning?: boolean
}>()

const abbreviatedSlotId = computed(() => {
  const match = props.slotId.match(/slot_(\d+)/)
  return match ? `S${match[1]}` : props.slotId
})

const stageClass = computed(() => `stage-${props.stage.toLowerCase()}`)

const showAlpha = computed(() => {
  return props.alpha !== undefined && (props.stage === 'BLENDING' || props.stage === 'HOLDING')
})

const alphaPercent = computed(() => {
  if (props.alpha === undefined) return ''
  return `${Math.round(props.alpha * 100)}%`
})

const tooltip = computed(() => `${props.slotId}: ${props.stage}`)
</script>

<template>
  <span
    class="seed-chip"
    :class="stageClass"
    :title="tooltip"
    data-testid="seed-chip"
  >
    <span class="slot-id" data-testid="slot-id">{{ abbreviatedSlotId }}</span>
    <span v-if="showAlpha" class="alpha" data-testid="alpha">{{ alphaPercent }}</span>
    <span v-if="hasWarning" class="warning" data-testid="warning-indicator">!</span>
  </span>
</template>

<style scoped>
.seed-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 10px;
  font-weight: 600;
  font-family: var(--font-mono);
  color: var(--text-bright);
  white-space: nowrap;
  cursor: default;
}

.slot-id {
  letter-spacing: 0.5px;
}

.alpha {
  font-size: 9px;
  opacity: 0.9;
}

.warning {
  color: var(--status-warn);
  font-weight: 700;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Stage-specific colors */
.stage-dormant {
  background: var(--stage-dormant);
}

.stage-germinated {
  background: var(--stage-germinated);
}

.stage-training {
  background: var(--stage-training);
}

.stage-blending {
  background: var(--stage-blending);
}

.stage-holding {
  background: var(--stage-holding);
}

.stage-fossilized {
  background: var(--stage-fossilized);
}

.stage-pruned {
  background: var(--stage-pruned);
}

.stage-unknown,
.stage-embargoed,
.stage-resetting {
  background: var(--text-dim);
}
</style>
