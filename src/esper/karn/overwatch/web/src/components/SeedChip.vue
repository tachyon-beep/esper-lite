<!-- src/esper/karn/overwatch/web/src/components/SeedChip.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { SeedStage } from '../types/sanctum'

// Curve glyph mapping for visual display (matches Python TUI)
// Always shown (UX policy: data points don't disappear) - bright when active, dim otherwise.
const CURVE_GLYPHS: Record<string, string> = {
  'LINEAR': '\u2571',        // ╱
  'COSINE': '\u223F',        // ∿
  'SIGMOID_GENTLE': '\u2312', // ⌒
  'SIGMOID': '\u222B',       // ∫
  'SIGMOID_SHARP': '\u2290', // ⊐
}

const props = defineProps<{
  slotId: string
  stage: SeedStage
  alpha?: number
  alphaCurve?: string
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

// Curve glyph display logic
// UX policy: Data points don't disappear. Always show a curve indicator:
// - Bright when causally active (BLENDING/HOLDING)
// - Dim when historical (FOSSILIZED) or not yet relevant (other stages)
const curveGlyph = computed(() => {
  const curve = props.alphaCurve ?? 'LINEAR'
  const glyph = CURVE_GLYPHS[curve] ?? '\u2212' // − (minus sign) as fallback
  return glyph
})

const isCurveActive = computed(() => {
  return props.stage === 'BLENDING' || props.stage === 'HOLDING'
})

const isCurveHistorical = computed(() => {
  return props.stage === 'FOSSILIZED'
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
    <span
      class="curve"
      :class="{ 'curve-active': isCurveActive, 'curve-historical': isCurveHistorical, 'curve-dim': !isCurveActive && !isCurveHistorical }"
      data-testid="curve-glyph"
    >{{ curveGlyph }}</span>
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

/* Curve glyph styling */
/* UX policy: Always visible, brightness indicates causal relevance */
.curve {
  font-size: 10px;
}

.curve-active {
  opacity: 1;
  color: var(--text-bright);
}

.curve-historical {
  opacity: 0.6;
}

.curve-dim {
  opacity: 0.4;
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
