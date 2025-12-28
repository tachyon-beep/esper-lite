<!-- src/esper/karn/overwatch/web/src/components/SeedSwimlane.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { SeedState, SeedStage } from '../types/sanctum'

const props = defineProps<{
  seeds: Record<string, SeedState>
  slotIds: string[]
  currentEpoch: number
}>()

// All stage types for legend display
const allStages: SeedStage[] = [
  'DORMANT', 'GERMINATED', 'TRAINING', 'BLENDING',
  'HOLDING', 'FOSSILIZED', 'PRUNED'
]

// Check if we have any seeds to display
const hasSeeds = computed(() => props.slotIds.length > 0)

// Get seed state for a slot, or return a default dormant state
function getSeedForSlot(slotId: string): SeedState {
  if (props.seeds[slotId]) {
    return props.seeds[slotId]
  }
  // Return default dormant state for missing seeds
  return {
    slot_id: slotId,
    stage: 'DORMANT',
    blueprint_id: null,
    alpha: 0.0,
    accuracy_delta: 0.0,
    seed_params: 0,
    grad_ratio: 0.0,
    has_vanishing: false,
    has_exploding: false,
    epochs_in_stage: 0,
    improvement: 0.0,
    prune_reason: '',
    auto_pruned: false,
    epochs_total: 0,
    counterfactual: 0.0,
    blend_tempo_epochs: 0,
    alpha_curve: 'LINEAR'
  }
}

// Abbreviate slot ID for display (e.g., "slot_0" -> "S0")
function abbreviateSlotId(slotId: string): string {
  const match = slotId.match(/slot_(\d+)/)
  return match ? `S${match[1]}` : slotId
}

// Get the stage class for styling
function getStageClass(stage: string): string {
  return `stage-${stage.toLowerCase()}`
}

// Calculate bar width as percentage of current epoch
function getBarWidth(epochsInStage: number): string {
  if (props.currentEpoch <= 0) return '0%'
  const widthPercent = Math.min(100, (epochsInStage / props.currentEpoch) * 100)
  return `${widthPercent}%`
}

// Generate tooltip for a stage bar
function getBarTooltip(seed: SeedState): string {
  return `${seed.stage}: ${seed.epochs_in_stage} epochs`
}

// Format stage name for legend display
function formatStageName(stage: SeedStage): string {
  return stage.charAt(0) + stage.slice(1).toLowerCase()
}
</script>

<template>
  <div class="seed-swimlane" data-testid="seed-swimlane">
    <!-- Empty state -->
    <div v-if="!hasSeeds" class="empty-state" data-testid="empty-state">
      No active seeds
    </div>

    <!-- Swimlane chart -->
    <template v-else>
      <div class="swimlane-container">
        <!-- Slot rows -->
        <div
          v-for="slotId in slotIds"
          :key="slotId"
          class="swimlane-row"
          data-testid="swimlane-row"
        >
          <span class="slot-label" data-testid="slot-label">
            {{ abbreviateSlotId(slotId) }}
          </span>
          <div class="bar-container">
            <div
              class="stage-bar"
              :class="getStageClass(getSeedForSlot(slotId).stage)"
              :style="{ width: getBarWidth(getSeedForSlot(slotId).epochs_in_stage) }"
              :title="getBarTooltip(getSeedForSlot(slotId))"
              data-testid="stage-bar"
            />
          </div>
        </div>
      </div>

      <!-- Legend -->
      <div class="swimlane-legend" data-testid="swimlane-legend">
        <div
          v-for="stage in allStages"
          :key="stage"
          class="legend-item"
          data-testid="legend-item"
        >
          <span class="legend-color" :class="getStageClass(stage)" />
          <span class="legend-label">{{ formatStageName(stage) }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.seed-swimlane {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  padding: var(--space-sm);
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-lg);
  color: var(--text-dim);
  font-size: 12px;
  font-style: italic;
}

.swimlane-container {
  display: flex;
  flex-direction: column;
  gap: var(--space-xs);
}

.swimlane-row {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  height: 24px;
}

.slot-label {
  width: 32px;
  flex-shrink: 0;
  font-size: 10px;
  font-weight: 600;
  font-family: var(--font-mono);
  color: var(--text-secondary);
  text-align: right;
}

.bar-container {
  flex: 1;
  height: 16px;
  background: var(--bg-elevated);
  border-radius: 4px;
  overflow: hidden;
}

.stage-bar {
  height: 100%;
  min-width: 2px;
  border-radius: 4px;
  transition: width 0.3s ease;
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

/* Legend */
.swimlane-legend {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
  padding-top: var(--space-sm);
  border-top: 1px solid var(--border-subtle);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.legend-color {
  width: 12px;
  height: 12px;
  border-radius: 2px;
}

.legend-label {
  font-size: 10px;
  color: var(--text-secondary);
}
</style>
