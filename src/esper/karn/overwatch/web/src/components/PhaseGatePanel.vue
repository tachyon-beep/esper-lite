<!-- src/esper/karn/overwatch/web/src/components/PhaseGatePanel.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { BestRunRecord, SanctumSnapshot } from '../types/sanctum'

const props = defineProps<{
  snapshot: SanctumSnapshot
}>()

type GateState = 'pass' | 'watch' | 'fail' | 'pending'

interface GateRow {
  id: string
  label: string
  value: string
  state: GateState
}

interface EvidenceMetric {
  id: string
  label: string
  value: string
  state: GateState
}

const BASELINE_ACCURACY = 0.6
const BASELINE_GROWTH_RATIO = 0.1
const MIN_EVIDENCE_EPISODES = 10
const EPSILON = 0.000001

const bestRun = computed<BestRunRecord | null>(() => {
  if (props.snapshot.best_runs.length === 0) {
    return null
  }

  return [...props.snapshot.best_runs].sort((a, b) => {
    const finalDiff = b.final_accuracy - a.final_accuracy
    if (Math.abs(finalDiff) > EPSILON) {
      return finalDiff
    }
    return b.peak_accuracy - a.peak_accuracy
  })[0]
})

const completedEpisodes = computed(() => props.snapshot.episode_stats.total_episodes)
const targetEpisodes = computed(() => props.snapshot.run_config.n_episodes)

const evidenceRatio = computed(() => {
  if (targetEpisodes.value <= 0) {
    return 0
  }
  return Math.min(1, completedEpisodes.value / targetEpisodes.value)
})

const finalAccuracy = computed(() => {
  if (bestRun.value === null) {
    return props.snapshot.aggregate_mean_accuracy
  }
  return bestRun.value.final_accuracy
})

const addedParamRatio = computed(() => {
  if (bestRun.value === null) {
    return 0
  }
  return Math.max(EPSILON, bestRun.value.growth_ratio - 1)
})

const accuracyDelta = computed(() => finalAccuracy.value - BASELINE_ACCURACY)

const accuracyRoi = computed(() => {
  if (bestRun.value === null) {
    return 0
  }
  return accuracyDelta.value / addedParamRatio.value
})

const lifecycleEfficiency = computed(() => {
  if (props.snapshot.cumulative_pruned === 0) {
    return props.snapshot.cumulative_fossilized
  }
  return props.snapshot.cumulative_fossilized / props.snapshot.cumulative_pruned
})

const completionGate = computed<GateRow>(() => {
  if (completedEpisodes.value >= targetEpisodes.value && targetEpisodes.value > 0) {
    return {
      id: 'completion',
      label: 'Evidence',
      value: `${completedEpisodes.value}/${targetEpisodes.value} episodes`,
      state: 'pass'
    }
  }

  if (completedEpisodes.value >= MIN_EVIDENCE_EPISODES) {
    return {
      id: 'completion',
      label: 'Evidence',
      value: `${completedEpisodes.value}/${targetEpisodes.value} episodes`,
      state: 'watch'
    }
  }

  return {
    id: 'completion',
    label: 'Evidence',
    value: `${completedEpisodes.value}/${targetEpisodes.value} episodes`,
    state: 'pending'
  }
})

const baselineGate = computed<GateRow>(() => {
  if (bestRun.value === null) {
    return {
      id: 'baseline',
      label: 'Control',
      value: 'no complete run',
      state: 'pending'
    }
  }

  if (finalAccuracy.value <= BASELINE_ACCURACY) {
    return {
      id: 'baseline',
      label: 'Control',
      value: `${formatPercent(finalAccuracy.value)} <= ${formatPercent(BASELINE_ACCURACY)}`,
      state: 'fail'
    }
  }

  return {
    id: 'baseline',
    label: 'Control',
    value: `${formatSignedPercent(accuracyDelta.value)} vs baseline`,
    state: 'pass'
  }
})

const efficiencyGate = computed<GateRow>(() => {
  if (bestRun.value === null) {
    return {
      id: 'efficiency',
      label: 'ROI',
      value: 'waiting',
      state: 'pending'
    }
  }

  if (accuracyDelta.value <= 0) {
    return {
      id: 'efficiency',
      label: 'ROI',
      value: 'negative',
      state: 'fail'
    }
  }

  if (addedParamRatio.value > BASELINE_GROWTH_RATIO && accuracyRoi.value < 2) {
    return {
      id: 'efficiency',
      label: 'ROI',
      value: `${formatRatio(accuracyRoi.value)} at ${formatPercent(addedParamRatio.value)} growth`,
      state: 'watch'
    }
  }

  return {
    id: 'efficiency',
    label: 'ROI',
    value: `${formatRatio(accuracyRoi.value)} at ${formatPercent(addedParamRatio.value)} growth`,
    state: 'pass'
  }
})

const lifecycleGate = computed<GateRow>(() => {
  if (props.snapshot.cumulative_germinated < 3) {
    return {
      id: 'lifecycle',
      label: 'Lifecycle',
      value: `${props.snapshot.cumulative_germinated} germinated`,
      state: 'pending'
    }
  }

  if (props.snapshot.episode_stats.yield_rate < 0.2 || lifecycleEfficiency.value < 0.5) {
    return {
      id: 'lifecycle',
      label: 'Lifecycle',
      value: `${formatPercent(props.snapshot.episode_stats.yield_rate)} yield`,
      state: 'watch'
    }
  }

  return {
    id: 'lifecycle',
    label: 'Lifecycle',
    value: `${formatLifecycleRatio(lifecycleEfficiency.value)} foss/prune`,
    state: 'pass'
  }
})

const decisionGate = computed<GateRow>(() => {
  if (props.snapshot.tamiyo.cumulative_total_actions < 20) {
    return {
      id: 'decision',
      label: 'Decision',
      value: `${props.snapshot.tamiyo.cumulative_total_actions} actions`,
      state: 'pending'
    }
  }

  if (props.snapshot.episode_stats.action_entropy < 0.15) {
    return {
      id: 'decision',
      label: 'Decision',
      value: 'collapsed',
      state: 'fail'
    }
  }

  if (props.snapshot.episode_stats.action_entropy > 1.55 || props.snapshot.tamiyo.decision_density < 0.35) {
    return {
      id: 'decision',
      label: 'Decision',
      value: `entropy ${props.snapshot.episode_stats.action_entropy.toFixed(2)}`,
      state: 'watch'
    }
  }

  return {
    id: 'decision',
    label: 'Decision',
    value: `entropy ${props.snapshot.episode_stats.action_entropy.toFixed(2)}`,
    state: 'pass'
  }
})

const gates = computed<GateRow[]>(() => [
  completionGate.value,
  baselineGate.value,
  efficiencyGate.value,
  lifecycleGate.value,
  decisionGate.value
])

const gateState = computed<GateState>(() => {
  if (gates.value.some((gate) => gate.state === 'fail')) {
    return 'fail'
  }
  if (gates.value.some((gate) => gate.state === 'pending')) {
    return 'pending'
  }
  if (gates.value.some((gate) => gate.state === 'watch')) {
    return 'watch'
  }
  return 'pass'
})

const gateLabel = computed(() => {
  switch (gateState.value) {
    case 'fail':
      return 'Investigate'
    case 'pending':
      return 'Keep Running'
    case 'watch':
      return 'Review'
    case 'pass':
      return 'Candidate'
  }
})

const nextAction = computed(() => {
  switch (gateState.value) {
    case 'fail':
      return 'A blocker contradicts the reward-efficiency gate.'
    case 'pending':
      return 'Evidence is still below the planned exam horizon.'
    case 'watch':
      return 'Evidence is usable, but at least one gate needs operator review.'
    case 'pass':
      return 'This run can be compared against the companion cohorts.'
  }
})

const evidenceMetrics = computed<EvidenceMetric[]>(() => [
  {
    id: 'baseline',
    label: 'Baseline',
    value: `${formatPercent(BASELINE_ACCURACY)} @ ${formatPercent(BASELINE_GROWTH_RATIO)} growth`,
    state: 'pending'
  },
  {
    id: 'final',
    label: 'Best Final',
    value: bestRun.value === null ? '--' : formatPercent(finalAccuracy.value),
    state: baselineGate.value.state
  },
  {
    id: 'growth',
    label: 'Added Params',
    value: bestRun.value === null ? '--' : formatPercent(addedParamRatio.value),
    state: efficiencyGate.value.state
  },
  {
    id: 'roi',
    label: 'Accuracy ROI',
    value: bestRun.value === null ? '--' : formatRatio(accuracyRoi.value),
    state: efficiencyGate.value.state
  }
])

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function formatSignedPercent(value: number): string {
  return value >= 0 ? `+${formatPercent(value)}` : `-${formatPercent(Math.abs(value))}`
}

function formatRatio(value: number): string {
  return `${value.toFixed(2)}x`
}

function formatLifecycleRatio(value: number): string {
  if (props.snapshot.cumulative_pruned === 0) {
    return `${props.snapshot.cumulative_fossilized}:0`
  }
  return value.toFixed(2)
}
</script>

<template>
  <section
    class="phase-gate"
    :class="`phase-${gateState}`"
    data-testid="phase-gate"
  >
    <div class="phase-header">
      <div>
        <h3>Phase 2.5 Gate</h3>
        <p>{{ snapshot.reward_mode ?? 'unknown' }} reward signal</p>
      </div>
      <div
        class="phase-pill"
        :class="`pill-${gateState}`"
        data-testid="phase-gate-label"
      >
        {{ gateLabel }}
      </div>
    </div>

    <div class="evidence-track" aria-hidden="true">
      <div class="evidence-fill" :style="{ width: `${Math.round(evidenceRatio * 100)}%` }"></div>
    </div>

    <div class="evidence-grid" data-testid="phase-evidence">
      <div
        v-for="metric in evidenceMetrics"
        :key="metric.id"
        class="evidence-cell"
        :class="`gate-${metric.state}`"
        :data-testid="`phase-metric-${metric.id}`"
      >
        <span>{{ metric.label }}</span>
        <strong>{{ metric.value }}</strong>
      </div>
    </div>

    <div class="gate-list" data-testid="phase-gates">
      <div
        v-for="gate in gates"
        :key="gate.id"
        class="gate-item"
        :class="`gate-${gate.state}`"
        :data-testid="`phase-gate-${gate.id}`"
      >
        <span class="gate-dot" aria-hidden="true"></span>
        <span class="gate-label">{{ gate.label }}</span>
        <span class="gate-value">{{ gate.value }}</span>
      </div>
    </div>

    <div class="next-action" data-testid="phase-next-action">
      {{ nextAction }}
    </div>
  </section>
</template>

<style scoped>
.phase-gate {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  padding: var(--space-md);
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
}

.phase-pass {
  border-color: color-mix(in srgb, var(--status-win) 55%, var(--border-subtle));
}

.phase-watch,
.phase-pending {
  border-color: color-mix(in srgb, var(--status-warn) 55%, var(--border-subtle));
}

.phase-fail {
  border-color: color-mix(in srgb, var(--status-loss) 60%, var(--border-subtle));
}

.phase-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-md);
}

h3,
p {
  margin: 0;
}

h3 {
  font-family: var(--font-display);
  font-size: 15px;
  font-weight: 600;
  color: var(--text-bright);
}

p {
  margin-top: 2px;
  font-size: 11px;
  color: var(--text-secondary);
}

.phase-pill {
  flex: 0 0 auto;
  padding: 5px 9px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.pill-pass {
  background: var(--status-win);
  color: var(--bg-void);
}

.pill-watch,
.pill-pending {
  background: var(--status-warn);
  color: var(--bg-void);
}

.pill-fail {
  background: var(--status-loss);
  color: var(--bg-void);
}

.evidence-track {
  height: 5px;
  overflow: hidden;
  background: var(--bg-elevated);
  border-radius: 2px;
}

.evidence-fill {
  height: 100%;
  background: var(--glow-cyan);
  transition: width 0.2s ease;
}

.evidence-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--space-sm);
}

.evidence-cell {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-sm);
  background: var(--bg-elevated);
  border-left: 3px solid var(--status-neutral);
}

.evidence-cell span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
}

.evidence-cell strong {
  font-family: var(--font-mono);
  font-size: 14px;
  color: var(--text-bright);
}

.gate-list {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-sm);
}

.gate-item {
  min-width: 0;
  display: grid;
  grid-template-columns: 8px 1fr;
  grid-template-areas:
    "dot label"
    "dot value";
  column-gap: var(--space-sm);
  align-items: center;
  padding: var(--space-sm);
  background: var(--bg-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 3px;
}

.gate-dot {
  grid-area: dot;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--status-neutral);
}

.gate-label {
  grid-area: label;
  font-size: 10px;
  color: var(--text-secondary);
  text-transform: uppercase;
}

.gate-value {
  grid-area: value;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-primary);
}

.gate-pass {
  border-color: color-mix(in srgb, var(--status-win) 45%, var(--border-subtle));
}

.gate-pass .gate-dot {
  background: var(--status-win);
}

.gate-watch,
.gate-pending {
  border-color: color-mix(in srgb, var(--status-warn) 45%, var(--border-subtle));
}

.gate-watch .gate-dot,
.gate-pending .gate-dot {
  background: var(--status-warn);
}

.gate-fail {
  border-color: color-mix(in srgb, var(--status-loss) 50%, var(--border-subtle));
}

.gate-fail .gate-dot {
  background: var(--status-loss);
}

.next-action {
  padding: var(--space-sm);
  background: var(--bg-primary);
  border-left: 3px solid var(--glow-cyan);
  font-size: 12px;
  color: var(--text-primary);
}

@media (max-width: 1400px) {
  .evidence-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .gate-list {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .phase-header {
    flex-direction: column;
  }

  .evidence-grid,
  .gate-list {
    grid-template-columns: 1fr;
  }
}
</style>
