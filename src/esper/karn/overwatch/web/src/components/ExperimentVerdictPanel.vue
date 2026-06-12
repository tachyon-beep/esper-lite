<!-- src/esper/karn/overwatch/web/src/components/ExperimentVerdictPanel.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { SanctumSnapshot } from '../types/sanctum'

const props = defineProps<{
  snapshot: SanctumSnapshot
}>()

type VerdictState = 'ready' | 'watch' | 'blocked' | 'collecting'
type GateState = 'pass' | 'watch' | 'fail' | 'pending'

interface GateRow {
  id: string
  label: string
  value: string
  state: GateState
}

interface Kpi {
  id: string
  label: string
  value: string
  tone: GateState
}

interface ActionShare {
  action: string
  count: number
  percent: number
}

const EPSILON = 0.000001

const progressPercent = computed(() => {
  if (props.snapshot.max_batches <= 0) {
    return 0
  }
  return Math.min(100, Math.round((props.snapshot.current_batch / props.snapshot.max_batches) * 100))
})

const progressText = computed(() => {
  return `${props.snapshot.current_batch}/${props.snapshot.max_batches}`
})

const completedEpisodes = computed(() => {
  return props.snapshot.episode_stats.total_episodes
})

const targetEpisodes = computed(() => {
  return props.snapshot.run_config.n_episodes
})

const episodeProgressText = computed(() => {
  return `${completedEpisodes.value}/${targetEpisodes.value}`
})

const dataGate = computed<GateRow>(() => {
  if (props.snapshot.training_thread_alive === false) {
    return {
      id: 'data-feed',
      label: 'Data Feed',
      value: 'thread stopped',
      state: 'fail'
    }
  }

  if (props.snapshot.total_events_received < 5 || props.snapshot.current_batch < 2) {
    return {
      id: 'data-feed',
      label: 'Data Feed',
      value: `${props.snapshot.total_events_received} events`,
      state: 'pending'
    }
  }

  return {
    id: 'data-feed',
    label: 'Data Feed',
    value: `${props.snapshot.total_events_received} events`,
    state: 'pass'
  }
})

const numericalGate = computed<GateRow>(() => {
  const hardFailures = props.snapshot.observation_stats.nan_count
    + props.snapshot.observation_stats.inf_count
    + props.snapshot.tamiyo.nan_grad_count
    + props.snapshot.tamiyo.inf_grad_count

  if (hardFailures > 0 || props.snapshot.tamiyo.lstm_has_nan || props.snapshot.tamiyo.lstm_has_inf) {
    return {
      id: 'numerics',
      label: 'Numerics',
      value: `${hardFailures} tensor faults`,
      state: 'fail'
    }
  }

  if (props.snapshot.observation_stats.clip_pct > 0.05) {
    return {
      id: 'numerics',
      label: 'Numerics',
      value: `${formatPercent(props.snapshot.observation_stats.clip_pct)} obs clipped`,
      state: 'watch'
    }
  }

  return {
    id: 'numerics',
    label: 'Numerics',
    value: 'clean',
    state: 'pass'
  }
})

const policyGate = computed<GateRow>(() => {
  if (props.snapshot.tamiyo.entropy_collapsed || props.snapshot.tamiyo.collapse_risk_score >= 0.85) {
    return {
      id: 'policy',
      label: 'Policy',
      value: `${formatPercent(props.snapshot.tamiyo.collapse_risk_score)} collapse risk`,
      state: 'fail'
    }
  }

  if (
    props.snapshot.tamiyo.collapse_risk_score >= 0.55
    || props.snapshot.tamiyo.explained_variance < 0.3
    || props.snapshot.tamiyo.decision_density < 0.35
  ) {
    return {
      id: 'policy',
      label: 'Policy',
      value: `EV ${formatSigned(props.snapshot.tamiyo.explained_variance)}`,
      state: 'watch'
    }
  }

  return {
    id: 'policy',
    label: 'Policy',
    value: `EV ${formatSigned(props.snapshot.tamiyo.explained_variance)}`,
    state: 'pass'
  }
})

const criticGate = computed<GateRow>(() => {
  if (props.snapshot.tamiyo.q_variance < 0.01 && props.snapshot.tamiyo.cumulative_total_actions >= 20) {
    return {
      id: 'critic',
      label: 'Critic',
      value: `var ${formatCompact(props.snapshot.tamiyo.q_variance)}`,
      state: 'fail'
    }
  }

  if (props.snapshot.tamiyo.q_variance < 0.1 && props.snapshot.tamiyo.cumulative_total_actions >= 20) {
    return {
      id: 'critic',
      label: 'Critic',
      value: `spread ${formatCompact(props.snapshot.tamiyo.q_spread)}`,
      state: 'watch'
    }
  }

  if (props.snapshot.tamiyo.cumulative_total_actions < 20) {
    return {
      id: 'critic',
      label: 'Critic',
      value: `${props.snapshot.tamiyo.cumulative_total_actions} actions`,
      state: 'pending'
    }
  }

  return {
    id: 'critic',
    label: 'Critic',
    value: `spread ${formatCompact(props.snapshot.tamiyo.q_spread)}`,
    state: 'pass'
  }
})

const lifecycleGate = computed<GateRow>(() => {
  if (props.snapshot.cumulative_germinated === 0) {
    return {
      id: 'lifecycle',
      label: 'Lifecycle',
      value: 'no germination',
      state: 'pending'
    }
  }

  if (props.snapshot.episode_stats.yield_rate < 0.2 && props.snapshot.cumulative_germinated >= 5) {
    return {
      id: 'lifecycle',
      label: 'Lifecycle',
      value: `${formatPercent(props.snapshot.episode_stats.yield_rate)} yield`,
      state: 'watch'
    }
  }

  if (props.snapshot.episode_stats.slot_utilization > 0.95 || props.snapshot.episode_stats.slot_utilization < 0.1) {
    return {
      id: 'lifecycle',
      label: 'Lifecycle',
      value: `${formatPercent(props.snapshot.episode_stats.slot_utilization)} slots`,
      state: 'watch'
    }
  }

  return {
    id: 'lifecycle',
    label: 'Lifecycle',
    value: `${formatPercent(props.snapshot.episode_stats.yield_rate)} yield`,
    state: 'pass'
  }
})

const gates = computed<GateRow[]>(() => [
  dataGate.value,
  numericalGate.value,
  policyGate.value,
  criticGate.value,
  lifecycleGate.value
])

const verdictState = computed<VerdictState>(() => {
  if (gates.value.some((gate) => gate.state === 'fail')) {
    return 'blocked'
  }
  if (gates.value.some((gate) => gate.state === 'pending')) {
    return 'collecting'
  }
  if (gates.value.some((gate) => gate.state === 'watch')) {
    return 'watch'
  }
  return 'ready'
})

const verdictLabel = computed(() => {
  switch (verdictState.value) {
    case 'blocked':
      return 'Blocked'
    case 'collecting':
      return 'Collecting'
    case 'watch':
      return 'Watch'
    case 'ready':
      return 'Interpretable'
  }
})

const kpis = computed<Kpi[]>(() => [
  {
    id: 'accuracy',
    label: 'Mean Acc',
    value: formatPercent(props.snapshot.aggregate_mean_accuracy),
    tone: props.snapshot.aggregate_mean_accuracy > 0 ? 'pass' : 'pending'
  },
  {
    id: 'reward',
    label: 'Batch Reward',
    value: formatSigned(props.snapshot.batch_avg_reward),
    tone: props.snapshot.batch_avg_reward >= 0 ? 'pass' : 'watch'
  },
  {
    id: 'return',
    label: 'Return p50',
    value: formatSigned(props.snapshot.tamiyo.value_function.return_p50),
    tone: props.snapshot.tamiyo.value_function.return_p50 >= 0 ? 'pass' : 'watch'
  },
  {
    id: 'value-corr',
    label: 'V/Return',
    value: formatSigned(props.snapshot.tamiyo.value_function.v_return_correlation),
    tone: props.snapshot.tamiyo.value_function.v_return_correlation >= 0.5 ? 'pass' : 'watch'
  },
  {
    id: 'q-spread',
    label: 'Q Spread',
    value: formatCompact(props.snapshot.tamiyo.q_spread),
    tone: criticGate.value.state
  },
  {
    id: 'risk',
    label: 'Risk',
    value: formatPercent(props.snapshot.tamiyo.collapse_risk_score),
    tone: policyGate.value.state
  }
])

const actionShares = computed<ActionShare[]>(() => {
  const total = props.snapshot.tamiyo.cumulative_total_actions
  if (total <= 0) {
    return []
  }

  return Object.entries(props.snapshot.tamiyo.cumulative_action_counts)
    .map(([action, count]) => ({
      action,
      count,
      percent: count / total
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 6)
})

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function formatSigned(value: number): string {
  return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3)
}

function formatCompact(value: number): string {
  if (Math.abs(value) < EPSILON) {
    return '0.000'
  }
  if (Math.abs(value) < 0.01) {
    return value.toExponential(1)
  }
  return value.toFixed(3)
}
</script>

<template>
  <section
    class="experiment-verdict"
    :class="`verdict-${verdictState}`"
    data-testid="experiment-verdict"
  >
    <div class="verdict-header">
      <div>
        <div class="run-line">
          <span class="task-name" data-testid="verdict-task">{{ snapshot.task_name }}</span>
          <span class="run-id" data-testid="verdict-run">{{ snapshot.run_id }}</span>
        </div>
        <div class="progress-line">
          <span>Batch {{ progressText }}</span>
          <span>Episodes {{ episodeProgressText }}</span>
          <span>Seed {{ snapshot.run_config.seed }}</span>
        </div>
      </div>

      <div
        class="verdict-pill"
        :class="`pill-${verdictState}`"
        data-testid="verdict-label"
      >
        {{ verdictLabel }}
      </div>
    </div>

    <div class="progress-track" aria-hidden="true">
      <div class="progress-fill" :style="{ width: `${progressPercent}%` }"></div>
    </div>

    <div class="kpi-grid" data-testid="verdict-kpis">
      <div
        v-for="kpi in kpis"
        :key="kpi.id"
        class="kpi-cell"
        :class="`gate-${kpi.tone}`"
        :data-testid="`kpi-${kpi.id}`"
      >
        <span class="kpi-label">{{ kpi.label }}</span>
        <span class="kpi-value">{{ kpi.value }}</span>
      </div>
    </div>

    <div class="gate-grid" data-testid="verdict-gates">
      <div
        v-for="gate in gates"
        :key="gate.id"
        class="gate-row"
        :class="`gate-${gate.state}`"
        :data-testid="`gate-${gate.id}`"
      >
        <span class="gate-dot" aria-hidden="true"></span>
        <span class="gate-label">{{ gate.label }}</span>
        <span class="gate-value">{{ gate.value }}</span>
      </div>
    </div>

    <div class="action-mix" data-testid="action-mix">
      <div
        v-for="share in actionShares"
        :key="share.action"
        class="action-row"
        :data-testid="`action-${share.action}`"
      >
        <span class="action-label">{{ share.action }}</span>
        <div class="action-track" aria-hidden="true">
          <div class="action-fill" :style="{ width: `${Math.round(share.percent * 100)}%` }"></div>
        </div>
        <span class="action-value">{{ Math.round(share.percent * 100) }}%</span>
      </div>
      <div v-if="actionShares.length === 0" class="action-empty" data-testid="action-empty">
        No actions
      </div>
    </div>
  </section>
</template>

<style scoped>
.experiment-verdict {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  padding: var(--space-md);
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
}

.verdict-ready {
  border-color: color-mix(in srgb, var(--status-win) 55%, var(--border-subtle));
}

.verdict-watch,
.verdict-collecting {
  border-color: color-mix(in srgb, var(--status-warn) 55%, var(--border-subtle));
}

.verdict-blocked {
  border-color: color-mix(in srgb, var(--status-loss) 60%, var(--border-subtle));
}

.verdict-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-md);
}

.run-line {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: var(--space-sm);
}

.task-name {
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 600;
  color: var(--text-bright);
}

.run-id {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
}

.progress-line {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-md);
  margin-top: var(--space-xs);
  font-size: 11px;
  color: var(--text-secondary);
}

.verdict-pill {
  flex: 0 0 auto;
  padding: 5px 10px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.pill-ready {
  background: var(--status-win);
  color: var(--bg-void);
}

.pill-watch,
.pill-collecting {
  background: var(--status-warn);
  color: var(--bg-void);
}

.pill-blocked {
  background: var(--status-loss);
  color: var(--bg-void);
}

.progress-track,
.action-track {
  height: 6px;
  overflow: hidden;
  background: var(--bg-elevated);
  border-radius: 2px;
}

.progress-fill {
  height: 100%;
  background: var(--glow-cyan);
  transition: width 0.2s ease;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: var(--space-sm);
}

.kpi-cell {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-sm);
  background: var(--bg-elevated);
  border-left: 3px solid var(--status-neutral);
}

.kpi-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
}

.kpi-value {
  font-family: var(--font-mono);
  font-size: 15px;
  font-weight: 700;
  color: var(--text-bright);
}

.gate-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: var(--space-sm);
}

.gate-row {
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

.action-mix {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-sm) var(--space-md);
}

.action-row {
  min-width: 0;
  display: grid;
  grid-template-columns: minmax(76px, 1fr) 2fr 34px;
  align-items: center;
  gap: var(--space-sm);
}

.action-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
}

.action-fill {
  height: 100%;
  background: var(--stage-training);
  transition: width 0.2s ease;
}

.action-value {
  text-align: right;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-primary);
}

.action-empty {
  font-size: 12px;
  color: var(--text-dim);
}

@media (max-width: 1400px) {
  .kpi-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .gate-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .action-mix {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .verdict-header {
    flex-direction: column;
  }

  .kpi-grid,
  .gate-grid,
  .action-mix {
    grid-template-columns: 1fr;
  }
}
</style>
