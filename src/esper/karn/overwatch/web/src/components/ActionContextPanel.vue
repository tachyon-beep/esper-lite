<!-- src/esper/karn/overwatch/web/src/components/ActionContextPanel.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { DecisionSnapshot, SanctumSnapshot } from '../types/sanctum'

const props = defineProps<{
  snapshot: SanctumSnapshot
}>()

const OP_NAMES = ['WAIT', 'GERMINATE', 'SET_ALPHA_TARGET', 'PRUNE', 'FOSSILIZE', 'ADVANCE']
const ACTION_ABBREVS: Record<string, string> = {
  WAIT: 'WAIT',
  GERMINATE: 'GERM',
  SET_ALPHA_TARGET: 'ALPH',
  PRUNE: 'PRUN',
  FOSSILIZE: 'FOSS',
  ADVANCE: 'ADVN'
}
const ACTION_GLYPHS: Record<string, string> = {
  WAIT: 'W',
  GERMINATE: 'G',
  SET_ALPHA_TARGET: 'A',
  PRUNE: 'P',
  FOSSILIZE: 'F',
  ADVANCE: 'V'
}

interface CriticRow {
  action: string
  label: string
  qValue: number
  valid: boolean
  width: number
  rank: 'best' | 'worst' | 'mid' | 'masked'
}

const validCriticValues = computed(() => {
  return OP_NAMES
    .map((action, index) => ({
      action,
      qValue: props.snapshot.tamiyo.op_q_values[index],
      valid: props.snapshot.tamiyo.op_valid_mask[index]
    }))
    .filter((row) => row.valid && !Number.isNaN(row.qValue))
})

const criticRows = computed<CriticRow[]>(() => {
  const sortedValid = [...validCriticValues.value].sort((a, b) => b.qValue - a.qValue)
  const masked = OP_NAMES
    .map((action, index) => ({
      action,
      qValue: props.snapshot.tamiyo.op_q_values[index],
      valid: props.snapshot.tamiyo.op_valid_mask[index]
    }))
    .filter((row) => !row.valid || Number.isNaN(row.qValue))

  const qMin = Math.min(...sortedValid.map((row) => row.qValue))
  const qMax = Math.max(...sortedValid.map((row) => row.qValue))
  const qRange = qMax - qMin

  const ranked: CriticRow[] = sortedValid.map((row, index) => ({
    action: row.action,
    label: ACTION_ABBREVS[row.action],
    qValue: row.qValue,
    valid: true,
    width: qRange === 0 ? 50 : Math.round(((row.qValue - qMin) / qRange) * 100),
    rank: index === 0 ? 'best' : index === sortedValid.length - 1 ? 'worst' : 'mid'
  }))

  return ranked.concat(
    masked.map((row) => ({
      action: row.action,
      label: ACTION_ABBREVS[row.action],
      qValue: row.qValue,
      valid: false,
      width: 0,
      rank: 'masked'
    }))
  )
})

const qHealth = computed(() => {
  if (props.snapshot.tamiyo.q_variance < 0.01) {
    return 'flat'
  }
  if (props.snapshot.tamiyo.q_variance < 0.1) {
    return 'weak'
  }
  return 'separated'
})

const returns = computed(() => props.snapshot.tamiyo.value_function)

const rewardRows = computed(() => [
  { id: 'total', label: 'Total', value: props.snapshot.rewards.total },
  { id: 'attrib', label: 'Attrib', value: props.snapshot.rewards.bounded_attribution },
  { id: 'rent', label: 'Rent', value: props.snapshot.rewards.compute_rent },
  { id: 'escrow', label: 'Escrow', value: props.snapshot.rewards.escrow_delta },
  { id: 'hindsight', label: 'Hindsight', value: props.snapshot.rewards.hindsight_credit }
])

const decisionSequence = computed<DecisionSnapshot[]>(() => {
  return [...props.snapshot.tamiyo.recent_decisions].slice(0, 8).reverse()
})

const lastDecision = computed(() => props.snapshot.tamiyo.recent_decisions[0])

function formatSigned(value: number): string {
  return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3)
}

function formatCompact(value: number): string {
  if (Math.abs(value) < 0.01) {
    return value.toExponential(1)
  }
  return value.toFixed(3)
}

function actionLabel(action: string): string {
  return ACTION_ABBREVS[action]
}

function actionGlyph(action: string): string {
  return ACTION_GLYPHS[action]
}

function successText(decision: DecisionSnapshot): string {
  if (decision.action_success === true) {
    return 'OK'
  }
  if (decision.action_success === false) {
    return 'FAIL'
  }
  return 'PEND'
}
</script>

<template>
  <div class="action-context" data-testid="action-context">
    <section class="context-section critic-section">
      <div class="section-header">
        <h4>Critic Preference</h4>
        <span class="health-pill" :class="`q-${qHealth}`" data-testid="q-health">
          {{ qHealth }}
        </span>
      </div>

      <div class="critic-list" data-testid="critic-list">
        <div
          v-for="row in criticRows"
          :key="row.action"
          class="critic-row"
          :class="[`rank-${row.rank}`, { masked: !row.valid }]"
          :data-testid="`critic-${row.action}`"
        >
          <span class="critic-action">{{ row.label }}</span>
          <div class="critic-track" aria-hidden="true">
            <div class="critic-fill" :style="{ width: `${row.width}%` }"></div>
          </div>
          <span class="critic-value">{{ row.valid ? formatSigned(row.qValue) : '--' }}</span>
        </div>
      </div>

      <div class="critic-footer">
        <span>Var {{ formatCompact(snapshot.tamiyo.q_variance) }}</span>
        <span>Spread {{ formatCompact(snapshot.tamiyo.q_spread) }}</span>
      </div>
    </section>

    <section class="context-section">
      <h4>Return Shape</h4>
      <div class="return-grid" data-testid="return-shape">
        <div>
          <span class="metric-label">p10</span>
          <span class="metric-value">{{ formatSigned(returns.return_p10) }}</span>
        </div>
        <div>
          <span class="metric-label">p50</span>
          <span class="metric-value">{{ formatSigned(returns.return_p50) }}</span>
        </div>
        <div>
          <span class="metric-label">p90</span>
          <span class="metric-value">{{ formatSigned(returns.return_p90) }}</span>
        </div>
        <div>
          <span class="metric-label">V/R</span>
          <span class="metric-value">{{ formatSigned(returns.v_return_correlation) }}</span>
        </div>
      </div>
    </section>

    <section class="context-section">
      <h4>Reward Signal</h4>
      <div class="reward-grid" data-testid="reward-signal">
        <div
          v-for="row in rewardRows"
          :key="row.id"
          class="reward-cell"
          :class="{ negative: row.value < 0 }"
          :data-testid="`reward-${row.id}`"
        >
          <span class="metric-label">{{ row.label }}</span>
          <span class="metric-value">{{ formatSigned(row.value) }}</span>
        </div>
      </div>
    </section>

    <section class="context-section">
      <h4>Decision Sequence</h4>
      <div v-if="decisionSequence.length > 0" class="decision-sequence" data-testid="decision-sequence">
        <span
          v-for="decision in decisionSequence"
          :key="decision.decision_id"
          class="decision-token"
          :class="{ failed: decision.action_success === false }"
          :data-testid="`decision-${decision.decision_id}`"
        >
          {{ actionGlyph(decision.chosen_action) }} {{ successText(decision) }}
        </span>
      </div>
      <div v-else class="empty-sequence" data-testid="decision-empty">
        No decisions
      </div>

      <div v-if="lastDecision" class="last-decision" data-testid="last-decision">
        <span>{{ actionLabel(lastDecision.chosen_action) }}</span>
        <span>conf {{ Math.round(lastDecision.confidence * 100) }}%</span>
        <span>td {{ lastDecision.td_advantage === null ? '--' : formatSigned(lastDecision.td_advantage) }}</span>
      </div>
    </section>
  </div>
</template>

<style scoped>
.action-context {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  padding: var(--space-md);
}

.context-section {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-sm);
}

h4 {
  margin: 0;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.health-pill {
  padding: 2px 6px;
  border-radius: 2px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
}

.q-separated {
  background: var(--status-win);
  color: var(--bg-void);
}

.q-weak {
  background: var(--status-warn);
  color: var(--bg-void);
}

.q-flat {
  background: var(--status-loss);
  color: var(--bg-void);
}

.critic-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-xs);
}

.critic-row {
  display: grid;
  grid-template-columns: 44px 1fr 72px;
  align-items: center;
  gap: var(--space-sm);
  min-width: 0;
}

.critic-action,
.critic-value {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-primary);
}

.critic-value {
  text-align: right;
}

.critic-track {
  height: 8px;
  overflow: hidden;
  background: var(--bg-elevated);
  border-radius: 2px;
}

.critic-fill {
  height: 100%;
  min-width: 2px;
  background: var(--glow-cyan);
}

.rank-best .critic-fill {
  background: var(--status-win);
}

.rank-worst .critic-fill {
  background: var(--status-loss);
}

.masked {
  opacity: 0.55;
}

.critic-footer {
  display: flex;
  gap: var(--space-md);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
}

.return-grid,
.reward-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-sm);
}

.return-grid > div,
.reward-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-sm);
  background: var(--bg-elevated);
  border-left: 2px solid var(--glow-cyan);
  min-width: 0;
}

.reward-cell.negative {
  border-left-color: var(--status-loss);
}

.metric-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
}

.metric-value {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 700;
  color: var(--text-bright);
}

.decision-sequence {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-xs);
}

.decision-token {
  padding: 3px 6px;
  border: 1px solid var(--status-win);
  border-radius: 2px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-primary);
  background: var(--bg-elevated);
}

.decision-token.failed {
  border-color: var(--status-loss);
}

.empty-sequence {
  color: var(--text-dim);
  font-size: 12px;
  font-style: italic;
}

.last-decision {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
}
</style>
