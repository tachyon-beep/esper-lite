<!-- src/esper/karn/overwatch/web/src/components/CohortComparisonPanel.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { BestRunRecord, EventLogEntry, SanctumSnapshot } from '../types/sanctum'

const props = defineProps<{
  snapshotsByGroup: Record<string, SanctumSnapshot>
  primaryGroupId: string | null
}>()

interface CohortRow {
  groupId: string
  rewardMode: string
  episodes: string
  finalAccuracy: number
  growthRatio: number
  accuracyRoi: number
  yieldRate: number
  actionEntropy: number
  collapseRisk: number
  blocked: boolean
  statusLabel: string
  statusDetail: string
  active: boolean
  leader: boolean
}

interface CohortSummary {
  tone: 'blocked' | 'leader' | 'collecting' | 'single'
  verdict: string
  label: string
  detail: string
}

const BASELINE_ACCURACY = 0.6
const EPSILON = 0.000001

const groupEntries = computed(() => {
  return Object.entries(props.snapshotsByGroup).sort(([left], [right]) => left.localeCompare(right))
})

const hasCohorts = computed(() => groupEntries.value.length >= 2)

function bestRun(snapshot: SanctumSnapshot): BestRunRecord | null {
  if (snapshot.best_runs.length === 0) {
    return null
  }
  return [...snapshot.best_runs].sort((a, b) => {
    const finalDiff = b.final_accuracy - a.final_accuracy
    if (Math.abs(finalDiff) > EPSILON) {
      return finalDiff
    }
    return b.peak_accuracy - a.peak_accuracy
  })[0]
}

function rewardMode(groupId: string, snapshot: SanctumSnapshot, run: BestRunRecord | null): string {
  if (snapshot.reward_mode !== null) {
    return snapshot.reward_mode
  }
  if (run !== null && run.reward_mode !== null) {
    return run.reward_mode
  }
  return groupId
}

function finalAccuracy(snapshot: SanctumSnapshot, run: BestRunRecord | null): number {
  if (run !== null) {
    return run.final_accuracy
  }
  return snapshot.aggregate_mean_accuracy
}

function addedGrowth(run: BestRunRecord | null): number {
  if (run === null) {
    return 0
  }
  return Math.max(EPSILON, run.growth_ratio - 1)
}

function roi(snapshot: SanctumSnapshot, run: BestRunRecord | null): number {
  if (run === null) {
    return 0
  }
  return (finalAccuracy(snapshot, run) - BASELINE_ACCURACY) / addedGrowth(run)
}

function isAnomalyEvent(event: EventLogEntry): boolean {
  return event.event_type.endsWith('_DETECTED') || event.event_type.includes('ANOMALY')
}

function formatAnomalyStatus(event: EventLogEntry): string {
  const detail = event.metadata.detail
  if (typeof detail === 'string' && detail.length > 0) {
    return `unstable: ${detail}`
  }
  return `unstable: ${event.message}`
}

function latestAnomalyStatus(snapshot: SanctumSnapshot): string | null {
  const anomaly = [...snapshot.event_log].reverse().find(isAnomalyEvent)
  if (anomaly === undefined) {
    return null
  }
  return formatAnomalyStatus(anomaly)
}

const leaderGroupId = computed(() => {
  if (!hasCohorts.value) {
    return null
  }

  const ranked = groupEntries.value
    .filter(([, snapshot]) => latestAnomalyStatus(snapshot) === null)
    .map(([groupId, snapshot]) => {
      const run = bestRun(snapshot)
      return {
        groupId,
        accuracyRoi: roi(snapshot, run),
        finalAccuracy: finalAccuracy(snapshot, run)
      }
    })
    .sort((a, b) => {
      const roiDiff = b.accuracyRoi - a.accuracyRoi
      if (Math.abs(roiDiff) > EPSILON) {
        return roiDiff
      }
      return b.finalAccuracy - a.finalAccuracy
    })

  if (ranked.length === 0) {
    return null
  }
  return ranked[0].groupId
})

const rows = computed<CohortRow[]>(() => {
  return groupEntries.value.map(([groupId, snapshot]) => {
    const run = bestRun(snapshot)
    const completed = snapshot.episode_stats.total_episodes
    const target = snapshot.run_config.n_episodes
    const anomalyStatus = latestAnomalyStatus(snapshot)

    return {
      groupId,
      rewardMode: rewardMode(groupId, snapshot, run),
      episodes: `${completed}/${target}`,
      finalAccuracy: finalAccuracy(snapshot, run),
      growthRatio: addedGrowth(run),
      accuracyRoi: roi(snapshot, run),
      yieldRate: snapshot.episode_stats.yield_rate,
      actionEntropy: snapshot.episode_stats.action_entropy,
      collapseRisk: snapshot.tamiyo.collapse_risk_score,
      blocked: anomalyStatus !== null,
      statusLabel: anomalyStatus === null ? 'Collecting' : 'Blocked',
      statusDetail: anomalyStatus === null ? 'telemetry still accumulating' : anomalyStatus,
      active: props.primaryGroupId === groupId,
      leader: anomalyStatus === null && leaderGroupId.value === groupId
    }
  })
})

const summary = computed<CohortSummary>(() => {
  if (!hasCohorts.value) {
    return {
      tone: 'single',
      verdict: 'single policy',
      label: 'single policy',
      detail: 'Waiting for grouped telemetry before comparing reward cohorts.'
    }
  }
  const blocked = rows.value.find((row) => row.blocked)
  if (blocked !== undefined) {
    return {
      tone: 'blocked',
      verdict: `${blocked.rewardMode} blocked`,
      label: 'do not advance',
      detail: `${blocked.groupId} / ${blocked.rewardMode} cannot be used for the Phase 2.5 verdict: ${blocked.statusDetail}`
    }
  }
  const leader = rows.value.find((row) => row.leader)
  if (leader === undefined) {
    return {
      tone: 'collecting',
      verdict: 'collecting',
      label: 'collecting',
      detail: 'Cohorts are reporting, but no usable leader has enough evidence yet.'
    }
  }
  return {
    tone: 'leader',
    verdict: `${leader.rewardMode} leads`,
    label: 'usable leader',
    detail: `${leader.groupId} currently leads on accuracy ROI; confirm against the phase gates before advancing.`
  }
})

const verdict = computed(() => {
  return summary.value.verdict
})

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function formatRatio(value: number): string {
  return `${value.toFixed(2)}x`
}
</script>

<template>
  <section class="cohort-comparison" data-testid="cohort-comparison">
    <div class="cohort-header">
      <div>
        <h3>Cohort Comparison</h3>
        <p>Phase 2.5 reward exam</p>
      </div>
      <span
        class="cohort-pill"
        :class="`pill-${summary.tone}`"
        data-testid="cohort-verdict"
      >
        {{ verdict }}
      </span>
    </div>

    <div
      class="cohort-summary"
      :class="`summary-${summary.tone}`"
      :role="summary.tone === 'blocked' ? 'alert' : 'status'"
      data-testid="cohort-summary"
    >
      <span class="summary-label">{{ summary.label }}</span>
      <span class="summary-detail">{{ summary.detail }}</span>
    </div>

    <div v-if="!hasCohorts" class="cohort-empty" data-testid="cohort-empty">
      Waiting for grouped telemetry; current run is single policy.
    </div>

    <div v-else class="cohort-table" data-testid="cohort-table">
      <div class="cohort-row cohort-table-head">
        <span>Group</span>
        <span>Reward</span>
        <span>Episodes</span>
        <span>Final</span>
        <span>Growth</span>
        <span>ROI</span>
        <span>Yield</span>
        <span>Entropy</span>
        <span>Risk</span>
      </div>
      <div
        v-for="row in rows"
        :key="row.groupId"
        class="cohort-row"
        :class="{ active: row.active, leader: row.leader, blocked: row.blocked }"
        :data-testid="`cohort-${row.groupId}`"
      >
        <span class="group-cell">
          <span class="group-id">{{ row.groupId }}</span>
          <span v-if="row.leader" class="leader-mark">lead</span>
        </span>
        <span class="reward-cell">
          <span>{{ row.rewardMode }}</span>
          <span
            class="status-chip"
            :class="{ blocked: row.blocked }"
            :data-testid="`cohort-status-${row.groupId}`"
          >
            {{ row.statusLabel }}
          </span>
          <span class="status-detail">{{ row.statusDetail }}</span>
        </span>
        <span class="metric-cell" data-label="Episodes">{{ row.episodes }}</span>
        <span class="metric-cell" data-label="Final">{{ formatPercent(row.finalAccuracy) }}</span>
        <span class="metric-cell" data-label="Growth">{{ formatPercent(row.growthRatio) }}</span>
        <span class="metric-cell" data-label="ROI">{{ formatRatio(row.accuracyRoi) }}</span>
        <span class="metric-cell" data-label="Yield">{{ formatPercent(row.yieldRate) }}</span>
        <span class="metric-cell" data-label="Entropy">{{ row.actionEntropy.toFixed(2) }}</span>
        <span class="metric-cell" data-label="Risk">{{ formatPercent(row.collapseRisk) }}</span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.cohort-comparison {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  padding: var(--space-md);
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
}

.cohort-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-md);
}

h3 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 15px;
  color: var(--text-bright);
}

p {
  margin: 2px 0 0;
  font-size: 11px;
  color: var(--text-secondary);
}

.cohort-pill {
  padding: var(--space-xs) var(--space-sm);
  border-radius: 4px;
  background: var(--glow-cyan-dim);
  border: 1px solid var(--glow-cyan);
  color: var(--glow-cyan);
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  white-space: nowrap;
}

.pill-blocked {
  background: var(--status-loss-glow);
  border-color: var(--status-loss);
  color: var(--status-loss);
}

.pill-leader {
  background: var(--status-win-glow);
  border-color: var(--status-win);
  color: var(--status-win);
}

.cohort-summary {
  display: grid;
  grid-template-columns: minmax(120px, 0.35fr) minmax(0, 1fr);
  gap: var(--space-sm);
  align-items: center;
  padding: var(--space-sm);
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--glow-cyan);
  background: var(--bg-primary);
}

.summary-blocked {
  border-left-color: var(--status-loss);
  background: color-mix(in srgb, var(--status-loss) 9%, var(--bg-primary));
}

.summary-leader {
  border-left-color: var(--status-win);
}

.summary-label {
  color: var(--text-bright);
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}

.summary-detail {
  color: var(--text-primary);
  font-size: 12px;
  line-height: 1.35;
}

.cohort-empty {
  padding: var(--space-md);
  color: var(--text-secondary);
  border: 1px solid var(--border-subtle);
  background: var(--bg-primary);
}

.cohort-table {
  overflow-x: auto;
}

.cohort-row {
  display: grid;
  grid-template-columns: 58px minmax(168px, 1.65fr) repeat(7, minmax(48px, 0.68fr));
  min-width: 704px;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-sm);
  border-bottom: 1px solid var(--border-subtle);
  font-size: 12px;
  color: var(--text-primary);
}

.cohort-table-head {
  background: var(--bg-elevated);
  color: var(--text-secondary);
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.cohort-row.active {
  outline: 1px solid var(--glow-cyan);
  outline-offset: -1px;
}

.cohort-row.leader {
  background: var(--glow-cyan-dim);
}

.cohort-row.blocked {
  background: color-mix(in srgb, var(--status-loss) 10%, transparent);
  outline: 1px solid var(--status-loss);
  outline-offset: -1px;
}

.group-cell {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
}

.group-id {
  font-weight: 700;
  color: var(--text-bright);
}

.reward-cell {
  display: grid;
  grid-template-columns: auto auto;
  gap: 2px var(--space-xs);
  align-items: center;
  min-width: 0;
}

.status-chip {
  justify-self: start;
  padding: 1px 5px;
  border: 1px solid var(--border-subtle);
  border-radius: 3px;
  color: var(--text-secondary);
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
}

.status-chip.blocked {
  border-color: var(--status-loss);
  background: var(--status-loss-glow);
  color: var(--status-loss);
}

.status-detail {
  grid-column: 1 / -1;
  overflow: hidden;
  color: var(--text-secondary);
  font-size: 10px;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cohort-row.blocked .status-detail {
  color: var(--status-loss);
  font-weight: 700;
}

.leader-mark {
  padding: 1px 4px;
  border-radius: 3px;
  background: var(--status-win-glow);
  color: var(--status-win);
  font-size: 9px;
  text-transform: uppercase;
}

@media (max-width: 720px) {
  .cohort-header,
  .cohort-summary {
    grid-template-columns: 1fr;
  }

  .cohort-header {
    display: grid;
  }

  .cohort-pill {
    justify-self: start;
  }

  .cohort-table {
    overflow-x: visible;
  }

  .cohort-table-head {
    display: none;
  }

  .cohort-row {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    min-width: 0;
    gap: var(--space-sm);
    padding: var(--space-sm) var(--space-xs);
  }

  .group-cell {
    grid-column: 1;
    align-self: start;
  }

  .reward-cell {
    grid-column: 2 / -1;
  }

  .metric-cell {
    display: grid;
    gap: 2px;
    min-width: 0;
  }

  .metric-cell::before {
    content: attr(data-label);
    color: var(--text-secondary);
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
  }
}
</style>
