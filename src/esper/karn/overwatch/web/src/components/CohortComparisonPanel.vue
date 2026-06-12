<!-- src/esper/karn/overwatch/web/src/components/CohortComparisonPanel.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { BestRunRecord, SanctumSnapshot } from '../types/sanctum'

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
  active: boolean
  leader: boolean
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

const leaderGroupId = computed(() => {
  if (!hasCohorts.value) {
    return null
  }

  const ranked = groupEntries.value
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

  return ranked[0].groupId
})

const rows = computed<CohortRow[]>(() => {
  return groupEntries.value.map(([groupId, snapshot]) => {
    const run = bestRun(snapshot)
    const completed = snapshot.episode_stats.total_episodes
    const target = snapshot.run_config.n_episodes

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
      active: props.primaryGroupId === groupId,
      leader: leaderGroupId.value === groupId
    }
  })
})

const verdict = computed(() => {
  if (!hasCohorts.value) {
    return 'single policy'
  }
  const leader = rows.value.find((row) => row.leader)
  return leader === undefined ? 'collecting' : `${leader.rewardMode} leads`
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
      <span class="cohort-pill" data-testid="cohort-verdict">{{ verdict }}</span>
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
        :class="{ active: row.active, leader: row.leader }"
        :data-testid="`cohort-${row.groupId}`"
      >
        <span class="group-cell">
          <span class="group-id">{{ row.groupId }}</span>
          <span v-if="row.leader" class="leader-mark">lead</span>
        </span>
        <span>{{ row.rewardMode }}</span>
        <span>{{ row.episodes }}</span>
        <span>{{ formatPercent(row.finalAccuracy) }}</span>
        <span>{{ formatPercent(row.growthRatio) }}</span>
        <span>{{ formatRatio(row.accuracyRoi) }}</span>
        <span>{{ formatPercent(row.yieldRate) }}</span>
        <span>{{ row.actionEntropy.toFixed(2) }}</span>
        <span>{{ formatPercent(row.collapseRisk) }}</span>
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
  grid-template-columns: 70px minmax(94px, 1fr) repeat(7, minmax(64px, 0.7fr));
  min-width: 760px;
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

.group-cell {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
}

.group-id {
  font-weight: 700;
  color: var(--text-bright);
}

.leader-mark {
  padding: 1px 4px;
  border-radius: 3px;
  background: var(--status-win-glow);
  color: var(--status-win);
  font-size: 9px;
  text-transform: uppercase;
}
</style>
