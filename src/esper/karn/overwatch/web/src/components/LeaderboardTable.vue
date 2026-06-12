<!-- src/esper/karn/overwatch/web/src/components/LeaderboardTable.vue -->
<script setup lang="ts">
import { ref, computed } from 'vue'
import type { BestRunRecord, SeedState } from '../types/sanctum'

const props = withDefaults(defineProps<{
  runs: BestRunRecord[]
  maxRows?: number
  /** Currently selected row index (for keyboard navigation) */
  selectedRowIndex?: number
}>(), {
  maxRows: 10,
  selectedRowIndex: -1
})

const emit = defineEmits<{
  select: [recordId: string]
}>()

type SortColumn =
  | 'env_id'
  | 'episode'
  | 'epoch'
  | 'peak_accuracy'
  | 'growth_ratio'
  | 'cumulative_reward'

const sortColumn = ref<SortColumn>('peak_accuracy')
const sortAscending = ref(false)

const sortedRuns = computed(() => {
  const sorted = [...props.runs].sort((a, b) => {
    const aVal = a[sortColumn.value]
    const bVal = b[sortColumn.value]
    const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0
    return sortAscending.value ? cmp : -cmp
  })
  return sorted.slice(0, props.maxRows)
})

function handleSort(column: SortColumn) {
  if (sortColumn.value === column) {
    sortAscending.value = !sortAscending.value
  } else {
    sortColumn.value = column
    sortAscending.value = column === 'env_id' || column === 'episode'
  }
}

function formatAccuracy(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatGrowthRatio(value: number): string {
  return `${value.toFixed(2)}x`
}

function formatReward(value: number): string {
  if (Math.abs(value) < 0.1) {
    return '~0.0'
  }
  return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1)
}

function trajectoryClass(run: BestRunRecord): string {
  const deltaPoints = (run.final_accuracy - run.peak_accuracy) * 100
  if (deltaPoints > 0.5) {
    return 'trajectory-up'
  }
  if (deltaPoints >= -1.0) {
    return 'trajectory-held'
  }
  if (deltaPoints >= -2.0) {
    return 'trajectory-soft-down'
  }
  return 'trajectory-down'
}

function trajectoryGlyph(run: BestRunRecord): string {
  const tone = trajectoryClass(run)
  if (tone === 'trajectory-up') {
    return '\u2197'
  }
  if (tone === 'trajectory-held') {
    return '\u2500\u2192'
  }
  return '\u2198'
}

function seedSnapshot(run: BestRunRecord): Record<string, SeedState> {
  const endSeeds = Object.values(run.end_seeds)
  return endSeeds.length > 0 ? run.end_seeds : run.seeds
}

function formatSeedComposition(run: BestRunRecord): string {
  const seeds = Object.values(seedSnapshot(run))
  const blendingCount = seeds.filter(seed => seed.stage === 'BLENDING').length
  const holdingCount = seeds.filter(seed => seed.stage === 'HOLDING').length
  const fossilizedCount = seeds.filter(seed => seed.stage === 'FOSSILIZED').length

  if (blendingCount + holdingCount + fossilizedCount === 0) {
    return '-'
  }

  return `${blendingCount}/${holdingCount}/${fossilizedCount}`
}

function handleRowClick(recordId: string) {
  emit('select', recordId)
}
</script>

<template>
  <div class="leaderboard-table">
    <div v-if="runs.length === 0" class="empty-state" data-testid="empty-state">
      No runs recorded
    </div>
    <table v-else>
      <thead>
        <tr>
          <th class="rank-col">Rank</th>
          <th
            class="sortable"
            data-testid="header-env"
            @click="handleSort('env_id')"
          >
            Env
            <span v-if="sortColumn === 'env_id'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
          <th
            class="sortable"
            data-testid="header-episode"
            @click="handleSort('episode')"
          >
            Ep
            <span v-if="sortColumn === 'episode'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
          <th
            class="sortable"
            data-testid="header-epoch"
            @click="handleSort('epoch')"
          >
            @
            <span v-if="sortColumn === 'epoch'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
          <th
            class="sortable"
            data-testid="header-peak-acc"
            @click="handleSort('peak_accuracy')"
          >
            Peak
            <span v-if="sortColumn === 'peak_accuracy'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
          <th>Traj</th>
          <th
            class="sortable"
            data-testid="header-growth"
            @click="handleSort('growth_ratio')"
          >
            Grw
            <span v-if="sortColumn === 'growth_ratio'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
          <th
            class="sortable"
            data-testid="header-reward"
            @click="handleSort('cumulative_reward')"
          >
            EndRwd
            <span v-if="sortColumn === 'cumulative_reward'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
          <th>Seeds</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(run, index) in sortedRuns"
          :key="run.record_id"
          :class="{ 'keyboard-selected': index === selectedRowIndex }"
          :data-testid="`leaderboard-row-${run.record_id}`"
          @click="handleRowClick(run.record_id)"
        >
          <td :data-testid="`rank-${run.record_id}`">{{ index + 1 }}</td>
          <td :data-testid="`env-${run.record_id}`">{{ run.env_id }}</td>
          <td :data-testid="`episode-${run.record_id}`">{{ run.episode }}</td>
          <td :data-testid="`epoch-${run.record_id}`">{{ run.epoch }}</td>
          <td :data-testid="`peak-acc-${run.record_id}`">{{ formatAccuracy(run.peak_accuracy) }}</td>
          <td
            class="trajectory-cell"
            :class="trajectoryClass(run)"
            :data-testid="`trajectory-${run.record_id}`"
          >
            {{ trajectoryGlyph(run) }}{{ formatAccuracy(run.final_accuracy) }}
          </td>
          <td :data-testid="`growth-${run.record_id}`">{{ formatGrowthRatio(run.growth_ratio) }}</td>
          <td :data-testid="`reward-${run.record_id}`">{{ formatReward(run.cumulative_reward) }}</td>
          <td class="seed-counts" :data-testid="`seeds-${run.record_id}`">
            {{ formatSeedComposition(run) }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.leaderboard-table {
  width: 100%;
  overflow-x: auto;
}

.empty-state {
  padding: var(--space-lg);
  text-align: center;
  color: var(--text-secondary);
  font-style: italic;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

thead {
  background: var(--bg-elevated);
}

th {
  padding: var(--space-sm) var(--space-md);
  text-align: left;
  font-weight: 600;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-subtle);
  white-space: nowrap;
  user-select: none;
}

th.sortable {
  cursor: pointer;
}

th.sortable:hover {
  color: var(--glow-cyan);
}

.sort-indicator {
  margin-left: var(--space-xs);
  font-size: 10px;
  color: var(--glow-cyan);
}

.rank-col {
  width: 50px;
  text-align: center;
}

tbody tr {
  cursor: pointer;
  transition: background-color 0.15s ease;
}

tbody tr:nth-child(odd) {
  background: var(--bg-panel);
}

tbody tr:nth-child(even) {
  background: var(--bg-primary);
}

tbody tr:hover {
  background: var(--bg-elevated);
}

tbody tr.keyboard-selected {
  outline: 2px solid var(--glow-cyan);
  outline-offset: -2px;
  background: var(--bg-elevated);
}

tbody tr.keyboard-selected:nth-child(odd),
tbody tr.keyboard-selected:nth-child(even) {
  background: var(--bg-elevated);
}

td {
  padding: var(--space-sm) var(--space-md);
  color: var(--text-primary);
  border-bottom: 1px solid var(--border-subtle);
}

td:first-child {
  text-align: center;
  font-weight: 600;
  color: var(--text-secondary);
}

.trajectory-cell,
.seed-counts {
  font-variant-numeric: tabular-nums;
}

.trajectory-up {
  color: var(--status-win);
}

.trajectory-held,
.trajectory-soft-down {
  color: var(--text-secondary);
}

.trajectory-down {
  color: var(--status-warn);
}
</style>
