<!-- src/esper/karn/overwatch/web/src/components/LeaderboardTable.vue -->
<script setup lang="ts">
import { ref, computed } from 'vue'
import type { BestRunRecord } from '../types/sanctum'

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

type SortColumn = 'env_id' | 'episode' | 'peak_accuracy' | 'fossilized_count' | 'pruned_count'

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
            Episode
            <span v-if="sortColumn === 'episode'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
          <th
            class="sortable"
            data-testid="header-peak-acc"
            @click="handleSort('peak_accuracy')"
          >
            Peak Acc
            <span v-if="sortColumn === 'peak_accuracy'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
          <th
            class="sortable"
            data-testid="header-fossilized"
            @click="handleSort('fossilized_count')"
          >
            Fossilized
            <span v-if="sortColumn === 'fossilized_count'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
          <th
            class="sortable"
            data-testid="header-pruned"
            @click="handleSort('pruned_count')"
          >
            Pruned
            <span v-if="sortColumn === 'pruned_count'" class="sort-indicator">
              {{ sortAscending ? '\u25B2' : '\u25BC' }}
            </span>
          </th>
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
          <td :data-testid="`peak-acc-${run.record_id}`">{{ formatAccuracy(run.peak_accuracy) }}</td>
          <td :data-testid="`fossilized-${run.record_id}`">{{ run.fossilized_count }}</td>
          <td :data-testid="`pruned-${run.record_id}`">{{ run.pruned_count }}</td>
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
</style>
