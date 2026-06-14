<!-- src/esper/karn/overwatch/web/src/components/MorphologyCausalLogPanel.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { MorphologyCausalLogEntry, SanctumSnapshot } from '../types/sanctum'

const props = withDefaults(
  defineProps<{
    snapshot: SanctumSnapshot
    maxHeight?: string
  }>(),
  {
    maxHeight: '400px'
  }
)

// Phase -> icon/color mapping. Matches MorphologyCausalLogPhase literal.
const PHASE_CONFIG: Record<string, { icon: string; colorClass: string }> = {
  proposal: { icon: '📝', colorClass: 'phase-proposal' },
  verdict: { icon: '⚖️', colorClass: 'phase-verdict' },
  mutation: { icon: '🧬', colorClass: 'phase-mutation' },
  dispatch: { icon: '🚀', colorClass: 'phase-dispatch' },
  commit: { icon: '✅', colorClass: 'phase-commit' },
  rollback: { icon: '↩️', colorClass: 'phase-rollback' },
  fossilization: { icon: '🪨', colorClass: 'phase-fossilization' },
  cooldown: { icon: '❄️', colorClass: 'phase-cooldown' },
  audit: { icon: '🔍', colorClass: 'phase-audit' }
}

const DEFAULT_CONFIG = { icon: '🔗', colorClass: 'phase-default' }

const entries = computed<MorphologyCausalLogEntry[]>(
  () => props.snapshot.morphology_causal_log
)

const entryCount = computed(() => entries.value.length)

const getPhaseConfig = (phase: string) => {
  return PHASE_CONFIG[phase] ?? DEFAULT_CONFIG
}

const shortId = (id: string | null): string => {
  if (id === null || id === '') {
    return '—'
  }
  return id.length > 12 ? `${id.slice(0, 12)}…` : id
}

const formatEvidence = (value: number | null): string => {
  if (value === null) {
    return '—'
  }
  return value.toFixed(3)
}

const containerStyle = computed(() => ({
  maxHeight: props.maxHeight
}))
</script>

<template>
  <div class="causal-log" data-testid="morphology-causal-log">
    <header class="causal-header" data-testid="causal-header">
      Causal Log ({{ entryCount }})
    </header>

    <div
      class="causal-container causal-scroll"
      :style="containerStyle"
      data-testid="causal-container"
    >
      <div
        v-if="entryCount === 0"
        class="empty-state"
        data-testid="empty-state"
      >
        No causal-log entries yet
      </div>

      <ul v-else class="causal-list" data-testid="causal-list">
        <li
          v-for="(entry, index) in entries"
          :key="`${entry.action_id}-${entry.phase}-${index}`"
          class="causal-item"
          data-testid="causal-item"
        >
          <span
            class="phase-icon"
            :class="getPhaseConfig(entry.phase).colorClass"
            data-testid="phase-icon"
          >
            {{ getPhaseConfig(entry.phase).icon }}
          </span>

          <div class="causal-card">
            <div class="causal-row causal-top">
              <span
                class="phase-badge"
                :class="getPhaseConfig(entry.phase).colorClass"
                data-testid="phase-badge"
              >
                {{ entry.phase }}
              </span>
              <span class="operation" data-testid="operation">
                {{ entry.operation }}
              </span>
              <span class="env-chip" data-testid="env-chip">
                Env {{ entry.env_id }}
              </span>
              <span class="slot-chip" data-testid="slot-chip">
                {{ entry.slot_id }}
              </span>
            </div>

            <div class="causal-row id-chain" data-testid="id-chain">
              <span class="id-pair" data-testid="action-id">
                <span class="id-label">action</span>
                <span class="id-value">{{ shortId(entry.action_id) }}</span>
              </span>
              <span class="id-pair" data-testid="proposal-id">
                <span class="id-label">proposal</span>
                <span class="id-value">{{ shortId(entry.proposal_id) }}</span>
              </span>
              <span class="id-pair" data-testid="verdict-id">
                <span class="id-label">verdict</span>
                <span class="id-value">{{ shortId(entry.verdict_id) }}</span>
              </span>
              <span class="id-pair" data-testid="mutation-id">
                <span class="id-label">mutation</span>
                <span class="id-value">{{ shortId(entry.mutation_id) }}</span>
              </span>
            </div>

            <div class="causal-row meta-row">
              <span class="meta-pair" data-testid="watch-evidence">
                <span class="meta-label">watch</span>
                <span class="meta-value">{{ formatEvidence(entry.watch_window_evidence) }}</span>
              </span>
              <span class="meta-pair" data-testid="rng-identity">
                <span class="meta-label">rng</span>
                <span class="meta-value">{{ entry.rng_stream }}#{{ entry.rng_seed }}</span>
              </span>
              <span
                v-if="entry.linked_event_id !== null"
                class="meta-pair linked"
                data-testid="linked-event"
              >
                <span class="meta-label">linked</span>
                <span class="meta-value">{{ shortId(entry.linked_event_id) }}</span>
              </span>
            </div>

            <div
              v-if="entry.governor_reason !== null || entry.governor_approved !== null"
              class="causal-row governor-row"
              data-testid="governor-row"
            >
              <span
                v-if="entry.governor_approved !== null"
                class="governor-verdict"
                :class="entry.governor_approved ? 'approved' : 'blocked'"
                data-testid="governor-verdict"
              >
                {{ entry.governor_approved ? 'approved' : 'blocked' }}
              </span>
              <span
                v-if="entry.governor_reason !== null"
                class="governor-reason"
                data-testid="governor-reason"
              >
                {{ entry.governor_reason }}
              </span>
            </div>
          </div>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.causal-log {
  display: flex;
  flex-direction: column;
  background: var(--bg-panel);
  border-radius: 8px;
  border: 1px solid var(--border-subtle);
}

.causal-header {
  padding: var(--space-md);
  font-family: var(--font-display);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-subtle);
}

.causal-container {
  position: relative;
  padding: var(--space-md);
}

.causal-scroll {
  overflow-y: auto;
}

.empty-state {
  padding: var(--space-lg);
  text-align: center;
  color: var(--text-dim);
  font-size: 12px;
}

.causal-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.causal-item {
  display: flex;
  align-items: flex-start;
  gap: var(--space-md);
  margin-bottom: var(--space-md);
}

.causal-item:last-child {
  margin-bottom: 0;
}

.phase-icon {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  border-radius: 50%;
  background: var(--bg-elevated);
  border: 2px solid var(--border-subtle);
}

.causal-card {
  flex: 1;
  min-width: 0;
  background: var(--bg-elevated);
  border-radius: 6px;
  padding: var(--space-sm) var(--space-md);
  border: 1px solid var(--border-subtle);
  display: flex;
  flex-direction: column;
  gap: var(--space-xs);
}

.causal-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-sm);
}

.phase-badge {
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 600;
  text-transform: lowercase;
  letter-spacing: 0.5px;
  background: var(--bg-panel);
  color: var(--text-secondary);
}

.operation {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.env-chip,
.slot-chip {
  font-size: 9px;
  padding: 1px 6px;
  border-radius: 2px;
  background: var(--glow-cyan-dim);
  color: var(--glow-cyan);
  font-weight: 600;
}

.id-chain,
.meta-row {
  font-family: var(--font-mono);
  font-size: 10px;
}

.id-pair,
.meta-pair {
  display: inline-flex;
  gap: 4px;
  align-items: baseline;
}

.id-label,
.meta-label {
  color: var(--text-dim);
}

.id-value,
.meta-value {
  color: var(--text-primary);
}

.governor-row {
  font-size: 10px;
}

.governor-verdict {
  font-size: 9px;
  padding: 1px 6px;
  border-radius: 2px;
  font-weight: 600;
  text-transform: lowercase;
}

.governor-verdict.approved {
  background: rgba(0, 255, 157, 0.15);
  color: var(--status-win);
}

.governor-verdict.blocked {
  background: rgba(255, 92, 92, 0.15);
  color: var(--status-loss);
}

.governor-reason {
  color: var(--text-secondary);
}
</style>
