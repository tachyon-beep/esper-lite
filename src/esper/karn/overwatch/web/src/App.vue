<!-- src/esper/karn/overwatch/web/src/App.vue -->
<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useOverwatch } from './composables/useOverwatch'
import StatusBar from './components/StatusBar.vue'
import EnvironmentGrid from './components/EnvironmentGrid.vue'
import AnomalySidebar from './components/AnomalySidebar.vue'
import LeaderboardTable from './components/LeaderboardTable.vue'
import HealthGauges from './components/HealthGauges.vue'
import PolicyDiagnostics from './components/PolicyDiagnostics.vue'
import GradientHeatmap from './components/GradientHeatmap.vue'
import EventTimeline from './components/EventTimeline.vue'
import SeedSwimlane from './components/SeedSwimlane.vue'
import ContributionWaterfall from './components/ContributionWaterfall.vue'

// WebSocket URL - can be configured via environment variable
const wsUrl = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8765'

// Initialize the overwatch composable
const { snapshot, connectionState, staleness } = useOverwatch(wsUrl)

// Local focused env ID (can be overridden by user selection)
const localFocusedEnvId = ref<number | null>(null)

// Use local selection if set, otherwise use server's focused_env_id
const focusedEnvId = computed(() => {
  if (localFocusedEnvId.value !== null) {
    return localFocusedEnvId.value
  }
  return snapshot.value?.focused_env_id ?? 0
})

// Handle environment selection from EnvironmentGrid
function handleEnvSelect(envId: number) {
  localFocusedEnvId.value = envId
}

// Reset local selection when snapshot's focused env changes
watch(
  () => snapshot.value?.focused_env_id,
  (newFocused) => {
    if (newFocused !== undefined && localFocusedEnvId.value === null) {
      // Don't override user selection
    }
  }
)

// Computed props for StatusBar
const statusBarProps = computed(() => ({
  connectionState: connectionState.value,
  staleness: staleness.value,
  episode: snapshot.value?.current_episode ?? 0,
  epoch: snapshot.value?.current_epoch ?? 0,
  batch: snapshot.value?.current_batch ?? 0
}))

// Get seeds from focused environment for SeedSwimlane
const focusedEnvSeeds = computed(() => {
  if (!snapshot.value) return {}
  const env = snapshot.value.envs[focusedEnvId.value]
  return env?.seeds ?? {}
})

// Loading state text
const loadingText = computed(() => {
  if (connectionState.value === 'connecting') return 'Connecting to Overwatch...'
  if (connectionState.value === 'disconnected') return 'Disconnected. Reconnecting...'
  if (!snapshot.value) return 'Loading dashboard data...'
  return ''
})

const isLoading = computed(() => {
  return connectionState.value === 'connecting' || !snapshot.value
})
</script>

<template>
  <div class="overwatch">
    <!-- Status Bar - Full Width Header -->
    <StatusBar
      v-if="snapshot"
      data-testid="status-bar"
      :connection-state="statusBarProps.connectionState"
      :staleness="statusBarProps.staleness"
      :episode="statusBarProps.episode"
      :epoch="statusBarProps.epoch"
      :batch="statusBarProps.batch"
    />

    <!-- Loading State -->
    <div
      v-if="isLoading"
      class="loading-state"
      data-testid="loading-state"
    >
      <div class="loading-spinner"></div>
      <span class="loading-text">{{ loadingText }}</span>
    </div>

    <!-- Main Dashboard Layout -->
    <main v-else class="dashboard">
      <!-- Left Sidebar: Anomalies + Event Timeline -->
      <aside class="left-sidebar" data-testid="left-sidebar">
        <AnomalySidebar
          :events="snapshot!.event_log"
        />
        <EventTimeline
          :events="snapshot!.event_log"
          max-height="300px"
        />
      </aside>

      <!-- Main Content: Environment Grid + Health Gauges + Leaderboard -->
      <section class="main-content" data-testid="main-content">
        <HealthGauges
          :vitals="snapshot!.vitals"
          :tamiyo="snapshot!.tamiyo"
        />
        <EnvironmentGrid
          :envs="snapshot!.envs"
          :focused-env-id="focusedEnvId"
          @select="handleEnvSelect"
        />
        <LeaderboardTable
          :runs="snapshot!.best_runs"
        />
      </section>

      <!-- Right Panel: Policy Diagnostics + Gradient Heatmap + Seed Swimlane + Contribution Waterfall -->
      <aside class="right-panel" data-testid="right-panel">
        <div class="panel-section">
          <h3 class="panel-title">Policy Diagnostics</h3>
          <PolicyDiagnostics
            :tamiyo="snapshot!.tamiyo"
          />
        </div>

        <div class="panel-section">
          <h3 class="panel-title">Gradient Health</h3>
          <GradientHeatmap
            :tamiyo="snapshot!.tamiyo"
            :compact="true"
          />
        </div>

        <div class="panel-section">
          <h3 class="panel-title">Seed Timeline</h3>
          <SeedSwimlane
            :seeds="focusedEnvSeeds"
            :slot-ids="snapshot!.slot_ids"
            :current-epoch="snapshot!.current_epoch"
          />
        </div>

        <div class="panel-section">
          <h3 class="panel-title">Reward Breakdown</h3>
          <ContributionWaterfall
            :rewards="snapshot!.rewards"
          />
        </div>
      </aside>
    </main>
  </div>
</template>

<style scoped>
.overwatch {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--bg-void);
  color: var(--text-primary);
}

/* Loading State */
.loading-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--space-md);
  color: var(--text-secondary);
}

.loading-spinner {
  width: 48px;
  height: 48px;
  border: 3px solid var(--bg-elevated);
  border-top-color: var(--glow-cyan);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-text {
  font-size: 14px;
  letter-spacing: 1px;
}

/* Dashboard Layout */
.dashboard {
  flex: 1;
  display: grid;
  grid-template-columns: 250px 1fr 350px;
  gap: var(--space-md);
  padding: var(--space-md);
  min-height: 0;
  overflow: hidden;
}

/* Left Sidebar */
.left-sidebar {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  overflow-y: auto;
  min-height: 0;
}

/* Main Content Area */
.main-content {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  overflow-y: auto;
  min-height: 0;
}

/* Right Panel */
.right-panel {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  overflow-y: auto;
  min-height: 0;
}

.panel-section {
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  overflow: hidden;
}

.panel-title {
  font-family: var(--font-display);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text-secondary);
  padding: var(--space-sm) var(--space-md);
  border-bottom: 1px solid var(--border-subtle);
  margin: 0;
}

/* Responsive adjustments */
@media (max-width: 1400px) {
  .dashboard {
    grid-template-columns: 220px 1fr 300px;
  }
}

@media (max-width: 1200px) {
  .dashboard {
    grid-template-columns: 200px 1fr 280px;
  }
}

@media (max-width: 1024px) {
  .dashboard {
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr auto;
  }

  .left-sidebar {
    flex-direction: row;
    overflow-x: auto;
  }

  .right-panel {
    flex-direction: row;
    flex-wrap: wrap;
  }

  .panel-section {
    flex: 1;
    min-width: 280px;
  }
}
</style>
