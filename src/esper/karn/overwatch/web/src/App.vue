<!-- src/esper/karn/overwatch/web/src/App.vue -->
<script setup lang="ts">
import { ref } from 'vue'

const connectionState = ref<'connecting' | 'connected' | 'disconnected'>('connecting')
</script>

<template>
  <div class="overwatch">
    <header class="status-bar">
      <span class="status-indicator" :class="connectionState">
        {{ connectionState.toUpperCase() }}
      </span>
      <span class="title">OVERWATCH</span>
    </header>

    <main class="dashboard">
      <aside class="anomaly-sidebar">
        <!-- Anomaly sidebar -->
        <p>Anomalies</p>
      </aside>

      <section class="main-content">
        <!-- Main dashboard content -->
        <p>Dashboard loading...</p>
      </section>

      <aside class="detail-panel">
        <!-- Detail panel -->
        <p>Details</p>
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

.status-bar {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-panel);
  border-bottom: 1px solid var(--border-subtle);
}

.status-indicator {
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 2px;
}

.status-indicator.connecting {
  background: var(--status-warn);
  color: var(--bg-void);
}

.status-indicator.connected {
  background: var(--status-win);
  color: var(--bg-void);
}

.status-indicator.disconnected {
  background: var(--status-loss);
  color: var(--bg-void);
}

.title {
  font-family: var(--font-display);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 3px;
  color: var(--glow-cyan);
}

.dashboard {
  flex: 1;
  display: grid;
  grid-template-columns: 200px 1fr 300px;
  gap: var(--space-md);
  padding: var(--space-md);
}

.anomaly-sidebar,
.detail-panel {
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: var(--space-md);
}

.main-content {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}
</style>
