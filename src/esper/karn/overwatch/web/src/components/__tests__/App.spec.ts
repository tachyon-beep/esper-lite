// src/esper/karn/overwatch/web/src/components/__tests__/App.spec.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import App from '../../App.vue'
import type {
  SanctumSnapshot,
  EnvState,
  TamiyoState,
  SystemVitals,
  RewardComponents,
  EventLogEntry,
  BestRunRecord,
  CounterfactualSnapshot,
  RunConfig
} from '../../types/sanctum'

// Get the global object in a type-safe way
const globalObj = globalThis as typeof globalThis & { WebSocket: typeof WebSocket }

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  readyState = 0

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }

  close() {
    this.readyState = 3
  }

  simulateOpen() {
    this.readyState = 1
    this.onopen?.()
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) })
  }

  simulateClose() {
    this.readyState = 3
    this.onclose?.()
  }
}

// Factory functions for test data
function createRewardComponents(overrides: Partial<RewardComponents> = {}): RewardComponents {
  return {
    total: 0.5,
    base_acc_delta: 0.1,
    bounded_attribution: 0.2,
    seed_contribution: 0.1,
    compute_rent: -0.05,
    alpha_shock: 0,
    ratio_penalty: 0,
    stage_bonus: 0.05,
    fossilize_terminal_bonus: 0,
    blending_warning: 0,
    holding_warning: 0,
    env_id: 0,
    val_acc: 0.87,
    last_action: 'OBSERVE',
    ...overrides
  }
}

function createCounterfactualSnapshot(): CounterfactualSnapshot {
  return {
    slot_ids: [],
    configs: [],
    strategy: 'ablation',
    compute_time_ms: 10
  }
}

function createEnvState(overrides: Partial<EnvState> = {}): EnvState {
  return {
    env_id: 0,
    current_epoch: 100,
    host_accuracy: 0.873,
    host_loss: 0.45,
    host_params: 1000000,
    seeds: {},
    active_seed_count: 3,
    fossilized_count: 2,
    pruned_count: 1,
    fossilized_params: 50000,
    blueprint_spawns: {},
    blueprint_prunes: {},
    blueprint_fossilized: {},
    reward_components: createRewardComponents(),
    counterfactual_matrix: createCounterfactualSnapshot(),
    reward_history: [0.1, 0.2, 0.3],
    accuracy_history: [0.8, 0.85, 0.87],
    best_reward: 0.5,
    best_reward_epoch: 80,
    best_accuracy: 0.89,
    best_accuracy_epoch: 90,
    best_accuracy_episode: 5,
    best_seeds: {},
    action_history: ['OBSERVE', 'GERMINATE', 'OBSERVE'],
    action_counts: { OBSERVE: 50, GERMINATE: 10 },
    total_actions: 60,
    status: 'healthy',
    last_update: '2024-01-01T00:00:00Z',
    epochs_since_improvement: 5,
    stall_counter: 0,
    degraded_counter: 0,
    reward_mode: 'standard',
    ...overrides
  }
}

function createTamiyoState(overrides: Partial<TamiyoState> = {}): TamiyoState {
  return {
    entropy: 0.5,
    clip_fraction: 0.1,
    kl_divergence: 0.01,
    explained_variance: 0.7,
    policy_loss: 0.05,
    value_loss: 0.1,
    entropy_loss: -0.02,
    grad_norm: 1.5,
    learning_rate: 0.0003,
    entropy_coef: 0.01,
    ratio_mean: 1.0,
    ratio_min: 0.8,
    ratio_max: 1.2,
    ratio_std: 0.1,
    advantage_mean: 0.0,
    advantage_std: 1.0,
    advantage_min: -2.0,
    advantage_max: 2.0,
    advantage_raw_mean: 0.0,
    advantage_raw_std: 1.0,
    dead_layers: 0,
    exploding_layers: 0,
    nan_grad_count: 0,
    layer_gradient_health: { 'policy.0': 0.95, 'policy.1': 0.88, 'value.0': 0.92 },
    entropy_collapsed: false,
    update_time_ms: 150,
    early_stop_epoch: null,
    head_slot_entropy: 0.5,
    head_blueprint_entropy: 0.6,
    head_style_entropy: 0.4,
    head_tempo_entropy: 0.55,
    head_alpha_target_entropy: 0.45,
    head_alpha_speed_entropy: 0.5,
    head_alpha_curve_entropy: 0.48,
    head_op_entropy: 0.52,
    head_slot_grad_norm: 0.8,
    head_blueprint_grad_norm: 0.9,
    head_style_grad_norm: 0.7,
    head_tempo_grad_norm: 0.85,
    head_alpha_target_grad_norm: 0.75,
    head_alpha_speed_grad_norm: 0.8,
    head_alpha_curve_grad_norm: 0.78,
    head_op_grad_norm: 0.82,
    episode_return_history: [0.1, 0.2, 0.3],
    current_episode_return: 0.3,
    current_episode: 5,
    policy_loss_history: [0.1, 0.08, 0.05],
    value_loss_history: [0.2, 0.15, 0.1],
    grad_norm_history: [2.0, 1.8, 1.5],
    entropy_history: [0.6, 0.55, 0.5],
    explained_variance_history: [0.5, 0.6, 0.7],
    kl_divergence_history: [0.02, 0.015, 0.01],
    clip_fraction_history: [0.15, 0.12, 0.1],
    inner_epoch: 3,
    ppo_batch: 10,
    action_counts: { OBSERVE: 50, GERMINATE: 10 },
    total_actions: 60,
    ppo_data_received: true,
    recent_decisions: [],
    group_id: null,
    ...overrides
  }
}

function createSystemVitals(overrides: Partial<SystemVitals> = {}): SystemVitals {
  return {
    gpu_stats: {},
    gpu_memory_used_gb: 8.0,
    gpu_memory_total_gb: 24.0,
    gpu_utilization: 85,
    gpu_temperature: 65,
    cpu_percent: 45,
    ram_used_gb: 16.0,
    ram_total_gb: 64.0,
    epochs_per_second: 2.5,
    batches_per_hour: 100,
    host_params: 1000000,
    ...overrides
  }
}

function createEventLogEntry(overrides: Partial<EventLogEntry> = {}): EventLogEntry {
  return {
    timestamp: '2024-01-01T00:00:00Z',
    event_type: 'seed_germinated',
    env_id: 0,
    message: 'Seed germinated in slot_0',
    episode: 5,
    relative_time: '2m ago',
    metadata: {},
    ...overrides
  }
}

	function createBestRunRecord(overrides: Partial<BestRunRecord> = {}): BestRunRecord {
	  return {
	    env_id: 0,
	    episode: 5,
	    peak_accuracy: 0.92,
	    final_accuracy: 0.90,
	    epoch: 100,
	    seeds: {},
	    slot_ids: ['slot_0', 'slot_1'],
	    growth_ratio: 1.05,
	    record_id: 'record-001',
	    reward_components: createRewardComponents(),
	    counterfactual_matrix: createCounterfactualSnapshot(),
	    action_history: ['OBSERVE', 'GERMINATE'],
	    reward_history: [0.1, 0.2],
	    accuracy_history: [0.85, 0.90],
    host_loss: 0.3,
    host_params: 1000000,
    fossilized_count: 2,
    pruned_count: 1,
    reward_mode: 'standard',
    blueprint_spawns: {},
    blueprint_fossilized: {},
    blueprint_prunes: {},
    ...overrides
  }
}

function createRunConfig(): RunConfig {
  return {
    seed: 42,
    n_episodes: 100,
    lr: 0.0003,
    clip_ratio: 0.2,
    entropy_coef: 0.01,
    param_budget: 1000000,
    resume_path: '',
    entropy_anneal: {}
  }
}

function createSnapshot(overrides: Partial<SanctumSnapshot> = {}): SanctumSnapshot {
  return {
    envs: {
      0: createEnvState({ env_id: 0 }),
      1: createEnvState({ env_id: 1, status: 'stalled' })
    },
    tamiyo: createTamiyoState(),
    vitals: createSystemVitals(),
    rewards: createRewardComponents(),
    slot_ids: ['slot_0', 'slot_1', 'slot_2', 'slot_3'],
    current_episode: 5,
    current_batch: 10,
    max_batches: 100,
    current_epoch: 100,
    max_epochs: 1000,
    run_id: 'run-001',
    task_name: 'test-task',
    run_config: createRunConfig(),
    start_time: '2024-01-01T00:00:00Z',
    connected: true,
    runtime_seconds: 3600,
    staleness_seconds: 0.5,
    captured_at: '2024-01-01T01:00:00Z',
    total_events_received: 100,
    poll_count: 50,
    training_thread_alive: true,
    aggregate_mean_accuracy: 0.85,
    aggregate_mean_reward: 0.4,
    batch_avg_reward: 0.45,
    batch_total_episodes: 8,
    mean_accuracy_history: [0.8, 0.82, 0.85],
    event_log: [
      createEventLogEntry({ event_type: 'seed_germinated', message: 'Seed germinated' }),
      createEventLogEntry({ event_type: 'anomaly_detected', message: 'Gradient explosion detected' })
    ],
    best_runs: [
      createBestRunRecord({ record_id: 'record-001', peak_accuracy: 0.92 }),
      createBestRunRecord({ record_id: 'record-002', peak_accuracy: 0.88 })
    ],
    cumulative_fossilized: 10,
    cumulative_pruned: 5,
    cumulative_blueprint_spawns: {},
    cumulative_blueprint_fossilized: {},
    cumulative_blueprint_prunes: {},
    slot_stage_counts: { DORMANT: 1, TRAINING: 2, FOSSILIZED: 1 },
    total_slots: 4,
    active_slots: 3,
    avg_epochs_in_stage: 25,
    last_ppo_update: '2024-01-01T00:59:00Z',
    last_reward_update: '2024-01-01T00:59:30Z',
    focused_env_id: 0,
    ...overrides
  }
}

describe('App.vue Integration', () => {
  let originalWebSocket: typeof WebSocket

  beforeEach(() => {
    originalWebSocket = globalObj.WebSocket
    globalObj.WebSocket = MockWebSocket as unknown as typeof WebSocket
    MockWebSocket.instances = []
    vi.useFakeTimers()
  })

  afterEach(() => {
    globalObj.WebSocket = originalWebSocket
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  describe('Layout Structure', () => {
    it('renders three-column layout with data-testid attributes', async () => {
      const wrapper = mount(App)

      // Simulate WebSocket connection
      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: createSnapshot() })
      await flushPromises()

      expect(wrapper.find('[data-testid="left-sidebar"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="main-content"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="right-panel"]').exists()).toBe(true)
    })

    it('renders StatusBar at top spanning full width', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: createSnapshot() })
      await flushPromises()

      expect(wrapper.find('[data-testid="status-bar"]').exists()).toBe(true)
    })
  })

  describe('Loading States', () => {
    it('shows connecting state initially', () => {
      const wrapper = mount(App)

      expect(wrapper.find('[data-testid="loading-state"]').exists()).toBe(true)
      expect(wrapper.text()).toContain('Connecting')
    })

    it('shows loading state when connected but no snapshot yet', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      await flushPromises()

      expect(wrapper.find('[data-testid="loading-state"]').exists()).toBe(true)
      expect(wrapper.text()).toContain('Loading')
    })

    it('hides loading state when snapshot is received', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: createSnapshot() })
      await flushPromises()

      expect(wrapper.find('[data-testid="loading-state"]').exists()).toBe(false)
    })
  })

  describe('StatusBar Props', () => {
    it('passes connection state to StatusBar', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: createSnapshot() })
      await flushPromises()

      const statusBar = wrapper.findComponent({ name: 'StatusBar' })
      expect(statusBar.exists()).toBe(true)
      expect(statusBar.props('connectionState')).toBe('connected')
    })

    it('passes episode, epoch, batch from snapshot to StatusBar', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot({
        current_episode: 10,
        current_epoch: 200,
        current_batch: 25
      })

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const statusBar = wrapper.findComponent({ name: 'StatusBar' })
      expect(statusBar.props('episode')).toBe(10)
      expect(statusBar.props('epoch')).toBe(200)
      expect(statusBar.props('batch')).toBe(25)
    })
  })

  describe('EnvironmentGrid Props', () => {
    it('passes envs and focusedEnvId to EnvironmentGrid', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot({ focused_env_id: 1 })

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const grid = wrapper.findComponent({ name: 'EnvironmentGrid' })
      expect(grid.exists()).toBe(true)
      expect(grid.props('envs')).toEqual(snapshot.envs)
      expect(grid.props('focusedEnvId')).toBe(1)
    })
  })

  describe('AnomalySidebar Props', () => {
    it('passes event_log to AnomalySidebar', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const sidebar = wrapper.findComponent({ name: 'AnomalySidebar' })
      expect(sidebar.exists()).toBe(true)
      expect(sidebar.props('events')).toEqual(snapshot.event_log)
    })
  })

  describe('LeaderboardTable Props', () => {
    it('passes best_runs to LeaderboardTable', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const leaderboard = wrapper.findComponent({ name: 'LeaderboardTable' })
      expect(leaderboard.exists()).toBe(true)
      expect(leaderboard.props('runs')).toEqual(snapshot.best_runs)
    })
  })

  describe('HealthGauges Props', () => {
    it('passes vitals and tamiyo to HealthGauges', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const gauges = wrapper.findComponent({ name: 'HealthGauges' })
      expect(gauges.exists()).toBe(true)
      expect(gauges.props('vitals')).toEqual(snapshot.vitals)
      expect(gauges.props('tamiyo')).toEqual(snapshot.tamiyo)
    })
  })

  describe('PolicyDiagnostics Props', () => {
    it('passes tamiyo to PolicyDiagnostics', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const diagnostics = wrapper.findComponent({ name: 'PolicyDiagnostics' })
      expect(diagnostics.exists()).toBe(true)
      expect(diagnostics.props('tamiyo')).toEqual(snapshot.tamiyo)
    })
  })

  describe('GradientHeatmap Props', () => {
    it('passes tamiyo to GradientHeatmap', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const heatmap = wrapper.findComponent({ name: 'GradientHeatmap' })
      expect(heatmap.exists()).toBe(true)
      expect(heatmap.props('tamiyo')).toEqual(snapshot.tamiyo)
    })
  })

  describe('EventTimeline Props', () => {
    it('passes event_log to EventTimeline', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const timeline = wrapper.findComponent({ name: 'EventTimeline' })
      expect(timeline.exists()).toBe(true)
      expect(timeline.props('events')).toEqual(snapshot.event_log)
    })
  })

  describe('SeedSwimlane Props', () => {
    it('passes seeds, slotIds, and currentEpoch to SeedSwimlane', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const swimlane = wrapper.findComponent({ name: 'SeedSwimlane' })
      expect(swimlane.exists()).toBe(true)
      // SeedSwimlane uses seeds from focused env
      expect(swimlane.props('slotIds')).toEqual(snapshot.slot_ids)
      expect(swimlane.props('currentEpoch')).toBe(snapshot.current_epoch)
    })
  })

  describe('ContributionWaterfall Props', () => {
    it('passes rewards to ContributionWaterfall', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const waterfall = wrapper.findComponent({ name: 'ContributionWaterfall' })
      expect(waterfall.exists()).toBe(true)
      expect(waterfall.props('rewards')).toEqual(snapshot.rewards)
    })
  })

  describe('Environment Selection', () => {
    it('updates focused env when EnvironmentGrid emits select', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot({ focused_env_id: 0 })

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: snapshot })
      await flushPromises()

      const grid = wrapper.findComponent({ name: 'EnvironmentGrid' })
      await grid.vm.$emit('select', 1)
      await flushPromises()

      // Focused env should update locally
      expect(grid.props('focusedEnvId')).toBe(1)
    })
  })

  describe('Disconnection Handling', () => {
    it('shows disconnected state when WebSocket closes', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({ type: 'snapshot', data: createSnapshot() })
      await flushPromises()

      // Verify connected
      const statusBar = wrapper.findComponent({ name: 'StatusBar' })
      expect(statusBar.props('connectionState')).toBe('connected')

      // Simulate disconnect
      ws.simulateClose()
      await flushPromises()

      expect(statusBar.props('connectionState')).toBe('disconnected')
    })
  })
})
