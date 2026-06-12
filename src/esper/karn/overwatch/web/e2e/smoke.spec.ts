// src/esper/karn/overwatch/web/e2e/smoke.spec.ts
/**
 * E2E Smoke Tests for Overwatch Dashboard
 *
 * These tests verify critical user flows work correctly in a real browser.
 * WebSocket connections are mocked to allow testing without a backend.
 */
import { test, expect } from '@playwright/test'
import type {
  DecisionSnapshot,
  RewardComponents,
  SanctumSnapshot,
  SeedState
} from '../src/types/sanctum'

function createMockSeedState(overrides: Partial<SeedState> = {}): SeedState {
  return {
    slot_id: 'slot_0',
    stage: 'TRAINING',
    blueprint_id: 'resnet_block',
    alpha: 0.5,
    accuracy_delta: 0.02,
    seed_params: 50000,
    grad_ratio: 1.0,
    has_vanishing: false,
    has_exploding: false,
    epochs_in_stage: 10,
    improvement: 0.015,
    prune_reason: '',
    auto_pruned: false,
    epochs_total: 42,
    counterfactual: 0.8,
    blend_tempo_epochs: 20,
    alpha_curve: 'LINEAR',
    contribution_velocity: 0.01,
    interaction_sum: 0.02,
    boost_received: 0.01,
    upstream_alpha_sum: 0,
    downstream_alpha_sum: 0,
    ...overrides
  }
}

function createMockRewardComponents(
  overrides: Partial<RewardComponents> = {}
): RewardComponents {
  return {
    total: 0.5,
    base_acc_delta: 0.02,
    bounded_attribution: 0.1,
    seed_contribution: 0.15,
    escrow_credit_prev: 0.02,
    escrow_credit_target: 0.03,
    escrow_delta: 0.01,
    escrow_credit_next: 0.03,
    escrow_forfeit: 0,
    compute_rent: -0.01,
    alpha_shock: 0,
    ratio_penalty: 0,
    stage_bonus: 0.1,
    fossilize_terminal_bonus: 0,
    blending_warning: 0,
    holding_warning: 0,
    hindsight_credit: 0,
    scaffold_count: 0,
    avg_scaffold_delay: 0,
    env_id: 0,
    val_acc: 0.85,
    stable_val_acc: 0.84,
    last_action: 'SPAWN',
    ...overrides
  }
}

function createMockDecisionSnapshot(
  overrides: Partial<DecisionSnapshot> = {}
): DecisionSnapshot {
  return {
    timestamp: new Date().toISOString(),
    slot_states: { slot_0: 'TRAINING' },
    host_accuracy: 0.85,
    chosen_action: 'WAIT',
    chosen_slot: null,
    confidence: 0.81,
    expected_value: 0.2,
    actual_reward: 0.1,
    alternatives: [['WAIT', 0.2], ['GERMINATE', 0.12]],
    action_success: true,
    decision_id: 'decision-001',
    env_id: 0,
    episode: 1,
    epoch: 42,
    batch: 5,
    value_residual: 0.04,
    td_advantage: 0.04,
    decision_entropy: 0.5,
    chosen_blueprint: null,
    chosen_tempo: null,
    chosen_style: null,
    chosen_curve: null,
    chosen_alpha_target: null,
    chosen_alpha_speed: null,
    op_confidence: 0.81,
    slot_confidence: 0,
    blueprint_confidence: 0,
    style_confidence: 0,
    tempo_confidence: 0,
    alpha_target_confidence: 0,
    alpha_speed_confidence: 0,
    curve_confidence: 0,
    op_entropy: 0.5,
    slot_entropy: 0,
    blueprint_entropy: 0,
    style_entropy: 0,
    tempo_entropy: 0,
    alpha_target_entropy: 0,
    alpha_speed_entropy: 0,
    curve_entropy: 0,
    ...overrides
  }
}

/**
 * Creates a minimal valid SanctumSnapshot for testing.
 * Contains just enough data to render the dashboard.
 */
function createMockSnapshot(): SanctumSnapshot {
  return {
    envs: {
      0: {
        env_id: 0,
        current_epoch: 42,
        host_accuracy: 0.85,
        host_loss: 0.23,
        host_params: 1000000,
        seeds: {
          'slot_0': createMockSeedState()
        },
        active_seed_count: 1,
        fossilized_count: 0,
        pruned_count: 0,
        fossilized_params: 0,
        blueprint_spawns: { 'resnet_block': 1 },
        blueprint_prunes: {},
        blueprint_fossilized: {},
        reward_components: createMockRewardComponents(),
        counterfactual_matrix: {
          slot_ids: ['slot_0'],
          configs: [{ seed_mask: [true], accuracy: 0.85 }],
          strategy: 'factorial',
          compute_time_ms: 10
        },
        reward_history: [0.1, 0.2, 0.3, 0.4, 0.5],
        accuracy_history: [0.7, 0.75, 0.8, 0.82, 0.85],
        best_reward: 0.5,
        best_reward_epoch: 42,
        best_accuracy: 0.85,
        best_accuracy_epoch: 42,
        best_accuracy_episode: 1,
        best_seeds: {},
        action_history: ['SPAWN', 'WAIT', 'WAIT'],
        action_counts: { 'SPAWN': 1, 'WAIT': 2 },
        total_actions: 3,
        status: 'RUNNING',
        last_update: new Date().toISOString(),
        epochs_since_improvement: 0,
        stall_counter: 0,
        degraded_counter: 0,
        reward_mode: 'STANDARD'
      }
    },
    tamiyo: {
      entropy: 1.5,
      clip_fraction: 0.1,
      kl_divergence: 0.01,
      explained_variance: 0.85,
      policy_loss: -0.02,
      value_loss: 0.1,
      entropy_loss: -0.01,
      grad_norm: 0.5,
      learning_rate: 0.0003,
      entropy_coef: 0.01,
      ratio_mean: 1.0,
      ratio_min: 0.9,
      ratio_max: 1.1,
      ratio_std: 0.05,
      advantage_mean: 0.0,
      advantage_std: 1.0,
      advantage_skewness: 0.0,
      advantage_kurtosis: 0.0,
      advantage_min: -2.0,
      advantage_max: 2.0,
      advantage_raw_mean: 0.5,
      advantage_raw_std: 1.2,
      advantage_positive_ratio: 0.5,
      log_prob_min: -1.0,
      log_prob_max: -0.1,
      decision_density: 0.9,
      forced_step_ratio: 0.1,
      advantage_std_floored: false,
      pre_norm_advantage_std: 1.2,
      decision_density_history: [0.9],
      dead_layers: 0,
      exploding_layers: 0,
      nan_grad_count: 0,
      inf_grad_count: 0,
      head_nan_latch: {},
      head_inf_latch: {},
      layer_gradient_health: { 'layer_0': 1.0, 'layer_1': 0.9 },
      entropy_collapsed: false,
      lstm_h_l2_total: null,
      lstm_c_l2_total: null,
      lstm_h_rms: null,
      lstm_c_rms: null,
      lstm_h_env_rms_mean: null,
      lstm_h_env_rms_max: null,
      lstm_c_env_rms_mean: null,
      lstm_c_env_rms_max: null,
      lstm_h_max: null,
      lstm_c_max: null,
      lstm_has_nan: false,
      lstm_has_inf: false,
      update_time_ms: 50,
      early_stop_epoch: null,
      head_slot_entropy: 1.2,
      head_blueprint_entropy: 1.1,
      head_style_entropy: 1.0,
      head_tempo_entropy: 0.9,
      head_alpha_target_entropy: 0.8,
      head_alpha_speed_entropy: 0.7,
      head_alpha_curve_entropy: 0.6,
      head_op_entropy: 1.3,
      head_slot_grad_norm: 0.1,
      head_blueprint_grad_norm: 0.1,
      head_style_grad_norm: 0.1,
      head_tempo_grad_norm: 0.1,
      head_alpha_target_grad_norm: 0.1,
      head_alpha_speed_grad_norm: 0.1,
      head_alpha_curve_grad_norm: 0.1,
      head_op_grad_norm: 0.1,
      head_slot_grad_norm_prev: 0.1,
      head_blueprint_grad_norm_prev: 0.1,
      head_style_grad_norm_prev: 0.1,
      head_tempo_grad_norm_prev: 0.1,
      head_alpha_target_grad_norm_prev: 0.1,
      head_alpha_speed_grad_norm_prev: 0.1,
      head_alpha_curve_grad_norm_prev: 0.1,
      head_op_grad_norm_prev: 0.1,
      head_slot_ratio_max: 1.1,
      head_blueprint_ratio_max: 1.1,
      head_style_ratio_max: 1.1,
      head_tempo_ratio_max: 1.1,
      head_alpha_target_ratio_max: 1.1,
      head_alpha_speed_ratio_max: 1.1,
      head_alpha_curve_ratio_max: 1.1,
      head_op_ratio_max: 1.1,
      joint_ratio_max: 1.1,
      episode_return_history: [0.3, 0.4, 0.5],
      current_episode_return: 0.5,
      current_episode: 1,
      policy_loss_history: [-0.02, -0.02, -0.02],
      value_loss_history: [0.1, 0.1, 0.1],
      grad_norm_history: [0.5, 0.5, 0.5],
      entropy_history: [1.5, 1.5, 1.5],
      explained_variance_history: [0.85, 0.85, 0.85],
      kl_divergence_history: [0.01, 0.01, 0.01],
      clip_fraction_history: [0.1, 0.1, 0.1],
      inner_epoch: 0,
      ppo_batch: 1,
      action_counts: { 'SPAWN': 10, 'WAIT': 50 },
      total_actions: 60,
      cumulative_action_counts: { 'WAIT': 50, 'SPAWN': 10 },
      cumulative_total_actions: 60,
      ppo_data_received: true,
      recent_decisions: [
        createMockDecisionSnapshot({
          decision_id: 'decision-002',
          chosen_action: 'GERMINATE',
          chosen_slot: 'slot_0',
          confidence: 0.72,
          expected_value: 0.24,
          actual_reward: 0.18,
          alternatives: [['GERMINATE', 0.24], ['WAIT', 0.16], ['PRUNE', -0.08]],
          td_advantage: 0.12,
          op_confidence: 0.72,
          slot_confidence: 0.68
        }),
        createMockDecisionSnapshot()
      ],
      group_id: 'test-group',
      entropy_velocity: 0.0,
      collapse_risk_score: 0.1,
      _previous_risk: 0.1,
      entropy_clip_correlation: 0.0,
      value_mean: 0.2,
      value_std: 0.1,
      value_min: 0,
      value_max: 0.4,
      initial_value_spread: 0.4,
      op_q_values: [0.1, 0.2, 0.15, 0.05, 0.08, 0.12],
      op_valid_mask: [true, true, true, true, true, true],
      q_variance: 0.12,
      q_spread: 0.15,
      last_action_success: true,
      last_action_op: 'WAIT',
      infrastructure: {
        cuda_memory_allocated_gb: 2,
        cuda_memory_reserved_gb: 4,
        cuda_memory_peak_gb: 4,
        cuda_memory_fragmentation: 0.1,
        dataloader_wait_ratio: 0.05,
        compile_enabled: false,
        compile_backend: '',
        compile_mode: ''
      },
      gradient_quality: {
        gradient_cv: 0.5,
        clip_fraction_positive: 0.05,
        clip_fraction_negative: 0.04
      },
      value_function: {
        v_return_correlation: 0.65,
        td_error_mean: 0.1,
        td_error_std: 0.2,
        bellman_error: 0.3,
        return_p10: -1,
        return_p50: 0.5,
        return_p90: 1.5,
        return_skewness: 0.1,
        return_variance: 0.5,
        value_predictions: [0.2],
        actual_returns: [0.4],
        td_errors: [0.1]
      }
    },
    vitals: {
      gpu_stats: {
        0: {
          device_id: 0,
          memory_used_gb: 4.0,
          memory_total_gb: 8.0,
          utilization: 75,
          temperature: 65
        }
      },
      gpu_memory_used_gb: 4.0,
      gpu_memory_total_gb: 8.0,
      gpu_utilization: 75,
      gpu_temperature: 65,
      cpu_percent: 45,
      ram_used_gb: 16,
      ram_total_gb: 32,
      epochs_per_second: 2.5,
      batches_per_hour: 150,
      host_params: 1000000
    },
    rewards: createMockRewardComponents(),
    slot_ids: ['slot_0', 'slot_1', 'slot_2', 'slot_3'],
    current_episode: 1,
    current_batch: 5,
    max_batches: 100,
    current_epoch: 42,
    max_epochs: 1000,
    run_id: 'test-run-001',
    task_name: 'cifar10',
    run_config: {
      seed: 42,
      n_episodes: 100,
      lr: 0.0003,
      clip_ratio: 0.2,
      entropy_coef: 0.01,
      param_budget: 100000,
      resume_path: '',
      entropy_anneal: { '0': 0.01, '50': 0.005 }
    },
    start_time: new Date().toISOString(),
    connected: true,
    runtime_seconds: 3600,
    staleness_seconds: 0,
    captured_at: new Date().toISOString(),
    total_events_received: 100,
    poll_count: 50,
    training_thread_alive: true,
    aggregate_mean_accuracy: 0.82,
    aggregate_mean_reward: 0.45,
    batch_avg_reward: 0.5,
    batch_total_episodes: 8,
    mean_accuracy_history: [0.7, 0.75, 0.8, 0.82],
    event_log: [
      {
        timestamp: new Date().toISOString(),
        event_type: 'SPAWN',
        env_id: 0,
        message: 'Spawned seed slot_0 with blueprint resnet_block',
        episode: 1,
        relative_time: '00:01:00',
        metadata: { slot_id: 'slot_0', blueprint: 'resnet_block' }
      }
    ],
    best_runs: [
      {
        env_id: 0,
        episode: 1,
        peak_accuracy: 0.85,
        final_accuracy: 0.85,
        epoch: 42,
        seeds: {},
        slot_ids: ['slot_0'],
        growth_ratio: 1.05,
        record_id: 'best-001',
        cumulative_reward: 1.5,
        peak_cumulative_reward: 1.5,
        reward_components: null,
        counterfactual_matrix: null,
        shapley_snapshot: null,
        action_history: ['SPAWN', 'WAIT'],
        reward_history: [0.3, 0.5],
        accuracy_history: [0.8, 0.85],
        host_loss: 0.23,
        host_params: 1000000,
        fossilized_count: 0,
        pruned_count: 0,
        reward_mode: 'STANDARD',
        blueprint_spawns: { 'resnet_block': 1 },
        blueprint_fossilized: {},
        blueprint_prunes: {},
        end_seeds: {
          'slot_0': createMockSeedState({ stage: 'BLENDING' })
        },
        end_reward_components: null,
        best_lifecycle_events: [],
        end_lifecycle_events: []
      }
    ],
    cumulative_germinated: 1,
    cumulative_fossilized: 0,
    cumulative_pruned: 0,
    cumulative_blueprint_spawns: { 'resnet_block': 1 },
    cumulative_blueprint_fossilized: {},
    cumulative_blueprint_prunes: {},
    slot_stage_counts: { 'TRAINING': 1, 'DORMANT': 3 },
    total_slots: 4,
    active_slots: 1,
    avg_epochs_in_stage: 10,
    last_ppo_update: new Date().toISOString(),
    last_reward_update: new Date().toISOString(),
    seed_lifecycle: {
      germination_count: 1,
      prune_count: 0,
      fossilize_count: 0,
      active_count: 1,
      total_slots: 4,
      germination_rate: 1,
      prune_rate: 0,
      fossilize_rate: 0,
      blend_success_rate: 0,
      avg_lifespan_epochs: 10,
      germination_trend: 'stable',
      prune_trend: 'stable',
      fossilize_trend: 'stable'
    },
    observation_stats: {
      slot_features_mean: 0,
      slot_features_std: 1,
      host_features_mean: 0,
      host_features_std: 1,
      context_features_mean: 0,
      context_features_std: 1,
      outlier_pct: 0,
      near_clip_pct: 0,
      clip_pct: 0,
      nan_count: 0,
      inf_count: 0,
      nan_pct: 0,
      inf_pct: 0,
      normalization_drift: 0,
      batch_size: 1
    },
    episode_stats: {
      length_mean: 150,
      length_std: 0,
      length_min: 150,
      length_max: 150,
      total_episodes: 8,
      episodes_per_second: 0.3,
      timeout_count: 0,
      success_count: 4,
      early_termination_count: 0,
      timeout_rate: 0,
      success_rate: 0.5,
      early_termination_rate: 0,
      steps_per_germinate: 4,
      steps_per_prune: 0,
      steps_per_fossilize: 0,
      action_entropy: 0.4,
      yield_rate: 0.4,
      slot_utilization: 0.25,
      completion_trend: 'stable'
    },
    focused_env_id: 0,
    last_action_env_id: 0,
    last_action_timestamp: new Date().toISOString()
  }
}

function createMockCohortSnapshots(): Record<string, SanctumSnapshot> {
  const shaped = createMockSnapshot()
  shaped.reward_mode = 'shaped'
  shaped.tamiyo.group_id = 'A'
  shaped.episode_stats.total_episodes = 64
  shaped.episode_stats.yield_rate = 0.22
  shaped.episode_stats.action_entropy = 1.35
  shaped.best_runs = [
    {
      ...shaped.best_runs[0],
      record_id: 'best-A',
      final_accuracy: 0.74,
      peak_accuracy: 0.78,
      growth_ratio: 1.12,
      reward_mode: 'shaped'
    }
  ]

  const simplified = createMockSnapshot()
  simplified.reward_mode = 'simplified'
  simplified.tamiyo.group_id = 'B'
  simplified.episode_stats.total_episodes = 68
  simplified.episode_stats.yield_rate = 0.48
  simplified.episode_stats.action_entropy = 0.82
  simplified.best_runs = [
    {
      ...simplified.best_runs[0],
      record_id: 'best-B',
      final_accuracy: 0.83,
      peak_accuracy: 0.84,
      growth_ratio: 1.06,
      reward_mode: 'simplified'
    }
  ]

  return { A: shaped, B: simplified }
}

/**
 * Injects a mock WebSocket that immediately sends snapshot data.
 * This allows the dashboard to render without a real backend.
 */
async function mockWebSocket(page: typeof import('@playwright/test').Page.prototype) {
  await page.addInitScript(() => {
    // Store the original WebSocket
    const OriginalWebSocket = window.WebSocket

    // Create mock WebSocket class
    class MockWebSocket extends EventTarget {
      static readonly CONNECTING = 0
      static readonly OPEN = 1
      static readonly CLOSING = 2
      static readonly CLOSED = 3

      readonly CONNECTING = 0
      readonly OPEN = 1
      readonly CLOSING = 2
      readonly CLOSED = 3

      readyState = MockWebSocket.CONNECTING
      url: string
      protocol = ''
      extensions = ''
      bufferedAmount = 0
      binaryType: BinaryType = 'blob'

      onopen: ((ev: Event) => void) | null = null
      onmessage: ((ev: MessageEvent) => void) | null = null
      onclose: ((ev: CloseEvent) => void) | null = null
      onerror: ((ev: Event) => void) | null = null

      constructor(url: string | URL, _protocols?: string | string[]) {
        super()
        this.url = typeof url === 'string' ? url : url.toString()

        // Simulate connection after a brief delay
        setTimeout(() => {
          this.readyState = MockWebSocket.OPEN
          const openEvent = new Event('open')
          this.dispatchEvent(openEvent)
          if (this.onopen) this.onopen(openEvent)

          // Send mock snapshot data
          this.sendMockData()
        }, 50)
      }

      private sendMockData() {
        // This will be populated by the test
        const testWindow = window as unknown as {
          __MOCK_SNAPSHOT__?: unknown
          __MOCK_SNAPSHOTS_BY_GROUP__?: unknown
          __MOCK_PRIMARY_GROUP_ID__?: string
        }
        const mockData = testWindow.__MOCK_SNAPSHOT__
        if (mockData) {
          const primaryGroupId = testWindow.__MOCK_PRIMARY_GROUP_ID__ ?? 'default'
          const snapshotsByGroup = testWindow.__MOCK_SNAPSHOTS_BY_GROUP__ ?? {
            default: mockData
          }
          const messageEvent = new MessageEvent('message', {
            data: JSON.stringify({
              type: 'snapshot',
              primary_group_id: primaryGroupId,
              data: mockData,
              snapshots_by_group: snapshotsByGroup
            })
          })
          this.dispatchEvent(messageEvent)
          if (this.onmessage) this.onmessage(messageEvent)
        }
      }

      send(_data: string | ArrayBufferLike | Blob | ArrayBufferView) {
        // No-op for mock
      }

      close(_code?: number, _reason?: string) {
        this.readyState = MockWebSocket.CLOSED
        const closeEvent = new CloseEvent('close', { wasClean: true, code: 1000 })
        this.dispatchEvent(closeEvent)
        if (this.onclose) this.onclose(closeEvent)
      }
    }

    // Replace global WebSocket
    ;(window as unknown as { WebSocket: typeof MockWebSocket }).WebSocket = MockWebSocket
  })
}

test.describe('Overwatch Dashboard Smoke Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Set up WebSocket mock before navigating
    await mockWebSocket(page)
  })

  test('page loads without errors', async ({ page }) => {
    // Inject mock snapshot data before page loads
    const mockSnapshot = createMockSnapshot()
    await page.addInitScript((snapshot) => {
      ;(window as unknown as { __MOCK_SNAPSHOT__: unknown }).__MOCK_SNAPSHOT__ = snapshot
    }, mockSnapshot)

    // Navigate to the dashboard
    await page.goto('/')

    // Verify no console errors
    const errors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text())
      }
    })

    // Wait for the page to be interactive
    await page.waitForLoadState('networkidle')

    // Verify the page title
    await expect(page).toHaveTitle(/Overwatch/)

    // Verify no critical errors (ignore WebSocket-related warnings)
    const criticalErrors = errors.filter(
      (e) => !e.includes('WebSocket') && !e.includes('ws://')
    )
    expect(criticalErrors).toHaveLength(0)
  })

  test('shows loading spinner initially', async ({ page }) => {
    // Navigate without mock data to see loading state
    await page.goto('/')

    // The loading state should be visible
    const loadingState = page.locator('[data-testid="loading-state"]')
    await expect(loadingState).toBeVisible({ timeout: 1000 })

    // Verify spinner is present
    const spinner = loadingState.locator('.loading-spinner')
    await expect(spinner).toBeVisible()

    // Verify loading text is present
    const loadingText = loadingState.locator('.loading-text')
    await expect(loadingText).toBeVisible()
    await expect(loadingText).toContainText(/Connecting|Loading/)
  })

  test('three-column layout renders when data arrives', async ({ page }) => {
    // Inject mock snapshot data
    const mockSnapshot = createMockSnapshot()
    await page.addInitScript((snapshot) => {
      ;(window as unknown as { __MOCK_SNAPSHOT__: unknown }).__MOCK_SNAPSHOT__ = snapshot
    }, mockSnapshot)

    await page.goto('/')

    // Wait for the dashboard to render
    await page.waitForSelector('[data-testid="left-sidebar"]', { timeout: 5000 })

    // Verify three-column layout is present
    const leftSidebar = page.locator('[data-testid="left-sidebar"]')
    const mainContent = page.locator('[data-testid="main-content"]')
    const rightPanel = page.locator('[data-testid="right-panel"]')

    await expect(leftSidebar).toBeVisible()
    await expect(mainContent).toBeVisible()
    await expect(rightPanel).toBeVisible()

    // Verify status bar is rendered
    const statusBar = page.locator('[data-testid="status-bar"]')
    await expect(statusBar).toBeVisible()

    // Verify top-level interpretation panels are rendered
    await expect(page.locator('[data-testid="experiment-verdict"]')).toBeVisible()
    await expect(page.locator('[data-testid="phase-gate"]')).toBeVisible()

    // Verify loading state is hidden
    const loadingState = page.locator('[data-testid="loading-state"]')
    await expect(loadingState).not.toBeVisible()
  })

  test('cohort comparison renders when grouped telemetry arrives', async ({ page }) => {
    const groupedSnapshots = createMockCohortSnapshots()
    await page.addInitScript((payload) => {
      const testWindow = window as unknown as {
        __MOCK_SNAPSHOT__?: unknown
        __MOCK_SNAPSHOTS_BY_GROUP__?: unknown
        __MOCK_PRIMARY_GROUP_ID__?: string
      }
      testWindow.__MOCK_SNAPSHOT__ = payload.primary
      testWindow.__MOCK_SNAPSHOTS_BY_GROUP__ = payload.snapshotsByGroup
      testWindow.__MOCK_PRIMARY_GROUP_ID__ = 'B'
    }, {
      primary: groupedSnapshots.B,
      snapshotsByGroup: groupedSnapshots
    })

    await page.goto('/')
    await page.waitForSelector('[data-testid="cohort-comparison"]', { timeout: 5000 })

    await expect(page.locator('[data-testid="cohort-A"]')).toContainText('shaped')
    await expect(page.locator('[data-testid="cohort-B"]')).toContainText('simplified')
    await expect(page.locator('[data-testid="cohort-B"]')).toContainText('83%')
  })

  test('pressing ? opens keyboard help overlay', async ({ page }) => {
    // Inject mock snapshot data
    const mockSnapshot = createMockSnapshot()
    await page.addInitScript((snapshot) => {
      ;(window as unknown as { __MOCK_SNAPSHOT__: unknown }).__MOCK_SNAPSHOT__ = snapshot
    }, mockSnapshot)

    await page.goto('/')

    // Wait for dashboard to render
    await page.waitForSelector('[data-testid="main-content"]', { timeout: 5000 })

    // Initially, help overlay should not be visible
    const helpOverlay = page.locator('[data-testid="keyboard-help-overlay"]')
    await expect(helpOverlay).not.toBeVisible()

    // Press ? to open help (use type instead of press for character input)
    await page.keyboard.type('?')

    // Help overlay should now be visible
    await expect(helpOverlay).toBeVisible({ timeout: 1000 })

    // Verify help modal content
    const helpModal = page.locator('[data-testid="keyboard-help-modal"]')
    await expect(helpModal).toBeVisible()
    await expect(helpModal).toContainText('Keyboard Shortcuts')

    // Press Escape to close
    await page.keyboard.press('Escape')

    // Help overlay should be hidden again
    await expect(helpOverlay).not.toBeVisible()
  })

  test('clicking close button closes help overlay', async ({ page }) => {
    // Inject mock snapshot data
    const mockSnapshot = createMockSnapshot()
    await page.addInitScript((snapshot) => {
      ;(window as unknown as { __MOCK_SNAPSHOT__: unknown }).__MOCK_SNAPSHOT__ = snapshot
    }, mockSnapshot)

    await page.goto('/')

    // Wait for dashboard to render
    await page.waitForSelector('[data-testid="main-content"]', { timeout: 5000 })

    // Open help overlay
    await page.keyboard.type('?')

    const helpOverlay = page.locator('[data-testid="keyboard-help-overlay"]')
    await expect(helpOverlay).toBeVisible({ timeout: 1000 })

    // Click close button
    const closeButton = page.locator('[data-testid="keyboard-help-close"]')
    await closeButton.click()

    // Help overlay should be hidden
    await expect(helpOverlay).not.toBeVisible()
  })

  test('clicking backdrop closes help overlay', async ({ page }) => {
    // Inject mock snapshot data
    const mockSnapshot = createMockSnapshot()
    await page.addInitScript((snapshot) => {
      ;(window as unknown as { __MOCK_SNAPSHOT__: unknown }).__MOCK_SNAPSHOT__ = snapshot
    }, mockSnapshot)

    await page.goto('/')

    // Wait for dashboard to render
    await page.waitForSelector('[data-testid="main-content"]', { timeout: 5000 })

    // Open help overlay
    await page.keyboard.type('?')

    const helpOverlay = page.locator('[data-testid="keyboard-help-overlay"]')
    await expect(helpOverlay).toBeVisible({ timeout: 1000 })

    // Click backdrop (at top-left corner, outside the modal)
    const backdrop = page.locator('[data-testid="keyboard-help-backdrop"]')
    await backdrop.click({ position: { x: 10, y: 10 } })

    // Help overlay should be hidden
    await expect(helpOverlay).not.toBeVisible()
  })
})
