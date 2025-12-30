// src/esper/karn/overwatch/web/e2e/smoke.spec.ts
/**
 * E2E Smoke Tests for Overwatch Dashboard
 *
 * These tests verify critical user flows work correctly in a real browser.
 * WebSocket connections are mocked to allow testing without a backend.
 */
import { test, expect } from '@playwright/test'
import type { SanctumSnapshot } from '../src/types/sanctum'

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
          'slot_0': {
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
            blend_tempo_epochs: 20
          }
        },
        active_seed_count: 1,
        fossilized_count: 0,
        pruned_count: 0,
        fossilized_params: 0,
        blueprint_spawns: { 'resnet_block': 1 },
        blueprint_prunes: {},
        blueprint_fossilized: {},
        reward_components: {
          total: 0.5,
          base_acc_delta: 0.02,
          bounded_attribution: 0.1,
          seed_contribution: 0.15,
          compute_rent: -0.01,
          alpha_shock: 0,
          ratio_penalty: 0,
          stage_bonus: 0.1,
          fossilize_terminal_bonus: 0,
          blending_warning: 0,
          holding_warning: 0,
          env_id: 0,
          val_acc: 0.85,
          last_action: 'SPAWN'
        },
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
      advantage_min: -2.0,
      advantage_max: 2.0,
      advantage_raw_mean: 0.5,
      advantage_raw_std: 1.2,
      dead_layers: 0,
      exploding_layers: 0,
      nan_grad_count: 0,
      layer_gradient_health: { 'layer_0': 1.0, 'layer_1': 0.9 },
      entropy_collapsed: false,
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
      ppo_data_received: true,
      recent_decisions: [],
      group_id: 'test-group'
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
    rewards: {
      total: 0.5,
      base_acc_delta: 0.02,
      bounded_attribution: 0.1,
      seed_contribution: 0.15,
      compute_rent: -0.01,
      alpha_shock: 0,
      ratio_penalty: 0,
      stage_bonus: 0.1,
      fossilize_terminal_bonus: 0,
      blending_warning: 0,
      holding_warning: 0,
      env_id: 0,
      val_acc: 0.85,
      last_action: 'SPAWN'
    },
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
        growth_ratio: 0.05,
        record_id: 'best-001',
        pinned: false,
        reward_components: null,
        counterfactual_matrix: null,
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
        blueprint_prunes: {}
      }
    ],
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
    focused_env_id: 0
  }
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
        const mockData = (window as unknown as { __MOCK_SNAPSHOT__?: unknown }).__MOCK_SNAPSHOT__
        if (mockData) {
          // useOverwatch expects messages in format { type: 'snapshot', data: ... }
          const messageEvent = new MessageEvent('message', {
            data: JSON.stringify({ type: 'snapshot', data: mockData })
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

    // Verify loading state is hidden
    const loadingState = page.locator('[data-testid="loading-state"]')
    await expect(loadingState).not.toBeVisible()
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
