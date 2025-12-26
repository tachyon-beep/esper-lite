// src/esper/karn/overwatch/web/src/components/__tests__/EnvironmentGrid.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import EnvironmentGrid from '../EnvironmentGrid.vue'
import type { EnvState, SeedState, RewardComponents, CounterfactualSnapshot } from '../../types/sanctum'

// Factory to create a valid EnvState for testing
function createEnvState(overrides: Partial<EnvState> = {}): EnvState {
  const defaults: EnvState = {
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
    reward_components: {
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
      last_action: 'OBSERVE'
    } as RewardComponents,
    counterfactual_matrix: {
      slot_ids: [],
      configs: [],
      strategy: 'ablation',
      compute_time_ms: 10
    } as CounterfactualSnapshot,
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
    reward_mode: 'standard'
  }
  return { ...defaults, ...overrides }
}

describe('EnvironmentGrid', () => {
  it('renders correct number of environment cards', () => {
    const envs: Record<number, EnvState> = {
      0: createEnvState({ env_id: 0 }),
      1: createEnvState({ env_id: 1 }),
      2: createEnvState({ env_id: 2 }),
      3: createEnvState({ env_id: 3 })
    }

    const wrapper = mount(EnvironmentGrid, {
      props: {
        envs,
        focusedEnvId: 0
      }
    })

    const cards = wrapper.findAll('[data-testid^="env-card-"]')
    expect(cards).toHaveLength(4)
  })

  it('shows focused environment with highlight', () => {
    const envs: Record<number, EnvState> = {
      0: createEnvState({ env_id: 0 }),
      1: createEnvState({ env_id: 1 })
    }

    const wrapper = mount(EnvironmentGrid, {
      props: {
        envs,
        focusedEnvId: 1
      }
    })

    const focusedCard = wrapper.find('[data-testid="env-card-1"]')
    expect(focusedCard.classes()).toContain('focused')

    const unfocusedCard = wrapper.find('[data-testid="env-card-0"]')
    expect(unfocusedCard.classes()).not.toContain('focused')
  })

  it('displays accuracy as percentage', () => {
    const envs: Record<number, EnvState> = {
      0: createEnvState({ env_id: 0, host_accuracy: 0.8734 })
    }

    const wrapper = mount(EnvironmentGrid, {
      props: {
        envs,
        focusedEnvId: 0
      }
    })

    const accuracy = wrapper.find('[data-testid="env-accuracy-0"]')
    expect(accuracy.text()).toContain('87.3%')
  })

  it('displays epoch number', () => {
    const envs: Record<number, EnvState> = {
      0: createEnvState({ env_id: 0, current_epoch: 150 })
    }

    const wrapper = mount(EnvironmentGrid, {
      props: {
        envs,
        focusedEnvId: 0
      }
    })

    const epoch = wrapper.find('[data-testid="env-epoch-0"]')
    expect(epoch.text()).toContain('150')
  })

  it('displays seed status counts', () => {
    const envs: Record<number, EnvState> = {
      0: createEnvState({
        env_id: 0,
        active_seed_count: 5,
        fossilized_count: 3,
        pruned_count: 2
      })
    }

    const wrapper = mount(EnvironmentGrid, {
      props: {
        envs,
        focusedEnvId: 0
      }
    })

    const seedCounts = wrapper.find('[data-testid="env-seed-counts-0"]')
    expect(seedCounts.text()).toContain('5')
    expect(seedCounts.text()).toContain('3')
    expect(seedCounts.text()).toContain('2')
  })

  it('shows correct status indicator color class', () => {
    const envs: Record<number, EnvState> = {
      0: createEnvState({ env_id: 0, status: 'healthy' }),
      1: createEnvState({ env_id: 1, status: 'stalled' }),
      2: createEnvState({ env_id: 2, status: 'degraded' })
    }

    const wrapper = mount(EnvironmentGrid, {
      props: {
        envs,
        focusedEnvId: 0
      }
    })

    expect(wrapper.find('[data-testid="env-status-0"]').classes()).toContain('status-healthy')
    expect(wrapper.find('[data-testid="env-status-1"]').classes()).toContain('status-stalled')
    expect(wrapper.find('[data-testid="env-status-2"]').classes()).toContain('status-degraded')
  })

  it('emits select event when card is clicked', async () => {
    const envs: Record<number, EnvState> = {
      0: createEnvState({ env_id: 0 }),
      1: createEnvState({ env_id: 1 })
    }

    const wrapper = mount(EnvironmentGrid, {
      props: {
        envs,
        focusedEnvId: 0
      }
    })

    await wrapper.find('[data-testid="env-card-1"]').trigger('click')

    expect(wrapper.emitted('select')).toBeTruthy()
    expect(wrapper.emitted('select')![0]).toEqual([1])
  })

  it('displays env ID badge', () => {
    const envs: Record<number, EnvState> = {
      0: createEnvState({ env_id: 0 }),
      7: createEnvState({ env_id: 7 })
    }

    const wrapper = mount(EnvironmentGrid, {
      props: {
        envs,
        focusedEnvId: 0
      }
    })

    expect(wrapper.find('[data-testid="env-badge-0"]').text()).toBe('0')
    expect(wrapper.find('[data-testid="env-badge-7"]').text()).toBe('7')
  })
})
