// src/esper/karn/overwatch/web/src/components/__tests__/EnvironmentGrid.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import EnvironmentGrid from '../EnvironmentGrid.vue'
import type { EnvState } from '../../types/sanctum'
import { createEnvState } from './fixtures'

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
