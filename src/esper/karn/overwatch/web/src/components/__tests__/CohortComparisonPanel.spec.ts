import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import CohortComparisonPanel from '../CohortComparisonPanel.vue'
import {
  createBestRunRecord,
  createSnapshot,
  createTamiyoState
} from './factories'

describe('CohortComparisonPanel', () => {
  it('renders reward-efficiency comparison for multiple policy groups', () => {
    const wrapper = mount(CohortComparisonPanel, {
      props: {
        snapshotsByGroup: {
          A: createSnapshot({
            reward_mode: 'shaped',
            run_config: { ...createSnapshot().run_config, n_episodes: 100 },
            episode_stats: { ...createSnapshot().episode_stats, total_episodes: 80, yield_rate: 0.2, action_entropy: 1.4 },
            tamiyo: createTamiyoState({ group_id: 'A', collapse_risk_score: 0.2 }),
            best_runs: [
              createBestRunRecord({
                record_id: 'a-best',
                final_accuracy: 0.76,
                peak_accuracy: 0.79,
                growth_ratio: 1.12,
                reward_mode: 'shaped'
              })
            ]
          }),
          B: createSnapshot({
            reward_mode: 'simplified',
            run_config: { ...createSnapshot().run_config, n_episodes: 100 },
            episode_stats: { ...createSnapshot().episode_stats, total_episodes: 82, yield_rate: 0.55, action_entropy: 0.8 },
            tamiyo: createTamiyoState({ group_id: 'B', collapse_risk_score: 0.1 }),
            best_runs: [
              createBestRunRecord({
                record_id: 'b-best',
                final_accuracy: 0.83,
                peak_accuracy: 0.84,
                growth_ratio: 1.06,
                reward_mode: 'simplified'
              })
            ]
          })
        },
        primaryGroupId: 'B'
      }
    })

    expect(wrapper.find('[data-testid="cohort-comparison"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="cohort-A"]').text()).toContain('shaped')
    expect(wrapper.find('[data-testid="cohort-B"]').text()).toContain('simplified')
    expect(wrapper.find('[data-testid="cohort-B"]').text()).toContain('83%')
    expect(wrapper.find('[data-testid="cohort-B"]').classes()).toContain('active')
    expect(wrapper.find('[data-testid="cohort-B"]').classes()).toContain('leader')
  })

  it('shows an empty evidence state when fewer than two cohorts are present', () => {
    const wrapper = mount(CohortComparisonPanel, {
      props: {
        snapshotsByGroup: {
          default: createSnapshot()
        },
        primaryGroupId: 'default'
      }
    })

    expect(wrapper.find('[data-testid="cohort-empty"]').text()).toContain('single policy')
  })
})
