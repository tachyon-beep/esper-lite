import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import PhaseGatePanel from '../PhaseGatePanel.vue'
import { createBestRunRecord, createSnapshot } from './factories'

describe('PhaseGatePanel', () => {
  it('marks a complete efficient run as a candidate for cohort comparison', () => {
    const snapshot = createSnapshot()
    snapshot.episode_stats.total_episodes = snapshot.run_config.n_episodes
    snapshot.best_runs = [
      createBestRunRecord({
        final_accuracy: 0.9,
        peak_accuracy: 0.92,
        growth_ratio: 1.05
      })
    ]

    const wrapper = mount(PhaseGatePanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="phase-gate-label"]').text()).toBe('Candidate')
    expect(wrapper.find('[data-testid="phase-metric-baseline"]').text()).toContain('60% @ 10% growth')
    expect(wrapper.find('[data-testid="phase-metric-final"]').text()).toContain('90%')
    expect(wrapper.find('[data-testid="phase-metric-growth"]').text()).toContain('5%')
    expect(wrapper.find('[data-testid="phase-metric-roi"]').text()).toContain('6.00x')
    expect(wrapper.find('[data-testid="phase-next-action"]').text()).toContain('companion cohorts')
  })

  it('keeps early runs in keep-running state until enough evidence exists', () => {
    const snapshot = createSnapshot()
    snapshot.episode_stats.total_episodes = 3
    snapshot.cumulative_germinated = 1
    snapshot.tamiyo.cumulative_total_actions = 8

    const wrapper = mount(PhaseGatePanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="phase-gate-label"]').text()).toBe('Keep Running')
    expect(wrapper.find('[data-testid="phase-gate-completion"]').classes()).toContain('gate-pending')
    expect(wrapper.find('[data-testid="phase-gate-lifecycle"]').text()).toContain('1 germinated')
    expect(wrapper.find('[data-testid="phase-gate-decision"]').text()).toContain('8 actions')
  })

  it('fails the gate when final accuracy does not beat the control baseline', () => {
    const snapshot = createSnapshot()
    snapshot.episode_stats.total_episodes = snapshot.run_config.n_episodes
    snapshot.best_runs = [
      createBestRunRecord({
        final_accuracy: 0.58,
        peak_accuracy: 0.61,
        growth_ratio: 1.05
      })
    ]

    const wrapper = mount(PhaseGatePanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="phase-gate-label"]').text()).toBe('Investigate')
    expect(wrapper.find('[data-testid="phase-gate-baseline"]').classes()).toContain('gate-fail')
    expect(wrapper.find('[data-testid="phase-next-action"]').text()).toContain('contradicts')
  })

  it('moves complete but noisy evidence into review state', () => {
    const snapshot = createSnapshot()
    snapshot.episode_stats.total_episodes = snapshot.run_config.n_episodes
    snapshot.episode_stats.action_entropy = 1.7

    const wrapper = mount(PhaseGatePanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="phase-gate-label"]').text()).toBe('Review')
    expect(wrapper.find('[data-testid="phase-gate-decision"]').classes()).toContain('gate-watch')
    expect(wrapper.find('[data-testid="phase-gate-decision"]').text()).toContain('entropy 1.70')
  })

  it('uses the best final-accuracy run for ROI rather than list position', () => {
    const snapshot = createSnapshot()
    snapshot.episode_stats.total_episodes = snapshot.run_config.n_episodes
    snapshot.best_runs = [
      createBestRunRecord({
        record_id: 'lower-final',
        final_accuracy: 0.72,
        peak_accuracy: 0.99,
        growth_ratio: 1.1
      }),
      createBestRunRecord({
        record_id: 'higher-final',
        final_accuracy: 0.84,
        peak_accuracy: 0.85,
        growth_ratio: 1.04
      })
    ]

    const wrapper = mount(PhaseGatePanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="phase-metric-final"]').text()).toContain('84%')
    expect(wrapper.find('[data-testid="phase-metric-growth"]').text()).toContain('4%')
  })
})
