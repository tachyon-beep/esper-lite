import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import ExperimentVerdictPanel from '../ExperimentVerdictPanel.vue'
import { createSnapshot } from './factories'

describe('ExperimentVerdictPanel', () => {
  it('marks a healthy run as interpretable', () => {
    const wrapper = mount(ExperimentVerdictPanel, {
      props: { snapshot: createSnapshot() }
    })

    expect(wrapper.find('[data-testid="verdict-label"]').text()).toBe('Interpretable')
    expect(wrapper.find('[data-testid="gate-data-feed"]').classes()).toContain('gate-pass')
    expect(wrapper.find('[data-testid="gate-numerics"]').text()).toContain('clean')
    expect(wrapper.find('[data-testid="kpi-value-corr"]').text()).toContain('+0.650')
  })

  it('keeps early telemetry in collecting state until the run has enough signal', () => {
    const wrapper = mount(ExperimentVerdictPanel, {
      props: {
        snapshot: createSnapshot({
          current_batch: 1,
          total_events_received: 3,
          cumulative_germinated: 0,
          tamiyo: createSnapshot().tamiyo
        })
      }
    })

    expect(wrapper.find('[data-testid="verdict-label"]').text()).toBe('Collecting')
    expect(wrapper.find('[data-testid="gate-data-feed"]').classes()).toContain('gate-pending')
    expect(wrapper.find('[data-testid="gate-lifecycle"]').text()).toContain('no germination')
  })

  it('surfaces watch state when policy or lifecycle metrics are concerning but not failed', () => {
    const snapshot = createSnapshot()
    snapshot.tamiyo.collapse_risk_score = 0.7
    snapshot.episode_stats.yield_rate = 0.1

    const wrapper = mount(ExperimentVerdictPanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="verdict-label"]').text()).toBe('Watch')
    expect(wrapper.find('[data-testid="gate-policy"]').classes()).toContain('gate-watch')
    expect(wrapper.find('[data-testid="gate-lifecycle"]').classes()).toContain('gate-watch')
  })

  it('blocks interpretation when numerical faults or stopped training thread appear', () => {
    const snapshot = createSnapshot({ training_thread_alive: false })
    snapshot.observation_stats.nan_count = 1
    snapshot.tamiyo.nan_grad_count = 2

    const wrapper = mount(ExperimentVerdictPanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="verdict-label"]').text()).toBe('Blocked')
    expect(wrapper.find('[data-testid="gate-data-feed"]').text()).toContain('thread stopped')
    expect(wrapper.find('[data-testid="gate-numerics"]').classes()).toContain('gate-fail')
    expect(wrapper.find('[data-testid="gate-numerics"]').text()).toContain('3 tensor faults')
  })

  it('does not flag the policy gate on an artefactual low EV when value_nrmse is healthy', () => {
    // EV-telemetry-robustness: a deeply negative EV with ev_low_return_variance set is a
    // denominator artifact, not a policy collapse. The gate keys on the robust value_nrmse and
    // must NOT enter watch when the value head is actually fitting (low NRMSE).
    const snapshot = createSnapshot()
    snapshot.tamiyo.explained_variance = -8.0
    snapshot.tamiyo.ev_low_return_variance = true
    snapshot.tamiyo.value_nrmse = 0.3

    const wrapper = mount(ExperimentVerdictPanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="gate-policy"]').classes()).not.toContain('gate-watch')
  })

  it('flags the policy gate on a genuinely bad value fit (high value_nrmse)', () => {
    // A high value_nrmse means residual RMSE exceeds the return spread — a real fit failure,
    // regardless of the EV sign.
    const snapshot = createSnapshot()
    snapshot.tamiyo.explained_variance = 0.85
    snapshot.tamiyo.ev_low_return_variance = false
    snapshot.tamiyo.value_nrmse = 1.5

    const wrapper = mount(ExperimentVerdictPanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="gate-policy"]').classes()).toContain('gate-watch')
  })

  it('renders the dominant cumulative action mix in descending order', () => {
    const snapshot = createSnapshot()
    snapshot.tamiyo.cumulative_action_counts = {
      WAIT: 50,
      GERMINATE: 25,
      FOSSILIZE: 15,
      PRUNE: 10
    }
    snapshot.tamiyo.cumulative_total_actions = 100

    const wrapper = mount(ExperimentVerdictPanel, {
      props: { snapshot }
    })
    const rows = wrapper
      .findAll('[data-testid^="action-"]')
      .filter((row) => row.attributes('data-testid') !== 'action-mix')

    expect(rows.map((row) => row.attributes('data-testid'))).toEqual([
      'action-WAIT',
      'action-GERMINATE',
      'action-FOSSILIZE',
      'action-PRUNE'
    ])
    expect(wrapper.find('[data-testid="action-WAIT"]').text()).toContain('50%')
  })
})
