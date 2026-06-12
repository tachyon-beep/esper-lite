import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import ActionContextPanel from '../ActionContextPanel.vue'
import {
  createDecisionSnapshot,
  createSnapshot,
  createTamiyoState
} from './factories'

describe('ActionContextPanel', () => {
  it('ranks critic preferences by Q value and marks separated critics', () => {
    const snapshot = createSnapshot({
      tamiyo: createTamiyoState({
        op_q_values: [0.1, 0.4, 0.2, -0.1, 0.05, 0.3],
        op_valid_mask: [true, true, true, true, true, true],
        q_variance: 0.12,
        q_spread: 0.5
      })
    })

    const wrapper = mount(ActionContextPanel, {
      props: { snapshot }
    })
    const rows = wrapper.findAll('.critic-row')

    expect(wrapper.find('[data-testid="q-health"]').text()).toBe('separated')
    expect(rows[0].attributes('data-testid')).toBe('critic-GERMINATE')
    expect(rows[0].classes()).toContain('rank-best')
    expect(rows[5].attributes('data-testid')).toBe('critic-PRUNE')
    expect(rows[5].classes()).toContain('rank-worst')
  })

  it('surfaces flat critic as a failure signal', () => {
    const snapshot = createSnapshot({
      tamiyo: createTamiyoState({
        op_q_values: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        op_valid_mask: [true, true, true, true, true, true],
        q_variance: 0.005,
        q_spread: 0
      })
    })

    const wrapper = mount(ActionContextPanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="q-health"]').text()).toBe('flat')
    expect(wrapper.find('[data-testid="q-health"]').classes()).toContain('q-flat')
  })

  it('renders return distribution and reward signal from the snapshot', () => {
    const snapshot = createSnapshot()
    snapshot.tamiyo.value_function.return_p10 = -0.25
    snapshot.tamiyo.value_function.return_p50 = 0.5
    snapshot.tamiyo.value_function.return_p90 = 1.25
    snapshot.tamiyo.value_function.v_return_correlation = 0.7
    snapshot.rewards.total = 0.42
    snapshot.rewards.compute_rent = -0.06
    snapshot.rewards.escrow_delta = 0.03

    const wrapper = mount(ActionContextPanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="return-shape"]').text()).toContain('-0.250')
    expect(wrapper.find('[data-testid="return-shape"]').text()).toContain('+1.250')
    expect(wrapper.find('[data-testid="reward-total"]').text()).toContain('+0.420')
    expect(wrapper.find('[data-testid="reward-rent"]').classes()).toContain('negative')
    expect(wrapper.find('[data-testid="reward-escrow"]').text()).toContain('+0.030')
  })

  it('shows recent decisions chronologically with latest decision details', () => {
    const snapshot = createSnapshot({
      tamiyo: createTamiyoState({
        recent_decisions: [
          createDecisionSnapshot({
            decision_id: 'newest',
            chosen_action: 'PRUNE',
            confidence: 0.65,
            td_advantage: -0.12,
            action_success: false
          }),
          createDecisionSnapshot({
            decision_id: 'oldest',
            chosen_action: 'GERMINATE',
            confidence: 0.82,
            td_advantage: 0.2,
            action_success: true
          })
        ]
      })
    })

    const wrapper = mount(ActionContextPanel, {
      props: { snapshot }
    })
    const decisions = wrapper
      .findAll('[data-testid^="decision-"]')
      .filter((row) => row.attributes('data-testid') !== 'decision-sequence')

    expect(decisions[0].attributes('data-testid')).toBe('decision-oldest')
    expect(decisions[1].attributes('data-testid')).toBe('decision-newest')
    expect(wrapper.find('[data-testid="decision-newest"]').classes()).toContain('failed')
    expect(wrapper.find('[data-testid="last-decision"]').text()).toContain('PRUN')
    expect(wrapper.find('[data-testid="last-decision"]').text()).toContain('conf 65%')
  })

  it('renders an explicit empty state when no recent decisions exist', () => {
    const snapshot = createSnapshot({
      tamiyo: createTamiyoState({ recent_decisions: [] })
    })

    const wrapper = mount(ActionContextPanel, {
      props: { snapshot }
    })

    expect(wrapper.find('[data-testid="decision-empty"]').text()).toBe('No decisions')
  })
})
