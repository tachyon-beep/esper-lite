// src/esper/karn/overwatch/web/src/components/__tests__/PolicyDiagnostics.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import PolicyDiagnostics from '../PolicyDiagnostics.vue'
import type { TamiyoState } from '../../types/sanctum'

function createTamiyo(overrides: Partial<TamiyoState> = {}): TamiyoState {
  return {
    entropy: 0.5,
    clip_fraction: 0.1,
    kl_divergence: 0.01,
    explained_variance: 0.85,
    policy_loss: 0.15,
    value_loss: 0.25,
    entropy_loss: 0.05,
    grad_norm: 1.0,
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
    advantage_raw_mean: 0.0,
    advantage_raw_std: 1.0,
    dead_layers: 0,
    exploding_layers: 0,
    nan_grad_count: 0,
    layer_gradient_health: null,
    entropy_collapsed: false,
    update_time_ms: 150,
    early_stop_epoch: null,
    head_slot_entropy: 0.5,
    head_blueprint_entropy: 0.5,
    head_style_entropy: 0.5,
    head_tempo_entropy: 0.5,
    head_alpha_target_entropy: 0.5,
    head_alpha_speed_entropy: 0.5,
    head_alpha_curve_entropy: 0.5,
    head_op_entropy: 0.5,
    head_slot_grad_norm: 1.0,
    head_blueprint_grad_norm: 1.0,
    head_style_grad_norm: 1.0,
    head_tempo_grad_norm: 1.0,
    head_alpha_target_grad_norm: 1.0,
    head_alpha_speed_grad_norm: 1.0,
    head_alpha_curve_grad_norm: 1.0,
    head_op_grad_norm: 1.0,
    episode_return_history: [],
    current_episode_return: 0,
    current_episode: 0,
    policy_loss_history: [],
    value_loss_history: [],
    grad_norm_history: [],
    entropy_history: [],
    explained_variance_history: [],
    kl_divergence_history: [],
    clip_fraction_history: [],
    inner_epoch: 0,
    ppo_batch: 0,
    action_counts: {},
    total_actions: 0,
    ppo_data_received: true,
    recent_decisions: [],
    group_id: null,
    ...overrides
  }
}

describe('PolicyDiagnostics', () => {
  describe('Loss Display Section', () => {
    it('renders policy_loss, value_loss, entropy_loss with labels', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({
            policy_loss: 0.123,
            value_loss: 0.456,
            entropy_loss: 0.078
          })
        }
      })

      expect(wrapper.find('[data-testid="policy-loss"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="policy-loss"]').text()).toContain('0.123')

      expect(wrapper.find('[data-testid="value-loss"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="value-loss"]').text()).toContain('0.456')

      expect(wrapper.find('[data-testid="entropy-loss"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="entropy-loss"]').text()).toContain('0.078')
    })

    it('displays loss labels correctly', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: { tamiyo: createTamiyo() }
      })

      expect(wrapper.find('[data-testid="policy-loss"]').text()).toContain('Policy')
      expect(wrapper.find('[data-testid="value-loss"]').text()).toContain('Value')
      expect(wrapper.find('[data-testid="entropy-loss"]').text()).toContain('Entropy')
    })
  })

  describe('Ratio Statistics Section', () => {
    it('renders ratio_mean, ratio_min, ratio_max, ratio_std', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({
            ratio_mean: 1.05,
            ratio_min: 0.85,
            ratio_max: 1.25,
            ratio_std: 0.12
          })
        }
      })

      expect(wrapper.find('[data-testid="ratio-mean"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="ratio-mean"]').text()).toContain('1.05')

      expect(wrapper.find('[data-testid="ratio-min"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="ratio-min"]').text()).toContain('0.85')

      expect(wrapper.find('[data-testid="ratio-max"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="ratio-max"]').text()).toContain('1.25')

      expect(wrapper.find('[data-testid="ratio-std"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="ratio-std"]').text()).toContain('0.12')
    })

    it('displays ratio stats in a group', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: { tamiyo: createTamiyo() }
      })

      expect(wrapper.find('[data-testid="ratio-stats-group"]').exists()).toBe(true)
    })
  })

  describe('Advantage Statistics Section', () => {
    it('renders advantage_mean, advantage_std, advantage_min, advantage_max', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({
            advantage_mean: 0.25,
            advantage_std: 1.5,
            advantage_min: -3.0,
            advantage_max: 4.0
          })
        }
      })

      expect(wrapper.find('[data-testid="advantage-mean"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="advantage-mean"]').text()).toContain('0.25')

      expect(wrapper.find('[data-testid="advantage-std"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="advantage-std"]').text()).toContain('1.5')

      expect(wrapper.find('[data-testid="advantage-min"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="advantage-min"]').text()).toContain('-3')

      expect(wrapper.find('[data-testid="advantage-max"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="advantage-max"]').text()).toContain('4')
    })

    it('displays advantage stats in a group', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: { tamiyo: createTamiyo() }
      })

      expect(wrapper.find('[data-testid="advantage-stats-group"]').exists()).toBe(true)
    })
  })

  describe('Health Indicators', () => {
    it('shows dead_layers with warning class when non-zero', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ dead_layers: 2 })
        }
      })

      const indicator = wrapper.find('[data-testid="dead-layers"]')
      expect(indicator.exists()).toBe(true)
      expect(indicator.text()).toContain('2')
      expect(indicator.classes()).toContain('health-warning')
    })

    it('shows dead_layers with good class when zero', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ dead_layers: 0 })
        }
      })

      const indicator = wrapper.find('[data-testid="dead-layers"]')
      expect(indicator.classes()).toContain('health-good')
    })

    it('shows exploding_layers with critical class when non-zero', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ exploding_layers: 3 })
        }
      })

      const indicator = wrapper.find('[data-testid="exploding-layers"]')
      expect(indicator.exists()).toBe(true)
      expect(indicator.text()).toContain('3')
      expect(indicator.classes()).toContain('health-critical')
    })

    it('shows exploding_layers with good class when zero', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ exploding_layers: 0 })
        }
      })

      const indicator = wrapper.find('[data-testid="exploding-layers"]')
      expect(indicator.classes()).toContain('health-good')
    })

    it('shows nan_grad_count with critical class when non-zero', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ nan_grad_count: 5 })
        }
      })

      const indicator = wrapper.find('[data-testid="nan-grad-count"]')
      expect(indicator.exists()).toBe(true)
      expect(indicator.text()).toContain('5')
      expect(indicator.classes()).toContain('health-critical')
    })

    it('shows nan_grad_count with good class when zero', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ nan_grad_count: 0 })
        }
      })

      const indicator = wrapper.find('[data-testid="nan-grad-count"]')
      expect(indicator.classes()).toContain('health-good')
    })

    it('shows entropy_collapsed prominently with critical styling when true', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ entropy_collapsed: true })
        }
      })

      const indicator = wrapper.find('[data-testid="entropy-collapsed"]')
      expect(indicator.exists()).toBe(true)
      expect(indicator.classes()).toContain('health-critical')
      expect(indicator.classes()).toContain('prominent')
    })

    it('shows entropy_collapsed as OK when false', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ entropy_collapsed: false })
        }
      })

      const indicator = wrapper.find('[data-testid="entropy-collapsed"]')
      expect(indicator.classes()).toContain('health-good')
      expect(indicator.classes()).not.toContain('prominent')
    })

    it('shows entropy with warning when below 0.1', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ entropy: 0.08 })
        }
      })

      const indicator = wrapper.find('[data-testid="global-entropy"]')
      expect(indicator.exists()).toBe(true)
      expect(indicator.classes()).toContain('health-warning')
    })

    it('shows entropy with good class when above 0.1', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ entropy: 0.5 })
        }
      })

      const indicator = wrapper.find('[data-testid="global-entropy"]')
      expect(indicator.classes()).toContain('health-good')
    })
  })

  describe('Per-Head Entropy Grid', () => {
    const headEntropies = [
      'head-entropy-slot',
      'head-entropy-blueprint',
      'head-entropy-style',
      'head-entropy-tempo',
      'head-entropy-alpha-target',
      'head-entropy-alpha-speed',
      'head-entropy-alpha-curve',
      'head-entropy-op'
    ]

    it('displays all 8 head entropies in a grid', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: { tamiyo: createTamiyo() }
      })

      expect(wrapper.find('[data-testid="head-entropy-grid"]').exists()).toBe(true)

      for (const testId of headEntropies) {
        expect(wrapper.find(`[data-testid="${testId}"]`).exists()).toBe(true)
      }
    })

    it('shows correct values for each head entropy', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({
            head_slot_entropy: 0.42,
            head_blueprint_entropy: 0.38,
            head_style_entropy: 0.55,
            head_tempo_entropy: 0.60,
            head_alpha_target_entropy: 0.33,
            head_alpha_speed_entropy: 0.28,
            head_alpha_curve_entropy: 0.45,
            head_op_entropy: 0.52
          })
        }
      })

      expect(wrapper.find('[data-testid="head-entropy-slot"]').text()).toContain('0.42')
      expect(wrapper.find('[data-testid="head-entropy-blueprint"]').text()).toContain('0.38')
      expect(wrapper.find('[data-testid="head-entropy-style"]').text()).toContain('0.55')
      expect(wrapper.find('[data-testid="head-entropy-tempo"]').text()).toContain('0.60')
      expect(wrapper.find('[data-testid="head-entropy-alpha-target"]').text()).toContain('0.33')
      expect(wrapper.find('[data-testid="head-entropy-alpha-speed"]').text()).toContain('0.28')
      expect(wrapper.find('[data-testid="head-entropy-alpha-curve"]').text()).toContain('0.45')
      expect(wrapper.find('[data-testid="head-entropy-op"]').text()).toContain('0.52')
    })

    it('shows warning for individual head entropy below 0.05', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({
            head_slot_entropy: 0.03,
            head_blueprint_entropy: 0.5
          })
        }
      })

      expect(wrapper.find('[data-testid="head-entropy-slot"]').classes()).toContain('health-warning')
      expect(wrapper.find('[data-testid="head-entropy-blueprint"]').classes()).toContain('health-good')
    })
  })

  describe('Per-Head Gradient Norms Grid', () => {
    const headGradNorms = [
      'head-grad-slot',
      'head-grad-blueprint',
      'head-grad-style',
      'head-grad-tempo',
      'head-grad-alpha-target',
      'head-grad-alpha-speed',
      'head-grad-alpha-curve',
      'head-grad-op'
    ]

    it('displays all 8 head gradient norms in a grid', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: { tamiyo: createTamiyo() }
      })

      expect(wrapper.find('[data-testid="head-grad-grid"]').exists()).toBe(true)

      for (const testId of headGradNorms) {
        expect(wrapper.find(`[data-testid="${testId}"]`).exists()).toBe(true)
      }
    })

    it('shows correct values for each head grad norm', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({
            head_slot_grad_norm: 1.23,
            head_blueprint_grad_norm: 0.89,
            head_style_grad_norm: 1.45,
            head_tempo_grad_norm: 0.76,
            head_alpha_target_grad_norm: 1.12,
            head_alpha_speed_grad_norm: 0.98,
            head_alpha_curve_grad_norm: 1.34,
            head_op_grad_norm: 0.67
          })
        }
      })

      expect(wrapper.find('[data-testid="head-grad-slot"]').text()).toContain('1.23')
      expect(wrapper.find('[data-testid="head-grad-blueprint"]').text()).toContain('0.89')
      expect(wrapper.find('[data-testid="head-grad-style"]').text()).toContain('1.45')
      expect(wrapper.find('[data-testid="head-grad-tempo"]').text()).toContain('0.76')
      expect(wrapper.find('[data-testid="head-grad-alpha-target"]').text()).toContain('1.12')
      expect(wrapper.find('[data-testid="head-grad-alpha-speed"]').text()).toContain('0.98')
      expect(wrapper.find('[data-testid="head-grad-alpha-curve"]').text()).toContain('1.34')
      expect(wrapper.find('[data-testid="head-grad-op"]').text()).toContain('0.67')
    })
  })

  describe('Update Timing Section', () => {
    it('displays update_time_ms', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ update_time_ms: 245 })
        }
      })

      const timing = wrapper.find('[data-testid="update-time"]')
      expect(timing.exists()).toBe(true)
      expect(timing.text()).toContain('245')
      expect(timing.text()).toContain('ms')
    })

    it('displays early_stop_epoch when set', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ early_stop_epoch: 3 })
        }
      })

      const earlyStop = wrapper.find('[data-testid="early-stop-epoch"]')
      expect(earlyStop.exists()).toBe(true)
      expect(earlyStop.text()).toContain('3')
    })

    it('does not display early_stop_epoch when null', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({ early_stop_epoch: null })
        }
      })

      expect(wrapper.find('[data-testid="early-stop-epoch"]').exists()).toBe(false)
    })
  })

  describe('Component Structure', () => {
    it('renders all main sections', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: { tamiyo: createTamiyo() }
      })

      expect(wrapper.find('[data-testid="loss-section"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="health-section"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="ratio-section"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="advantage-section"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="head-entropy-section"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="head-grad-section"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="timing-section"]').exists()).toBe(true)
    })

    it('has proper component root class', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: { tamiyo: createTamiyo() }
      })

      expect(wrapper.find('.policy-diagnostics').exists()).toBe(true)
    })
  })

  describe('Value Formatting', () => {
    it('formats loss values to 3 decimal places', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({
            policy_loss: 0.12345,
            value_loss: 0.98765,
            entropy_loss: 0.00123
          })
        }
      })

      expect(wrapper.find('[data-testid="policy-loss"]').text()).toContain('0.123')
      expect(wrapper.find('[data-testid="value-loss"]').text()).toContain('0.988')
      expect(wrapper.find('[data-testid="entropy-loss"]').text()).toContain('0.001')
    })

    it('formats ratio values to 2 decimal places', () => {
      const wrapper = mount(PolicyDiagnostics, {
        props: {
          tamiyo: createTamiyo({
            ratio_mean: 1.0567,
            ratio_std: 0.12345
          })
        }
      })

      expect(wrapper.find('[data-testid="ratio-mean"]').text()).toContain('1.06')
      expect(wrapper.find('[data-testid="ratio-std"]').text()).toContain('0.12')
    })
  })
})
