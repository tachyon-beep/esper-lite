// src/esper/karn/overwatch/web/src/components/__tests__/GradientHeatmap.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import GradientHeatmap from '../GradientHeatmap.vue'
import type { TamiyoState } from '../../types/sanctum'

function createTamiyo(overrides: Partial<TamiyoState> = {}): TamiyoState {
  return {
    entropy: 0.5,
    clip_fraction: 0.1,
    kl_divergence: 0.01,
    explained_variance: 0.85,
    policy_loss: 0.1,
    value_loss: 0.2,
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
    update_time_ms: 100,
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

describe('GradientHeatmap', () => {
  describe('Empty State', () => {
    it('shows empty state message when layer_gradient_health is null', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({ layer_gradient_health: null })
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap-empty"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="gradient-heatmap-empty"]').text()).toBe('No gradient data')
    })

    it('shows empty state message when layer_gradient_health is empty object', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({ layer_gradient_health: {} })
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap-empty"]').exists()).toBe(true)
    })

    it('hides grid when no gradient data exists', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({ layer_gradient_health: null })
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap-grid"]').exists()).toBe(false)
    })
  })

  describe('Grid Layout', () => {
    it('renders heatmap container with correct structure', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95, fc2: 0.85 }
          })
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="gradient-heatmap-grid"]').exists()).toBe(true)
    })

    it('renders one row per layer', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: {
              fc1: 0.95,
              fc2: 0.85,
              conv1: 0.72
            }
          })
        }
      })

      const rows = wrapper.findAll('[data-testid^="layer-row-"]')
      expect(rows.length).toBe(3)
    })

    it('renders layer name on left side of each row', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: {
              fc1: 0.95,
              conv2_bn: 0.85
            }
          })
        }
      })

      expect(wrapper.find('[data-testid="layer-row-fc1"] [data-testid="layer-name"]').text()).toBe('fc1')
      expect(wrapper.find('[data-testid="layer-row-conv2_bn"] [data-testid="layer-name"]').text()).toBe('conv2_bn')
    })

    it('renders health value in each cell', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: {
              fc1: 0.95,
              fc2: 0.5
            }
          })
        }
      })

      expect(wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-value"]').text()).toBe('0.95')
      expect(wrapper.find('[data-testid="layer-row-fc2"] [data-testid="health-value"]').text()).toBe('0.50')
    })

    it('formats health values to 2 decimal places', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: {
              fc1: 0.956789,
              fc2: 0.1
            }
          })
        }
      })

      expect(wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-value"]').text()).toBe('0.96')
      expect(wrapper.find('[data-testid="layer-row-fc2"] [data-testid="health-value"]').text()).toBe('0.10')
    })
  })

  describe('Color Coding', () => {
    it('applies status-good class for health > 0.8 (normal gradient)', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          })
        }
      })

      const cell = wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]')
      expect(cell.classes()).toContain('status-good')
    })

    it('applies status-good class for health exactly 0.81', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.81 }
          })
        }
      })

      const cell = wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]')
      expect(cell.classes()).toContain('status-good')
    })

    it('applies status-warn class for 0.3 < health <= 0.8 (weakening gradient)', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.5 }
          })
        }
      })

      const cell = wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]')
      expect(cell.classes()).toContain('status-warn')
    })

    it('applies status-warn class for health exactly 0.8', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.8 }
          })
        }
      })

      const cell = wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]')
      expect(cell.classes()).toContain('status-warn')
    })

    it('applies status-warn class for health exactly 0.31', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.31 }
          })
        }
      })

      const cell = wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]')
      expect(cell.classes()).toContain('status-warn')
    })

    it('applies status-loss class for health <= 0.3 (vanishing/exploding)', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.15 }
          })
        }
      })

      const cell = wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]')
      expect(cell.classes()).toContain('status-loss')
    })

    it('applies status-loss class for health exactly 0.3', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.3 }
          })
        }
      })

      const cell = wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]')
      expect(cell.classes()).toContain('status-loss')
    })

    it('applies status-loss class for health at 0', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0 }
          })
        }
      })

      const cell = wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]')
      expect(cell.classes()).toContain('status-loss')
    })

    it('applies correct colors for multiple layers with different health', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: {
              healthy_layer: 0.95,
              weakening_layer: 0.5,
              dying_layer: 0.1
            }
          })
        }
      })

      expect(wrapper.find('[data-testid="layer-row-healthy_layer"] [data-testid="health-cell"]').classes()).toContain('status-good')
      expect(wrapper.find('[data-testid="layer-row-weakening_layer"] [data-testid="health-cell"]').classes()).toContain('status-warn')
      expect(wrapper.find('[data-testid="layer-row-dying_layer"] [data-testid="health-cell"]').classes()).toContain('status-loss')
    })
  })

  describe('Summary Row', () => {
    it('renders summary row at the bottom', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 },
            dead_layers: 2,
            exploding_layers: 1
          })
        }
      })

      expect(wrapper.find('[data-testid="gradient-summary"]').exists()).toBe(true)
    })

    it('displays dead layer count', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 },
            dead_layers: 3,
            exploding_layers: 0
          })
        }
      })

      const summary = wrapper.find('[data-testid="gradient-summary"]')
      expect(summary.text()).toContain('Dead: 3')
    })

    it('displays exploding layer count', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 },
            dead_layers: 0,
            exploding_layers: 2
          })
        }
      })

      const summary = wrapper.find('[data-testid="gradient-summary"]')
      expect(summary.text()).toContain('Exploding: 2')
    })

    it('displays both counts in correct format', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 },
            dead_layers: 5,
            exploding_layers: 3
          })
        }
      })

      const summary = wrapper.find('[data-testid="gradient-summary"]')
      expect(summary.text()).toMatch(/Dead:\s*5\s*\|\s*Exploding:\s*3/)
    })

    it('shows zero counts correctly', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 },
            dead_layers: 0,
            exploding_layers: 0
          })
        }
      })

      const summary = wrapper.find('[data-testid="gradient-summary"]')
      expect(summary.text()).toContain('Dead: 0')
      expect(summary.text()).toContain('Exploding: 0')
    })

    it('does not show summary when there is no gradient data', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: null,
            dead_layers: 2,
            exploding_layers: 1
          })
        }
      })

      expect(wrapper.find('[data-testid="gradient-summary"]').exists()).toBe(false)
    })

    it('applies warning style when dead or exploding layers > 0', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 },
            dead_layers: 1,
            exploding_layers: 0
          })
        }
      })

      const summary = wrapper.find('[data-testid="gradient-summary"]')
      expect(summary.classes()).toContain('has-issues')
    })

    it('does not apply warning style when no issues', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 },
            dead_layers: 0,
            exploding_layers: 0
          })
        }
      })

      const summary = wrapper.find('[data-testid="gradient-summary"]')
      expect(summary.classes()).not.toContain('has-issues')
    })
  })

  describe('Compact Mode', () => {
    it('does not apply compact class by default', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          })
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap"]').classes()).not.toContain('compact')
    })

    it('applies compact class when compact prop is true', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          }),
          compact: true
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap"]').classes()).toContain('compact')
    })

    it('does not apply compact class when compact prop is false', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          }),
          compact: false
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap"]').classes()).not.toContain('compact')
    })
  })

  describe('Layer Name Abbreviation', () => {
    it('displays full short layer names', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: {
              fc1: 0.95,
              bn: 0.9,
              conv: 0.85
            }
          })
        }
      })

      expect(wrapper.find('[data-testid="layer-row-fc1"] [data-testid="layer-name"]').text()).toBe('fc1')
      expect(wrapper.find('[data-testid="layer-row-bn"] [data-testid="layer-name"]').text()).toBe('bn')
      expect(wrapper.find('[data-testid="layer-row-conv"] [data-testid="layer-name"]').text()).toBe('conv')
    })

    it('handles common neural network layer name patterns', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: {
              'policy.0.weight': 0.95,
              'value.2.bias': 0.85,
              'shared.encoder': 0.9
            }
          })
        }
      })

      const rows = wrapper.findAll('[data-testid^="layer-row-"]')
      expect(rows.length).toBe(3)
    })
  })

  describe('Layer Ordering', () => {
    it('renders layers in consistent order', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: {
              zz_layer: 0.95,
              aa_layer: 0.85,
              mm_layer: 0.9
            }
          })
        }
      })

      const rows = wrapper.findAll('[data-testid^="layer-row-"]')
      const names = rows.map(row => row.find('[data-testid="layer-name"]').text())

      // Should be sorted alphabetically
      expect(names).toEqual(['aa_layer', 'mm_layer', 'zz_layer'])
    })
  })

  describe('Data Test IDs', () => {
    it('has data-testid on container', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({ layer_gradient_health: null })
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap"]').exists()).toBe(true)
    })

    it('has data-testid on grid', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          })
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap-grid"]').exists()).toBe(true)
    })

    it('has data-testid on each row with layer name', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95, fc2: 0.85 }
          })
        }
      })

      expect(wrapper.find('[data-testid="layer-row-fc1"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="layer-row-fc2"]').exists()).toBe(true)
    })

    it('has data-testid on layer name element', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          })
        }
      })

      expect(wrapper.find('[data-testid="layer-name"]').exists()).toBe(true)
    })

    it('has data-testid on health cell', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          })
        }
      })

      expect(wrapper.find('[data-testid="health-cell"]').exists()).toBe(true)
    })

    it('has data-testid on health value', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          })
        }
      })

      expect(wrapper.find('[data-testid="health-value"]').exists()).toBe(true)
    })

    it('has data-testid on empty state', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({ layer_gradient_health: null })
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap-empty"]').exists()).toBe(true)
    })

    it('has data-testid on summary', () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          })
        }
      })

      expect(wrapper.find('[data-testid="gradient-summary"]').exists()).toBe(true)
    })
  })

  describe('Reactivity', () => {
    it('updates when layer_gradient_health changes', async () => {
      const tamiyo = createTamiyo({
        layer_gradient_health: { fc1: 0.95 }
      })

      const wrapper = mount(GradientHeatmap, {
        props: { tamiyo }
      })

      expect(wrapper.find('[data-testid="layer-row-fc1"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="layer-row-fc2"]').exists()).toBe(false)

      await wrapper.setProps({
        tamiyo: createTamiyo({
          layer_gradient_health: { fc1: 0.95, fc2: 0.85 }
        })
      })

      expect(wrapper.find('[data-testid="layer-row-fc1"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="layer-row-fc2"]').exists()).toBe(true)
    })

    it('updates color when health value changes', async () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          })
        }
      })

      expect(wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]').classes()).toContain('status-good')

      await wrapper.setProps({
        tamiyo: createTamiyo({
          layer_gradient_health: { fc1: 0.1 }
        })
      })

      expect(wrapper.find('[data-testid="layer-row-fc1"] [data-testid="health-cell"]').classes()).toContain('status-loss')
    })

    it('updates summary when dead/exploding counts change', async () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 },
            dead_layers: 0,
            exploding_layers: 0
          })
        }
      })

      expect(wrapper.find('[data-testid="gradient-summary"]').text()).toContain('Dead: 0')

      await wrapper.setProps({
        tamiyo: createTamiyo({
          layer_gradient_health: { fc1: 0.95 },
          dead_layers: 5,
          exploding_layers: 2
        })
      })

      expect(wrapper.find('[data-testid="gradient-summary"]').text()).toContain('Dead: 5')
      expect(wrapper.find('[data-testid="gradient-summary"]').text()).toContain('Exploding: 2')
    })

    it('transitions from data to empty state', async () => {
      const wrapper = mount(GradientHeatmap, {
        props: {
          tamiyo: createTamiyo({
            layer_gradient_health: { fc1: 0.95 }
          })
        }
      })

      expect(wrapper.find('[data-testid="gradient-heatmap-grid"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="gradient-heatmap-empty"]').exists()).toBe(false)

      await wrapper.setProps({
        tamiyo: createTamiyo({
          layer_gradient_health: null
        })
      })

      expect(wrapper.find('[data-testid="gradient-heatmap-grid"]').exists()).toBe(false)
      expect(wrapper.find('[data-testid="gradient-heatmap-empty"]').exists()).toBe(true)
    })
  })
})
