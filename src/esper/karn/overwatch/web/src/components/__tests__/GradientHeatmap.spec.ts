// src/esper/karn/overwatch/web/src/components/__tests__/GradientHeatmap.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import GradientHeatmap from '../GradientHeatmap.vue'
import { createTamiyoState as createTamiyo } from './fixtures'

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
