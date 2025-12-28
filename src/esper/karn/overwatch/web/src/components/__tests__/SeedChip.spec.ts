// src/esper/karn/overwatch/web/src/components/__tests__/SeedChip.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import SeedChip from '../SeedChip.vue'
import type { SeedStage } from '../../types/sanctum'

describe('SeedChip', () => {
  it('renders with correct stage color class', () => {
    const stages: SeedStage[] = [
      'DORMANT', 'GERMINATED', 'TRAINING', 'BLENDING',
      'HOLDING', 'FOSSILIZED', 'PRUNED'
    ]

    for (const stage of stages) {
      const wrapper = mount(SeedChip, {
        props: {
          slotId: 'slot_0',
          stage
        }
      })

      const chip = wrapper.find('[data-testid="seed-chip"]')
      expect(chip.classes()).toContain(`stage-${stage.toLowerCase()}`)
      wrapper.unmount()
    }
  })

  it('shows abbreviated slot ID', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'TRAINING' as SeedStage
      }
    })

    expect(wrapper.find('[data-testid="slot-id"]').text()).toBe('S0')
  })

  it('shows abbreviated slot ID for double-digit slots', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_12',
        stage: 'TRAINING' as SeedStage
      }
    })

    expect(wrapper.find('[data-testid="slot-id"]').text()).toBe('S12')
  })

  it('displays alpha percentage when provided during BLENDING', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'BLENDING' as SeedStage,
        alpha: 0.75
      }
    })

    expect(wrapper.find('[data-testid="alpha"]').text()).toContain('75%')
  })

  it('displays alpha percentage when provided during HOLDING', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'HOLDING' as SeedStage,
        alpha: 0.333
      }
    })

    expect(wrapper.find('[data-testid="alpha"]').text()).toContain('33%')
  })

  it('does not display alpha during non-blending stages', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'TRAINING' as SeedStage,
        alpha: 0.5
      }
    })

    expect(wrapper.find('[data-testid="alpha"]').exists()).toBe(false)
  })

  it('shows warning indicator when hasWarning is true', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'TRAINING' as SeedStage,
        hasWarning: true
      }
    })

    expect(wrapper.find('[data-testid="warning-indicator"]').exists()).toBe(true)
  })

  it('does not show warning indicator when hasWarning is false', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'TRAINING' as SeedStage,
        hasWarning: false
      }
    })

    expect(wrapper.find('[data-testid="warning-indicator"]').exists()).toBe(false)
  })

  it('has tooltip with full stage name', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'GERMINATED' as SeedStage
      }
    })

    const chip = wrapper.find('[data-testid="seed-chip"]')
    expect(chip.attributes('title')).toBe('slot_0: GERMINATED')
  })

  it('applies unknown stage styling for UNKNOWN stage', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'UNKNOWN' as SeedStage
      }
    })

    const chip = wrapper.find('[data-testid="seed-chip"]')
    expect(chip.classes()).toContain('stage-unknown')
  })

  // Curve glyph tests
  it('always displays curve glyph element', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'DORMANT' as SeedStage
      }
    })

    expect(wrapper.find('[data-testid="curve-glyph"]').exists()).toBe(true)
  })

  it('displays LINEAR curve glyph by default', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'TRAINING' as SeedStage
      }
    })

    // LINEAR = ╱ (U+2571)
    expect(wrapper.find('[data-testid="curve-glyph"]').text()).toBe('\u2571')
  })

  it('displays SIGMOID curve glyph when specified', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'BLENDING' as SeedStage,
        alphaCurve: 'SIGMOID'
      }
    })

    // SIGMOID = ∫ (U+222B)
    expect(wrapper.find('[data-testid="curve-glyph"]').text()).toBe('\u222B')
  })

  it('displays COSINE curve glyph when specified', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'HOLDING' as SeedStage,
        alphaCurve: 'COSINE'
      }
    })

    // COSINE = ∿ (U+223F)
    expect(wrapper.find('[data-testid="curve-glyph"]').text()).toBe('\u223F')
  })

  it('applies curve-active class during BLENDING stage', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'BLENDING' as SeedStage,
        alphaCurve: 'SIGMOID'
      }
    })

    const curve = wrapper.find('[data-testid="curve-glyph"]')
    expect(curve.classes()).toContain('curve-active')
    expect(curve.classes()).not.toContain('curve-dim')
    expect(curve.classes()).not.toContain('curve-historical')
  })

  it('applies curve-active class during HOLDING stage', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'HOLDING' as SeedStage,
        alphaCurve: 'LINEAR'
      }
    })

    const curve = wrapper.find('[data-testid="curve-glyph"]')
    expect(curve.classes()).toContain('curve-active')
  })

  it('applies curve-historical class during FOSSILIZED stage', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'FOSSILIZED' as SeedStage,
        alphaCurve: 'SIGMOID_GENTLE'
      }
    })

    const curve = wrapper.find('[data-testid="curve-glyph"]')
    expect(curve.classes()).toContain('curve-historical')
    expect(curve.classes()).not.toContain('curve-active')
    expect(curve.classes()).not.toContain('curve-dim')
  })

  it('applies curve-dim class during other stages (TRAINING)', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'TRAINING' as SeedStage,
        alphaCurve: 'LINEAR'
      }
    })

    const curve = wrapper.find('[data-testid="curve-glyph"]')
    expect(curve.classes()).toContain('curve-dim')
    expect(curve.classes()).not.toContain('curve-active')
    expect(curve.classes()).not.toContain('curve-historical')
  })

  it('applies curve-dim class during DORMANT stage', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'DORMANT' as SeedStage
      }
    })

    const curve = wrapper.find('[data-testid="curve-glyph"]')
    expect(curve.classes()).toContain('curve-dim')
  })

  it('falls back to minus sign for unknown curve type', () => {
    const wrapper = mount(SeedChip, {
      props: {
        slotId: 'slot_0',
        stage: 'BLENDING' as SeedStage,
        alphaCurve: 'UNKNOWN_CURVE'
      }
    })

    // Fallback = − (U+2212 minus sign)
    expect(wrapper.find('[data-testid="curve-glyph"]').text()).toBe('\u2212')
  })
})
