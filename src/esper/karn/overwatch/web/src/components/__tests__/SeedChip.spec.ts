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
})
