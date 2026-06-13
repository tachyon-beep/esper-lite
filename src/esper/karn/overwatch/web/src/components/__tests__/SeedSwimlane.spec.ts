// src/esper/karn/overwatch/web/src/components/__tests__/SeedSwimlane.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import SeedSwimlane from '../SeedSwimlane.vue'
import type { SeedState, SeedStage } from '../../types/sanctum'
import { createSeedState } from './factories'

describe('SeedSwimlane', () => {
  it('renders correct number of slot rows', () => {
    const seeds: Record<string, SeedState> = {
      slot_0: createSeedState({ slot_id: 'slot_0', stage: 'TRAINING' }),
      slot_1: createSeedState({ slot_id: 'slot_1', stage: 'DORMANT' }),
      slot_2: createSeedState({ slot_id: 'slot_2', stage: 'FOSSILIZED' })
    }
    const slotIds = ['slot_0', 'slot_1', 'slot_2']

    const wrapper = mount(SeedSwimlane, {
      props: {
        seeds,
        slotIds,
        currentEpoch: 100
      }
    })

    const rows = wrapper.findAll('[data-testid="swimlane-row"]')
    expect(rows.length).toBe(3)
  })

  it('displays slot ID labels', () => {
    const seeds: Record<string, SeedState> = {
      slot_0: createSeedState({ slot_id: 'slot_0', stage: 'TRAINING' }),
      slot_1: createSeedState({ slot_id: 'slot_1', stage: 'BLENDING' })
    }
    const slotIds = ['slot_0', 'slot_1']

    const wrapper = mount(SeedSwimlane, {
      props: {
        seeds,
        slotIds,
        currentEpoch: 100
      }
    })

    const labels = wrapper.findAll('[data-testid="slot-label"]')
    expect(labels.length).toBe(2)
    expect(labels[0].text()).toBe('S0')
    expect(labels[1].text()).toBe('S1')
  })

  it('colors bar segments based on stage', () => {
    const stages: SeedStage[] = [
      'DORMANT', 'GERMINATED', 'TRAINING', 'BLENDING',
      'HOLDING', 'FOSSILIZED', 'PRUNED'
    ]

    for (const stage of stages) {
      const seeds: Record<string, SeedState> = {
        slot_0: createSeedState({ slot_id: 'slot_0', stage, epochs_in_stage: 20 })
      }

      const wrapper = mount(SeedSwimlane, {
        props: {
          seeds,
          slotIds: ['slot_0'],
          currentEpoch: 100
        }
      })

      const bar = wrapper.find('[data-testid="stage-bar"]')
      expect(bar.classes()).toContain(`stage-${stage.toLowerCase()}`)
      wrapper.unmount()
    }
  })

  it('shows empty state when no seeds provided', () => {
    const wrapper = mount(SeedSwimlane, {
      props: {
        seeds: {},
        slotIds: [],
        currentEpoch: 100
      }
    })

    expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('No active seeds')
  })

  it('renders bar width proportional to epochs_in_stage', () => {
    const seeds: Record<string, SeedState> = {
      slot_0: createSeedState({ slot_id: 'slot_0', stage: 'TRAINING', epochs_in_stage: 25 }),
      slot_1: createSeedState({ slot_id: 'slot_1', stage: 'BLENDING', epochs_in_stage: 50 })
    }
    const slotIds = ['slot_0', 'slot_1']

    const wrapper = mount(SeedSwimlane, {
      props: {
        seeds,
        slotIds,
        currentEpoch: 100
      }
    })

    const bars = wrapper.findAll('[data-testid="stage-bar"]')
    expect(bars.length).toBe(2)

    // First bar should be narrower than second bar
    const bar0Style = bars[0].attributes('style')
    const bar1Style = bars[1].attributes('style')

    // Extract width percentages (epochs_in_stage / currentEpoch * 100)
    // slot_0: 25/100 = 25%, slot_1: 50/100 = 50%
    expect(bar0Style).toContain('25%')
    expect(bar1Style).toContain('50%')
  })

  it('renders legend with stage colors', () => {
    const seeds: Record<string, SeedState> = {
      slot_0: createSeedState({ slot_id: 'slot_0', stage: 'TRAINING' })
    }

    const wrapper = mount(SeedSwimlane, {
      props: {
        seeds,
        slotIds: ['slot_0'],
        currentEpoch: 100
      }
    })

    const legend = wrapper.find('[data-testid="swimlane-legend"]')
    expect(legend.exists()).toBe(true)

    // Legend should show all stage types
    const legendItems = wrapper.findAll('[data-testid="legend-item"]')
    expect(legendItems.length).toBeGreaterThan(0)
  })

  it('renders a never-observed configured slot as pending, not dormant', () => {
    // slotIds includes a configured slot that has never been observed.
    const seeds: Record<string, SeedState> = {
      slot_0: createSeedState({ slot_id: 'slot_0', stage: 'TRAINING' })
    }
    const slotIds = ['slot_0', 'slot_1'] // slot_1 was never observed

    const wrapper = mount(SeedSwimlane, {
      props: {
        seeds,
        slotIds,
        currentEpoch: 100
      }
    })

    // Should still render both rows
    const rows = wrapper.findAll('[data-testid="swimlane-row"]')
    expect(rows.length).toBe(2)

    // Second row is a pending lane, distinct from a dormant seed.
    const secondRow = rows[1]
    const pendingBar = secondRow.find('[data-testid="pending-bar"]')
    expect(pendingBar.exists()).toBe(true)
    expect(pendingBar.classes()).toContain('stage-pending')
    // It must NOT be a stage bar, and must NOT read as dormant.
    expect(secondRow.find('[data-testid="stage-bar"]').exists()).toBe(false)
    expect(pendingBar.classes()).not.toContain('stage-dormant')
  })

  it('renders an explicitly-observed dormant seed as dormant, not pending', () => {
    // slot_1 was observed and is in the DORMANT stage (explicit SeedState).
    const seeds: Record<string, SeedState> = {
      slot_0: createSeedState({ slot_id: 'slot_0', stage: 'TRAINING' }),
      slot_1: createSeedState({ slot_id: 'slot_1', stage: 'DORMANT' })
    }
    const slotIds = ['slot_0', 'slot_1']

    const wrapper = mount(SeedSwimlane, {
      props: {
        seeds,
        slotIds,
        currentEpoch: 100
      }
    })

    const rows = wrapper.findAll('[data-testid="swimlane-row"]')
    const secondRow = rows[1]
    const stageBar = secondRow.find('[data-testid="stage-bar"]')
    expect(stageBar.exists()).toBe(true)
    expect(stageBar.classes()).toContain('stage-dormant')
    // It must NOT render as a pending lane.
    expect(secondRow.find('[data-testid="pending-bar"]').exists()).toBe(false)
  })

  it('displays epochs_in_stage as tooltip on bar', () => {
    const seeds: Record<string, SeedState> = {
      slot_0: createSeedState({ slot_id: 'slot_0', stage: 'TRAINING', epochs_in_stage: 42 })
    }

    const wrapper = mount(SeedSwimlane, {
      props: {
        seeds,
        slotIds: ['slot_0'],
        currentEpoch: 100
      }
    })

    const bar = wrapper.find('[data-testid="stage-bar"]')
    expect(bar.attributes('title')).toContain('42')
    expect(bar.attributes('title')).toContain('TRAINING')
  })
})
