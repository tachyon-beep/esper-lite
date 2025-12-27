// src/esper/karn/overwatch/web/src/components/__tests__/ContributionWaterfall.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import ContributionWaterfall from '../ContributionWaterfall.vue'
import type { RewardComponents } from '../../types/sanctum'

function createRewardComponents(overrides: Partial<RewardComponents> = {}): RewardComponents {
  return {
    total: 0.25,
    base_acc_delta: 0.15,
    bounded_attribution: 0.02,
    seed_contribution: 0.08,
    compute_rent: -0.05,
    alpha_shock: -0.02,
    ratio_penalty: -0.01,
    stage_bonus: 0.05,
    fossilize_terminal_bonus: 0.03,
    blending_warning: 0.0,
    holding_warning: 0.0,
    env_id: 0,
    val_acc: 0.85,
    last_action: 'WAIT',
    ...overrides
  }
}

describe('ContributionWaterfall', () => {
  it('renders all expected bar segments', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents()
      }
    })

    // Should have SVG waterfall chart
    expect(wrapper.find('svg').exists()).toBe(true)

    // Should have bars for each component
    expect(wrapper.find('[data-testid="bar-base-acc-delta"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="bar-seed-contribution"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="bar-stage-bonus"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="bar-compute-rent"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="bar-penalties"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="bar-total"]').exists()).toBe(true)
  })

  it('shows fossilize bonus bar when value is positive', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({ fossilize_terminal_bonus: 0.1 })
      }
    })

    expect(wrapper.find('[data-testid="bar-fossilize-bonus"]').exists()).toBe(true)
  })

  it('hides fossilize bonus bar when value is zero', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({ fossilize_terminal_bonus: 0 })
      }
    })

    expect(wrapper.find('[data-testid="bar-fossilize-bonus"]').exists()).toBe(false)
  })

  it('colors positive values green', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({
          base_acc_delta: 0.15,
          seed_contribution: 0.08,
          stage_bonus: 0.05
        })
      }
    })

    // Check that positive bars have the positive class
    const baseAccBar = wrapper.find('[data-testid="bar-base-acc-delta"]')
    expect(baseAccBar.classes()).toContain('positive')

    const seedContribBar = wrapper.find('[data-testid="bar-seed-contribution"]')
    expect(seedContribBar.classes()).toContain('positive')

    const stageBonusBar = wrapper.find('[data-testid="bar-stage-bonus"]')
    expect(stageBonusBar.classes()).toContain('positive')
  })

  it('colors negative values red', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({
          compute_rent: -0.05,
          alpha_shock: -0.02,
          ratio_penalty: -0.01
        })
      }
    })

    // Check that negative bars have the negative class
    const computeRentBar = wrapper.find('[data-testid="bar-compute-rent"]')
    expect(computeRentBar.classes()).toContain('negative')

    const penaltiesBar = wrapper.find('[data-testid="bar-penalties"]')
    expect(penaltiesBar.classes()).toContain('negative')
  })

  it('displays total bar with correct value', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({ total: 0.42 })
      }
    })

    const totalBar = wrapper.find('[data-testid="bar-total"]')
    expect(totalBar.exists()).toBe(true)
    expect(totalBar.classes()).toContain('total')

    // Check that total label shows the value
    const totalLabel = wrapper.find('[data-testid="label-total"]')
    expect(totalLabel.exists()).toBe(true)
    expect(totalLabel.text()).toContain('0.42')
  })

  it('displays labels with formatted values', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({
          base_acc_delta: 0.1234,
          compute_rent: -0.0567
        })
      }
    })

    // Labels should show formatted values
    const baseAccLabel = wrapper.find('[data-testid="label-base-acc-delta"]')
    expect(baseAccLabel.exists()).toBe(true)
    expect(baseAccLabel.text()).toContain('+0.12') // Positive values prefixed with +

    const computeRentLabel = wrapper.find('[data-testid="label-compute-rent"]')
    expect(computeRentLabel.exists()).toBe(true)
    expect(computeRentLabel.text()).toContain('-0.06') // Negative values show -
  })

  it('combines penalty values correctly', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({
          alpha_shock: -0.02,
          ratio_penalty: -0.01,
          blending_warning: -0.005,
          holding_warning: -0.005
        })
      }
    })

    // Penalties bar should be shown for combined value
    const penaltiesBar = wrapper.find('[data-testid="bar-penalties"]')
    expect(penaltiesBar.exists()).toBe(true)

    // Label should show combined value: -0.04
    const penaltiesLabel = wrapper.find('[data-testid="label-penalties"]')
    expect(penaltiesLabel.text()).toContain('-0.04')
  })

  it('handles zero-valued components appropriately', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({
          seed_contribution: 0,
          stage_bonus: 0
        })
      }
    })

    // Zero-valued components should still render but be minimal
    const seedContribBar = wrapper.find('[data-testid="bar-seed-contribution"]')
    expect(seedContribBar.exists()).toBe(true)

    const stageBonusBar = wrapper.find('[data-testid="bar-stage-bonus"]')
    expect(stageBonusBar.exists()).toBe(true)
  })

  it('handles negative total value correctly', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({
          total: -0.15,
          base_acc_delta: 0.05,
          seed_contribution: 0.02,
          compute_rent: -0.15,
          alpha_shock: -0.05,
          ratio_penalty: -0.02,
          stage_bonus: 0,
          fossilize_terminal_bonus: 0,
          blending_warning: 0,
          holding_warning: 0
        })
      }
    })

    const totalBar = wrapper.find('[data-testid="bar-total"]')
    expect(totalBar.exists()).toBe(true)

    const totalLabel = wrapper.find('[data-testid="label-total"]')
    expect(totalLabel.text()).toContain('-0.15')
  })

  // SVG Positioning Tests

  it('calculates SVG height based on visible bars', () => {
    // With fossilize_terminal_bonus = 0, we have 6 visible bars
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({ fossilize_terminal_bonus: 0 })
      }
    })

    const svg = wrapper.find('svg')
    // Height = TOP_PADDING(20) + BOTTOM_PADDING(20) + 6 bars * (BAR_HEIGHT(24) + BAR_GAP(8))
    // = 40 + 6 * 32 = 40 + 192 = 232
    expect(svg.attributes('height')).toBe('232')
  })

  it('increases SVG height when fossilize bonus is shown', () => {
    // With fossilize_terminal_bonus > 0, we have 7 visible bars
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({ fossilize_terminal_bonus: 0.1 })
      }
    })

    const svg = wrapper.find('svg')
    // Height = 40 + 7 bars * 32 = 40 + 224 = 264
    expect(svg.attributes('height')).toBe('264')
  })

  it('positions positive bar rect starting at midpoint', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({ base_acc_delta: 0.5 })
      }
    })

    const barRect = wrapper.find('[data-testid="bar-base-acc-delta"] rect')
    const x = parseFloat(barRect.attributes('x') || '0')

    // Midpoint = CHART_LEFT(110) + CHART_WIDTH(220) / 2 = 110 + 110 = 220
    // Positive bars start at midpoint
    expect(x).toBe(220)
  })

  it('positions negative bar rect ending at midpoint', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({
          compute_rent: -0.5,
          base_acc_delta: 0.5 // Need positive value to set scale
        })
      }
    })

    const barRect = wrapper.find('[data-testid="bar-compute-rent"] rect')
    const x = parseFloat(barRect.attributes('x') || '0')
    const width = parseFloat(barRect.attributes('width') || '0')

    // Negative bars should end at midpoint (220)
    // x + width should equal midpoint
    expect(x + width).toBe(220)
  })

  it('positions bars at correct Y offsets', () => {
    const wrapper = mount(ContributionWaterfall, {
      props: {
        rewards: createRewardComponents({ fossilize_terminal_bonus: 0 })
      }
    })

    // First bar at TOP_PADDING = 20
    const firstBar = wrapper.find('[data-testid="bar-base-acc-delta"] rect')
    expect(firstBar.attributes('y')).toBe('20')

    // Second bar at 20 + (24 + 8) = 52
    const secondBar = wrapper.find('[data-testid="bar-seed-contribution"] rect')
    expect(secondBar.attributes('y')).toBe('52')

    // Third bar at 20 + 2 * 32 = 84
    const thirdBar = wrapper.find('[data-testid="bar-stage-bonus"] rect')
    expect(thirdBar.attributes('y')).toBe('84')
  })
})
