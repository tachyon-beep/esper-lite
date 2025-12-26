// src/esper/karn/overwatch/web/src/components/__tests__/LeaderboardTable.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import LeaderboardTable from '../LeaderboardTable.vue'
import type { BestRunRecord, SeedState, RewardComponents, CounterfactualSnapshot } from '../../types/sanctum'

// Factory to create a valid BestRunRecord for testing
function createBestRunRecord(overrides: Partial<BestRunRecord> = {}): BestRunRecord {
  const defaults: BestRunRecord = {
    env_id: 0,
    episode: 1,
    peak_accuracy: 0.873,
    final_accuracy: 0.865,
    epoch: 100,
    seeds: {},
    slot_ids: ['slot_0', 'slot_1'],
    growth_ratio: 1.05,
    record_id: 'run-001',
    pinned: false,
    reward_components: null,
    counterfactual_matrix: null,
    action_history: ['OBSERVE', 'GERMINATE'],
    reward_history: [0.1, 0.2],
    accuracy_history: [0.8, 0.85],
    host_loss: 0.45,
    host_params: 1000000,
    fossilized_count: 2,
    pruned_count: 1,
    reward_mode: 'standard',
    blueprint_spawns: {},
    blueprint_fossilized: {},
    blueprint_prunes: {}
  }
  return { ...defaults, ...overrides }
}

describe('LeaderboardTable', () => {
  it('renders correct number of rows', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', env_id: 0, peak_accuracy: 0.95 }),
      createBestRunRecord({ record_id: 'run-002', env_id: 1, peak_accuracy: 0.90 }),
      createBestRunRecord({ record_id: 'run-003', env_id: 2, peak_accuracy: 0.85 })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    const rows = wrapper.findAll('[data-testid^="leaderboard-row-"]')
    expect(rows).toHaveLength(3)
  })

  it('sorts by column when header clicked', async () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', env_id: 0, peak_accuracy: 0.85 }),
      createBestRunRecord({ record_id: 'run-002', env_id: 1, peak_accuracy: 0.95 }),
      createBestRunRecord({ record_id: 'run-003', env_id: 2, peak_accuracy: 0.90 })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    // Default sort is peak_accuracy descending
    let rows = wrapper.findAll('[data-testid^="leaderboard-row-"]')
    expect(rows[0].attributes('data-testid')).toBe('leaderboard-row-run-002') // 0.95
    expect(rows[1].attributes('data-testid')).toBe('leaderboard-row-run-003') // 0.90
    expect(rows[2].attributes('data-testid')).toBe('leaderboard-row-run-001') // 0.85

    // Click on env column header to sort by env_id
    await wrapper.find('[data-testid="header-env"]').trigger('click')
    rows = wrapper.findAll('[data-testid^="leaderboard-row-"]')
    expect(rows[0].attributes('data-testid')).toBe('leaderboard-row-run-001') // env_id 0
    expect(rows[1].attributes('data-testid')).toBe('leaderboard-row-run-002') // env_id 1
    expect(rows[2].attributes('data-testid')).toBe('leaderboard-row-run-003') // env_id 2

    // Click again to reverse sort
    await wrapper.find('[data-testid="header-env"]').trigger('click')
    rows = wrapper.findAll('[data-testid^="leaderboard-row-"]')
    expect(rows[0].attributes('data-testid')).toBe('leaderboard-row-run-003') // env_id 2
    expect(rows[1].attributes('data-testid')).toBe('leaderboard-row-run-002') // env_id 1
    expect(rows[2].attributes('data-testid')).toBe('leaderboard-row-run-001') // env_id 0
  })

  it('displays accuracy as percentage', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', peak_accuracy: 0.8734 })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    const accuracy = wrapper.find('[data-testid="peak-acc-run-001"]')
    expect(accuracy.text()).toBe('87.3%')
  })

  it('respects maxRows prop', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', peak_accuracy: 0.95 }),
      createBestRunRecord({ record_id: 'run-002', peak_accuracy: 0.90 }),
      createBestRunRecord({ record_id: 'run-003', peak_accuracy: 0.85 }),
      createBestRunRecord({ record_id: 'run-004', peak_accuracy: 0.80 }),
      createBestRunRecord({ record_id: 'run-005', peak_accuracy: 0.75 })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs, maxRows: 3 }
    })

    const rows = wrapper.findAll('[data-testid^="leaderboard-row-"]')
    expect(rows).toHaveLength(3)
  })

  it('highlights pinned rows', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', pinned: true }),
      createBestRunRecord({ record_id: 'run-002', pinned: false })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    const pinnedRow = wrapper.find('[data-testid="leaderboard-row-run-001"]')
    const unpinnedRow = wrapper.find('[data-testid="leaderboard-row-run-002"]')

    expect(pinnedRow.classes()).toContain('pinned')
    expect(unpinnedRow.classes()).not.toContain('pinned')
  })

  it('shows correct rank numbers', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', peak_accuracy: 0.95 }),
      createBestRunRecord({ record_id: 'run-002', peak_accuracy: 0.90 }),
      createBestRunRecord({ record_id: 'run-003', peak_accuracy: 0.85 })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    // Default sort by peak_accuracy descending
    expect(wrapper.find('[data-testid="rank-run-001"]').text()).toBe('1')
    expect(wrapper.find('[data-testid="rank-run-002"]').text()).toBe('2')
    expect(wrapper.find('[data-testid="rank-run-003"]').text()).toBe('3')
  })

  it('displays empty state when no runs', () => {
    const wrapper = mount(LeaderboardTable, {
      props: { runs: [] }
    })

    expect(wrapper.find('[data-testid="empty-state"]').text()).toBe('No runs recorded')
    expect(wrapper.findAll('[data-testid^="leaderboard-row-"]')).toHaveLength(0)
  })

  it('emits select event when row is clicked', async () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001' }),
      createBestRunRecord({ record_id: 'run-002' })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    await wrapper.find('[data-testid="leaderboard-row-run-002"]').trigger('click')

    expect(wrapper.emitted('select')).toBeTruthy()
    expect(wrapper.emitted('select')![0]).toEqual(['run-002'])
  })

  it('displays fossilized and pruned counts', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', fossilized_count: 5, pruned_count: 3 })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    expect(wrapper.find('[data-testid="fossilized-run-001"]').text()).toBe('5')
    expect(wrapper.find('[data-testid="pruned-run-001"]').text()).toBe('3')
  })
})
