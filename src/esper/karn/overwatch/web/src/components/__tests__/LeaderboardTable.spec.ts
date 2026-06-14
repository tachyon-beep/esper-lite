// src/esper/karn/overwatch/web/src/components/__tests__/LeaderboardTable.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import LeaderboardTable from '../LeaderboardTable.vue'
import type { BestRunRecord } from '../../types/sanctum'
import { createBestRunRecord, createSeedState } from './factories'

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

  it('displays Sanctum scoreboard telemetry for reward-efficiency review', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({
        record_id: 'run-001',
        peak_accuracy: 0.8734,
        final_accuracy: 0.851,
        epoch: 47,
        growth_ratio: 1.052,
        cumulative_reward: 2.45,
        seeds: {},
        end_seeds: {
          alpha: createSeedState({ slot_id: 'alpha', stage: 'BLENDING' }),
          beta: createSeedState({ slot_id: 'beta', stage: 'HOLDING' }),
          gamma: createSeedState({ slot_id: 'gamma', stage: 'FOSSILIZED' })
        }
      })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    expect(wrapper.find('[data-testid="peak-acc-run-001"]').text()).toBe('87.3%')
    expect(wrapper.find('[data-testid="epoch-run-001"]').text()).toBe('47')
    expect(wrapper.find('[data-testid="trajectory-run-001"]').text()).toContain('85.1%')
    expect(wrapper.find('[data-testid="trajectory-run-001"]').classes()).toContain('trajectory-down')
    expect(wrapper.find('[data-testid="growth-run-001"]').text()).toBe('1.05x')
    expect(wrapper.find('[data-testid="reward-run-001"]').text()).toBe('+2.5')
    expect(wrapper.find('[data-testid="seeds-run-001"]').text()).toBe('1/1/1')
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

  it('uses the best-run seed composition when end-of-episode seeds are empty', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({
        record_id: 'run-001',
        seeds: {
          early: createSeedState({ slot_id: 'early', stage: 'BLENDING' }),
          mid: createSeedState({ slot_id: 'mid', stage: 'FOSSILIZED' }),
          late: createSeedState({ slot_id: 'late', stage: 'PRUNED' })
        },
        end_seeds: {}
      })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    expect(wrapper.find('[data-testid="seeds-run-001"]').text()).toBe('1/0/1')
  })

  it('sorts by reward telemetry when reward header is clicked', async () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', peak_accuracy: 0.95, cumulative_reward: -1.5 }),
      createBestRunRecord({ record_id: 'run-002', peak_accuracy: 0.90, cumulative_reward: 2.0 }),
      createBestRunRecord({ record_id: 'run-003', peak_accuracy: 0.85, cumulative_reward: 0.3 })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs }
    })

    await wrapper.find('[data-testid="header-reward"]').trigger('click')
    const rows = wrapper.findAll('[data-testid^="leaderboard-row-"]')
    expect(rows[0].attributes('data-testid')).toBe('leaderboard-row-run-002')
    expect(rows[1].attributes('data-testid')).toBe('leaderboard-row-run-003')
    expect(rows[2].attributes('data-testid')).toBe('leaderboard-row-run-001')
  })

  it('highlights selected row via keyboard navigation', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', peak_accuracy: 0.95 }),
      createBestRunRecord({ record_id: 'run-002', peak_accuracy: 0.90 }),
      createBestRunRecord({ record_id: 'run-003', peak_accuracy: 0.85 })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs, selectedRowIndex: 1 }
    })

    // Rows are sorted by peak_accuracy descending, so run-002 is at index 1
    const rows = wrapper.findAll('[data-testid^="leaderboard-row-"]')
    expect(rows[0].classes()).not.toContain('keyboard-selected')
    expect(rows[1].classes()).toContain('keyboard-selected')
    expect(rows[2].classes()).not.toContain('keyboard-selected')
  })

  it('does not highlight any row when selectedRowIndex is -1', () => {
    const runs: BestRunRecord[] = [
      createBestRunRecord({ record_id: 'run-001', peak_accuracy: 0.95 }),
      createBestRunRecord({ record_id: 'run-002', peak_accuracy: 0.90 })
    ]

    const wrapper = mount(LeaderboardTable, {
      props: { runs, selectedRowIndex: -1 }
    })

    const rows = wrapper.findAll('[data-testid^="leaderboard-row-"]')
    rows.forEach(row => {
      expect(row.classes()).not.toContain('keyboard-selected')
    })
  })
})
