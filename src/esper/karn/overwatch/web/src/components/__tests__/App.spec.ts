// src/esper/karn/overwatch/web/src/components/__tests__/App.spec.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import App from '../../App.vue'
import { createSnapshot } from './factories'

// Get the global object in a type-safe way
const globalObj = globalThis as typeof globalThis & { WebSocket: typeof WebSocket }

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  readyState = 0

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }

  close() {
    this.readyState = 3
  }

  simulateOpen() {
    this.readyState = 1
    this.onopen?.()
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) })
  }

  simulateClose() {
    this.readyState = 3
    this.onclose?.()
  }
}

describe('App.vue Integration', () => {
  let originalWebSocket: typeof WebSocket

  function snapshotMessage(snapshot: ReturnType<typeof createSnapshot>) {
    return {
      type: 'snapshot',
      primary_group_id: 'default',
      data: snapshot,
      snapshots_by_group: {
        default: snapshot
      }
    }
  }

  beforeEach(() => {
    originalWebSocket = globalObj.WebSocket
    globalObj.WebSocket = MockWebSocket as unknown as typeof WebSocket
    MockWebSocket.instances = []
    vi.useFakeTimers()
  })

  afterEach(() => {
    globalObj.WebSocket = originalWebSocket
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  describe('Layout Structure', () => {
    it('renders three-column layout with data-testid attributes', async () => {
      const wrapper = mount(App)

      // Simulate WebSocket connection
      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(createSnapshot()))
      await flushPromises()

      expect(wrapper.find('[data-testid="left-sidebar"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="main-content"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="right-panel"]').exists()).toBe(true)
    })

    it('renders StatusBar at top spanning full width', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(createSnapshot()))
      await flushPromises()

      expect(wrapper.find('[data-testid="status-bar"]').exists()).toBe(true)
    })

    it('renders experiment verdict before detail panels', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(createSnapshot()))
      await flushPromises()

      expect(wrapper.find('[data-testid="experiment-verdict"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="verdict-label"]').text()).toContain('Interpretable')
    })

    it('renders phase gate evidence for reward-efficiency interpretation', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(createSnapshot()))
      await flushPromises()

      expect(wrapper.find('[data-testid="phase-gate"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="phase-gate-label"]').text()).toContain('Keep Running')
    })

    it('renders cohort comparison when grouped snapshots arrive', async () => {
      const wrapper = mount(App)
      const shaped = createSnapshot({ reward_mode: 'shaped' })
      const simplified = createSnapshot({ reward_mode: 'simplified' })

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage({
        type: 'snapshot',
        primary_group_id: 'B',
        data: simplified,
        snapshots_by_group: {
          A: shaped,
          B: simplified
        }
      })
      await flushPromises()

      expect(wrapper.find('[data-testid="cohort-comparison"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="cohort-A"]').text()).toContain('shaped')
      expect(wrapper.find('[data-testid="cohort-B"]').text()).toContain('simplified')
    })
  })

  describe('Loading States', () => {
    it('shows connecting state initially', () => {
      const wrapper = mount(App)

      expect(wrapper.find('[data-testid="loading-state"]').exists()).toBe(true)
      expect(wrapper.text()).toContain('Connecting')
    })

    it('shows loading state when connected but no snapshot yet', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      await flushPromises()

      expect(wrapper.find('[data-testid="loading-state"]').exists()).toBe(true)
      expect(wrapper.text()).toContain('Loading')
    })

    it('hides loading state when snapshot is received', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(createSnapshot()))
      await flushPromises()

      expect(wrapper.find('[data-testid="loading-state"]').exists()).toBe(false)
    })
  })

  describe('StatusBar Props', () => {
    it('passes connection state to StatusBar', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(createSnapshot()))
      await flushPromises()

      const statusBar = wrapper.findComponent({ name: 'StatusBar' })
      expect(statusBar.exists()).toBe(true)
      expect(statusBar.props('connectionState')).toBe('connected')
    })

    it('passes episode, epoch, batch from snapshot to StatusBar', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot({
        current_episode: 10,
        current_epoch: 200,
        current_batch: 25
      })

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const statusBar = wrapper.findComponent({ name: 'StatusBar' })
      expect(statusBar.props('episode')).toBe(10)
      expect(statusBar.props('epoch')).toBe(200)
      expect(statusBar.props('batch')).toBe(25)
    })
  })

  describe('EnvironmentGrid Props', () => {
    it('passes envs and focusedEnvId to EnvironmentGrid', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot({ focused_env_id: 1 })

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const grid = wrapper.findComponent({ name: 'EnvironmentGrid' })
      expect(grid.exists()).toBe(true)
      expect(grid.props('envs')).toEqual(snapshot.envs)
      expect(grid.props('focusedEnvId')).toBe(1)
    })
  })

  describe('AnomalySidebar Props', () => {
    it('passes event_log to AnomalySidebar', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const sidebar = wrapper.findComponent({ name: 'AnomalySidebar' })
      expect(sidebar.exists()).toBe(true)
      expect(sidebar.props('events')).toEqual(snapshot.event_log)
    })
  })

  describe('LeaderboardTable Props', () => {
    it('passes best_runs to LeaderboardTable', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const leaderboard = wrapper.findComponent({ name: 'LeaderboardTable' })
      expect(leaderboard.exists()).toBe(true)
      expect(leaderboard.props('runs')).toEqual(snapshot.best_runs)
    })
  })

  describe('HealthGauges Props', () => {
    it('passes vitals and tamiyo to HealthGauges', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const gauges = wrapper.findComponent({ name: 'HealthGauges' })
      expect(gauges.exists()).toBe(true)
      expect(gauges.props('vitals')).toEqual(snapshot.vitals)
      expect(gauges.props('tamiyo')).toEqual(snapshot.tamiyo)
    })
  })

  describe('PolicyDiagnostics Props', () => {
    it('passes tamiyo to PolicyDiagnostics', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const diagnostics = wrapper.findComponent({ name: 'PolicyDiagnostics' })
      expect(diagnostics.exists()).toBe(true)
      expect(diagnostics.props('tamiyo')).toEqual(snapshot.tamiyo)
    })
  })

  describe('GradientHeatmap Props', () => {
    it('passes tamiyo to GradientHeatmap', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const heatmap = wrapper.findComponent({ name: 'GradientHeatmap' })
      expect(heatmap.exists()).toBe(true)
      expect(heatmap.props('tamiyo')).toEqual(snapshot.tamiyo)
    })
  })

  describe('EventTimeline Props', () => {
    it('passes event_log to EventTimeline', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const timeline = wrapper.findComponent({ name: 'EventTimeline' })
      expect(timeline.exists()).toBe(true)
      expect(timeline.props('events')).toEqual(snapshot.event_log)
    })
  })

  describe('SeedSwimlane Props', () => {
    it('passes seeds, slotIds, and currentEpoch to SeedSwimlane', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const swimlane = wrapper.findComponent({ name: 'SeedSwimlane' })
      expect(swimlane.exists()).toBe(true)
      // SeedSwimlane uses seeds from focused env
      expect(swimlane.props('slotIds')).toEqual(snapshot.slot_ids)
      expect(swimlane.props('currentEpoch')).toBe(snapshot.current_epoch)
    })
  })

  describe('ContributionWaterfall Props', () => {
    it('passes rewards to ContributionWaterfall', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot()

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const waterfall = wrapper.findComponent({ name: 'ContributionWaterfall' })
      expect(waterfall.exists()).toBe(true)
      expect(waterfall.props('rewards')).toEqual(snapshot.rewards)
    })
  })

  describe('Environment Selection', () => {
    it('updates focused env when EnvironmentGrid emits select', async () => {
      const wrapper = mount(App)
      const snapshot = createSnapshot({ focused_env_id: 0 })

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(snapshot))
      await flushPromises()

      const grid = wrapper.findComponent({ name: 'EnvironmentGrid' })
      await grid.vm.$emit('select', 1)
      await flushPromises()

      // Focused env should update locally
      expect(grid.props('focusedEnvId')).toBe(1)
    })
  })

  describe('Disconnection Handling', () => {
    it('shows disconnected state when WebSocket closes', async () => {
      const wrapper = mount(App)

      const ws = MockWebSocket.instances[0]
      ws.simulateOpen()
      ws.simulateMessage(snapshotMessage(createSnapshot()))
      await flushPromises()

      // Verify connected
      const statusBar = wrapper.findComponent({ name: 'StatusBar' })
      expect(statusBar.props('connectionState')).toBe('connected')

      // Simulate disconnect
      ws.simulateClose()
      await flushPromises()

      expect(statusBar.props('connectionState')).toBe('disconnected')
    })
  })
})
