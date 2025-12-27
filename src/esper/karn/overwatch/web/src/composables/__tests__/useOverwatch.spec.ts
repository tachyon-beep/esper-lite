import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useOverwatch } from '../useOverwatch'
import { nextTick } from 'vue'

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: ((error: Event) => void) | null = null
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

describe('useOverwatch', () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    vi.stubGlobal('WebSocket', MockWebSocket)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('starts in connecting state', () => {
    const { connectionState } = useOverwatch('ws://localhost:8080/ws')
    expect(connectionState.value).toBe('connecting')
  })

  it('transitions to connected when WebSocket opens', async () => {
    const { connectionState } = useOverwatch('ws://localhost:8080/ws')

    MockWebSocket.instances[0].simulateOpen()
    await nextTick()

    expect(connectionState.value).toBe('connected')
  })

  it('updates snapshot when message received', async () => {
    const { snapshot } = useOverwatch('ws://localhost:8080/ws')

    const ws = MockWebSocket.instances[0]
    ws.simulateOpen()
    ws.simulateMessage({
      type: 'snapshot',
      data: { current_episode: 42, current_epoch: 10 }
    })
    await nextTick()

    expect(snapshot.value?.current_episode).toBe(42)
    expect(snapshot.value?.current_epoch).toBe(10)
  })

  it('tracks staleness', async () => {
    vi.useFakeTimers()
    const { staleness, lastUpdate } = useOverwatch('ws://localhost:8080/ws')

    const ws = MockWebSocket.instances[0]
    ws.simulateOpen()
    ws.simulateMessage({ type: 'snapshot', data: { current_episode: 1 } })
    await nextTick()

    expect(lastUpdate.value).toBeGreaterThan(0)

    vi.advanceTimersByTime(5000)
    expect(staleness.value).toBeGreaterThanOrEqual(5000)

    vi.useRealTimers()
  })
})
