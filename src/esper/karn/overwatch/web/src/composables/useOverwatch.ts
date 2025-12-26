import { ref, computed, onUnmounted, type Ref, type ComputedRef } from 'vue'
import type { SanctumSnapshot } from '../types/sanctum'

export type ConnectionState = 'connecting' | 'connected' | 'disconnected'

export interface UseOverwatchReturn {
  snapshot: Ref<SanctumSnapshot | null>
  connectionState: Ref<ConnectionState>
  lastUpdate: Ref<number>
  staleness: ComputedRef<number>
  reconnect: () => void
}

export function useOverwatch(url: string): UseOverwatchReturn {
  const snapshot = ref<SanctumSnapshot | null>(null)
  const connectionState = ref<ConnectionState>('connecting')
  const lastUpdate = ref<number>(0)

  let ws: WebSocket | null = null
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null
  let stalenessInterval: ReturnType<typeof setInterval> | null = null

  // Track staleness reactively
  const now = ref(Date.now())
  stalenessInterval = setInterval(() => {
    now.value = Date.now()
  }, 100)

  const staleness = computed(() => {
    if (lastUpdate.value === 0) return 0
    return now.value - lastUpdate.value
  })

  function connect() {
    connectionState.value = 'connecting'
    ws = new WebSocket(url)

    ws.onopen = () => {
      connectionState.value = 'connected'
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        if (message.type === 'snapshot') {
          snapshot.value = message.data
          lastUpdate.value = Date.now()
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.onclose = () => {
      connectionState.value = 'disconnected'
      // Auto-reconnect after 2 seconds
      reconnectTimeout = setTimeout(connect, 2000)
    }

    ws.onerror = () => {
      ws?.close()
    }
  }

  function reconnect() {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout)
      reconnectTimeout = null
    }
    ws?.close()
    connect()
  }

  // Initial connection
  connect()

  // Cleanup on unmount
  onUnmounted(() => {
    if (reconnectTimeout) clearTimeout(reconnectTimeout)
    if (stalenessInterval) clearInterval(stalenessInterval)
    ws?.close()
  })

  return {
    snapshot,
    connectionState,
    lastUpdate,
    staleness,
    reconnect
  }
}
