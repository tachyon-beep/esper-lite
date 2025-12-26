// src/esper/karn/overwatch/web/src/composables/__tests__/useKeyboardNav.spec.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { ref, nextTick } from 'vue'
import { useKeyboardNav } from '../useKeyboardNav'

describe('useKeyboardNav', () => {
  let keydownHandler: ((event: KeyboardEvent) => void) | null = null

  beforeEach(() => {
    // Capture the keydown listener
    vi.spyOn(window, 'addEventListener').mockImplementation((type, handler) => {
      if (type === 'keydown') {
        keydownHandler = handler as (event: KeyboardEvent) => void
      }
    })
    vi.spyOn(window, 'removeEventListener').mockImplementation(() => {})
  })

  afterEach(() => {
    keydownHandler = null
    vi.restoreAllMocks()
  })

  function simulateKeydown(key: string, options: Partial<KeyboardEvent> = {}): KeyboardEvent {
    const event = new KeyboardEvent('keydown', {
      key,
      bubbles: true,
      ...options
    })
    vi.spyOn(event, 'preventDefault')
    keydownHandler?.(event)
    return event
  }

  describe('initialization', () => {
    it('registers keydown event listener', () => {
      useKeyboardNav({ envCount: ref(4) })

      expect(window.addEventListener).toHaveBeenCalledWith('keydown', expect.any(Function))
    })

    it('provides helpVisible ref', () => {
      const { helpVisible } = useKeyboardNav({ envCount: ref(4) })

      expect(helpVisible.value).toBe(false)
    })

    it('provides current panel state', () => {
      const { currentPanel } = useKeyboardNav({ envCount: ref(4) })

      expect(currentPanel.value).toBe('main')
    })

    it('provides current leaderboard row state', () => {
      const { currentLeaderboardRow } = useKeyboardNav({ envCount: ref(4), leaderboardRowCount: ref(10) })

      expect(currentLeaderboardRow.value).toBe(0)
    })
  })

  describe('? key - help toggle', () => {
    it('toggles helpVisible to true when pressed', async () => {
      const { helpVisible } = useKeyboardNav({ envCount: ref(4) })

      simulateKeydown('?')
      await nextTick()

      expect(helpVisible.value).toBe(true)
    })

    it('toggles helpVisible to false when pressed again', async () => {
      const { helpVisible } = useKeyboardNav({ envCount: ref(4) })

      simulateKeydown('?')
      await nextTick()
      expect(helpVisible.value).toBe(true)

      simulateKeydown('?')
      await nextTick()
      expect(helpVisible.value).toBe(false)
    })

    it('does not toggle when focused on input element', async () => {
      const { helpVisible } = useKeyboardNav({ envCount: ref(4) })

      const input = document.createElement('input')
      document.body.appendChild(input)
      input.focus()

      // Create event with input as target
      const event = new KeyboardEvent('keydown', { key: '?' })
      Object.defineProperty(event, 'target', { value: input })
      keydownHandler?.(event)
      await nextTick()

      expect(helpVisible.value).toBe(false)

      document.body.removeChild(input)
    })
  })

  describe('Escape key', () => {
    it('closes help overlay when visible', async () => {
      const { helpVisible } = useKeyboardNav({ envCount: ref(4) })

      // Open help first
      helpVisible.value = true
      await nextTick()

      simulateKeydown('Escape')
      await nextTick()

      expect(helpVisible.value).toBe(false)
    })

    it('emits clearSelection when help is not visible', async () => {
      const onClearSelection = vi.fn()
      const { helpVisible } = useKeyboardNav({
        envCount: ref(4),
        onClearSelection
      })

      expect(helpVisible.value).toBe(false)

      simulateKeydown('Escape')
      await nextTick()

      expect(onClearSelection).toHaveBeenCalled()
    })
  })

  describe('number keys 1-9 - environment selection', () => {
    it('emits selectEnv with index 0 when pressing 1', async () => {
      const onSelectEnv = vi.fn()
      useKeyboardNav({
        envCount: ref(4),
        onSelectEnv
      })

      simulateKeydown('1')
      await nextTick()

      expect(onSelectEnv).toHaveBeenCalledWith(0)
    })

    it('emits selectEnv with index 3 when pressing 4', async () => {
      const onSelectEnv = vi.fn()
      useKeyboardNav({
        envCount: ref(4),
        onSelectEnv
      })

      simulateKeydown('4')
      await nextTick()

      expect(onSelectEnv).toHaveBeenCalledWith(3)
    })

    it('does not emit for numbers greater than envCount', async () => {
      const onSelectEnv = vi.fn()
      useKeyboardNav({
        envCount: ref(4),
        onSelectEnv
      })

      simulateKeydown('5')
      await nextTick()

      expect(onSelectEnv).not.toHaveBeenCalled()
    })

    it('handles dynamic envCount changes', async () => {
      const onSelectEnv = vi.fn()
      const envCount = ref(2)
      useKeyboardNav({
        envCount,
        onSelectEnv
      })

      // 3 should not work with envCount=2
      simulateKeydown('3')
      await nextTick()
      expect(onSelectEnv).not.toHaveBeenCalled()

      // Update envCount
      envCount.value = 5
      await nextTick()

      // Now 3 should work
      simulateKeydown('3')
      await nextTick()
      expect(onSelectEnv).toHaveBeenCalledWith(2)
    })

    it('does not select env when help is visible', async () => {
      const onSelectEnv = vi.fn()
      const { helpVisible } = useKeyboardNav({
        envCount: ref(4),
        onSelectEnv
      })

      helpVisible.value = true
      await nextTick()

      simulateKeydown('1')
      await nextTick()

      expect(onSelectEnv).not.toHaveBeenCalled()
    })
  })

  describe('j/k keys - leaderboard navigation', () => {
    it('increments row with j key', async () => {
      const { currentLeaderboardRow } = useKeyboardNav({
        envCount: ref(4),
        leaderboardRowCount: ref(10)
      })

      expect(currentLeaderboardRow.value).toBe(0)

      simulateKeydown('j')
      await nextTick()

      expect(currentLeaderboardRow.value).toBe(1)
    })

    it('decrements row with k key', async () => {
      const { currentLeaderboardRow } = useKeyboardNav({
        envCount: ref(4),
        leaderboardRowCount: ref(10)
      })

      currentLeaderboardRow.value = 5

      simulateKeydown('k')
      await nextTick()

      expect(currentLeaderboardRow.value).toBe(4)
    })

    it('does not go below 0 with k key', async () => {
      const { currentLeaderboardRow } = useKeyboardNav({
        envCount: ref(4),
        leaderboardRowCount: ref(10)
      })

      expect(currentLeaderboardRow.value).toBe(0)

      simulateKeydown('k')
      await nextTick()

      expect(currentLeaderboardRow.value).toBe(0)
    })

    it('does not exceed max rows with j key', async () => {
      const { currentLeaderboardRow } = useKeyboardNav({
        envCount: ref(4),
        leaderboardRowCount: ref(5)
      })

      currentLeaderboardRow.value = 4

      simulateKeydown('j')
      await nextTick()

      expect(currentLeaderboardRow.value).toBe(4)
    })

    it('calls onLeaderboardNavigate callback', async () => {
      const onLeaderboardNavigate = vi.fn()
      useKeyboardNav({
        envCount: ref(4),
        leaderboardRowCount: ref(10),
        onLeaderboardNavigate
      })

      simulateKeydown('j')
      await nextTick()

      expect(onLeaderboardNavigate).toHaveBeenCalledWith(1)
    })
  })

  describe('h/l keys - panel navigation', () => {
    it('navigates left with h key', async () => {
      const { currentPanel } = useKeyboardNav({ envCount: ref(4) })

      // Start at main
      expect(currentPanel.value).toBe('main')

      simulateKeydown('h')
      await nextTick()

      expect(currentPanel.value).toBe('left')
    })

    it('navigates right with l key', async () => {
      const { currentPanel } = useKeyboardNav({ envCount: ref(4) })

      simulateKeydown('l')
      await nextTick()

      expect(currentPanel.value).toBe('right')
    })

    it('wraps from left to right with h key', async () => {
      const { currentPanel } = useKeyboardNav({ envCount: ref(4) })

      currentPanel.value = 'left'

      simulateKeydown('h')
      await nextTick()

      expect(currentPanel.value).toBe('right')
    })

    it('wraps from right to left with l key', async () => {
      const { currentPanel } = useKeyboardNav({ envCount: ref(4) })

      currentPanel.value = 'right'

      simulateKeydown('l')
      await nextTick()

      expect(currentPanel.value).toBe('left')
    })

    it('calls onPanelChange callback', async () => {
      const onPanelChange = vi.fn()
      useKeyboardNav({
        envCount: ref(4),
        onPanelChange
      })

      simulateKeydown('l')
      await nextTick()

      expect(onPanelChange).toHaveBeenCalledWith('right')
    })
  })

  describe('input focus guard', () => {
    it('ignores shortcuts when textarea is focused', async () => {
      const onSelectEnv = vi.fn()
      useKeyboardNav({
        envCount: ref(4),
        onSelectEnv
      })

      const textarea = document.createElement('textarea')
      document.body.appendChild(textarea)
      textarea.focus()

      const event = new KeyboardEvent('keydown', { key: '1' })
      Object.defineProperty(event, 'target', { value: textarea })
      keydownHandler?.(event)
      await nextTick()

      expect(onSelectEnv).not.toHaveBeenCalled()

      document.body.removeChild(textarea)
    })

    it('ignores shortcuts when contenteditable is focused', async () => {
      const onSelectEnv = vi.fn()
      useKeyboardNav({
        envCount: ref(4),
        onSelectEnv
      })

      const div = document.createElement('div')
      div.contentEditable = 'true'
      document.body.appendChild(div)
      div.focus()

      const event = new KeyboardEvent('keydown', { key: '1' })
      Object.defineProperty(event, 'target', { value: div })
      keydownHandler?.(event)
      await nextTick()

      expect(onSelectEnv).not.toHaveBeenCalled()

      document.body.removeChild(div)
    })
  })

  describe('cleanup', () => {
    it('provides cleanup function', () => {
      const { cleanup } = useKeyboardNav({ envCount: ref(4) })

      expect(typeof cleanup).toBe('function')
    })
  })
})
