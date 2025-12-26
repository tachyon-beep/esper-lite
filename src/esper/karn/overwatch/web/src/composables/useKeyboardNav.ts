// src/esper/karn/overwatch/web/src/composables/useKeyboardNav.ts
import { ref, onUnmounted, type Ref } from 'vue'

export type PanelPosition = 'left' | 'main' | 'right'

export interface UseKeyboardNavOptions {
  /** Number of environments available for selection (1-9) */
  envCount: Ref<number>
  /** Number of rows in the leaderboard table */
  leaderboardRowCount?: Ref<number>
  /** Callback when environment is selected via number keys */
  onSelectEnv?: (envIndex: number) => void
  /** Callback when selection is cleared via Escape */
  onClearSelection?: () => void
  /** Callback when leaderboard row changes via j/k */
  onLeaderboardNavigate?: (rowIndex: number) => void
  /** Callback when leaderboard row is selected via Enter */
  onLeaderboardSelect?: (rowIndex: number) => void
  /** Callback when panel focus changes via h/l */
  onPanelChange?: (panel: PanelPosition) => void
}

export interface UseKeyboardNavReturn {
  /** Whether the help overlay is visible */
  helpVisible: Ref<boolean>
  /** Current focused panel */
  currentPanel: Ref<PanelPosition>
  /** Current focused leaderboard row index */
  currentLeaderboardRow: Ref<number>
  /** Toggle help visibility */
  toggleHelp: () => void
  /** Close help overlay */
  closeHelp: () => void
  /** Cleanup function to remove event listeners */
  cleanup: () => void
}

const PANELS: PanelPosition[] = ['left', 'main', 'right']

/**
 * Composable for centralized keyboard navigation handling.
 *
 * Global shortcuts:
 * - 1-9: Select environment by index
 * - Escape: Clear selection / close modal
 * - ?: Toggle keyboard help overlay
 *
 * Navigation shortcuts:
 * - j/k: Navigate up/down in leaderboard
 * - h/l: Navigate left/right between panels
 */
export function useKeyboardNav(options: UseKeyboardNavOptions): UseKeyboardNavReturn {
  const {
    envCount,
    leaderboardRowCount = ref(0),
    onSelectEnv,
    onClearSelection,
    onLeaderboardNavigate,
    onLeaderboardSelect,
    onPanelChange
  } = options

  const helpVisible = ref(false)
  const currentPanel = ref<PanelPosition>('main')
  const currentLeaderboardRow = ref(0)

  function toggleHelp(): void {
    helpVisible.value = !helpVisible.value
  }

  function closeHelp(): void {
    helpVisible.value = false
  }

  /**
   * Check if the event target is an editable element that should capture keyboard input.
   */
  function isEditableElement(target: EventTarget | null): boolean {
    if (!target || !(target instanceof HTMLElement)) return false

    const tagName = target.tagName.toLowerCase()
    if (tagName === 'input' || tagName === 'textarea' || tagName === 'select') {
      return true
    }

    // Check both isContentEditable property and contentEditable attribute
    // isContentEditable is the computed property, contentEditable is the attribute
    if (target.isContentEditable || target.contentEditable === 'true') {
      return true
    }

    return false
  }

  /**
   * Handle panel navigation with h/l keys.
   * Wraps around: left -> main -> right -> left
   */
  function navigatePanel(direction: 'left' | 'right'): void {
    const currentIndex = PANELS.indexOf(currentPanel.value)
    let newIndex: number

    if (direction === 'left') {
      newIndex = currentIndex === 0 ? PANELS.length - 1 : currentIndex - 1
    } else {
      newIndex = currentIndex === PANELS.length - 1 ? 0 : currentIndex + 1
    }

    currentPanel.value = PANELS[newIndex]
    onPanelChange?.(currentPanel.value)
  }

  /**
   * Handle leaderboard row navigation with j/k keys.
   */
  function navigateLeaderboard(direction: 'up' | 'down'): void {
    const maxRow = Math.max(0, leaderboardRowCount.value - 1)

    if (direction === 'down') {
      currentLeaderboardRow.value = Math.min(currentLeaderboardRow.value + 1, maxRow)
    } else {
      currentLeaderboardRow.value = Math.max(currentLeaderboardRow.value - 1, 0)
    }

    onLeaderboardNavigate?.(currentLeaderboardRow.value)
  }

  /**
   * Main keydown event handler.
   */
  function handleKeydown(event: KeyboardEvent): void {
    // Ignore if focused on editable elements
    if (isEditableElement(event.target)) {
      return
    }

    const { key } = event

    // Help toggle - works regardless of helpVisible state
    if (key === '?') {
      event.preventDefault()
      toggleHelp()
      return
    }

    // Escape - close help or clear selection
    if (key === 'Escape') {
      if (helpVisible.value) {
        event.preventDefault()
        closeHelp()
      } else {
        event.preventDefault()
        onClearSelection?.()
      }
      return
    }

    // All other shortcuts are disabled when help is visible
    if (helpVisible.value) {
      return
    }

    // Number keys 1-9 for environment selection
    if (/^[1-9]$/.test(key)) {
      const envIndex = parseInt(key, 10) - 1
      if (envIndex < envCount.value) {
        event.preventDefault()
        onSelectEnv?.(envIndex)
      }
      return
    }

    // j/k for leaderboard navigation
    if (key === 'j') {
      event.preventDefault()
      navigateLeaderboard('down')
      return
    }

    if (key === 'k') {
      event.preventDefault()
      navigateLeaderboard('up')
      return
    }

    // Enter to select current leaderboard row
    if (key === 'Enter') {
      if (currentLeaderboardRow.value >= 0 && currentLeaderboardRow.value < leaderboardRowCount.value) {
        event.preventDefault()
        onLeaderboardSelect?.(currentLeaderboardRow.value)
      }
      return
    }

    // h/l for panel navigation
    if (key === 'h') {
      event.preventDefault()
      navigatePanel('left')
      return
    }

    if (key === 'l') {
      event.preventDefault()
      navigatePanel('right')
      return
    }
  }

  function cleanup(): void {
    window.removeEventListener('keydown', handleKeydown)
  }

  // Register event listener
  window.addEventListener('keydown', handleKeydown)

  // Cleanup on unmount if used in a component
  onUnmounted(() => {
    cleanup()
  })

  return {
    helpVisible,
    currentPanel,
    currentLeaderboardRow,
    toggleHelp,
    closeHelp,
    cleanup
  }
}
