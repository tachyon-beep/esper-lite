<!-- src/esper/karn/overwatch/web/src/components/KeyboardHelp.vue -->
<script setup lang="ts">
defineProps<{
  visible: boolean
}>()

const emit = defineEmits<{
  close: []
}>()

interface ShortcutItem {
  keys: string[]
  description: string
}

interface ShortcutSection {
  title: string
  shortcuts: ShortcutItem[]
}

const sections: ShortcutSection[] = [
  {
    title: 'Global',
    shortcuts: [
      { keys: ['1-9'], description: 'Select environment by index' },
      { keys: ['Esc'], description: 'Clear selection / close modal' },
      { keys: ['?'], description: 'Toggle help overlay' }
    ]
  },
  {
    title: 'Navigation',
    shortcuts: [
      { keys: ['j / k'], description: 'Navigate down/up in leaderboard' },
      { keys: ['h / l'], description: 'Navigate between panels' },
      { keys: ['Tab'], description: 'Move focus forward' },
      { keys: ['Shift', 'Tab'], description: 'Move focus backward' }
    ]
  }
]

function handleBackdropClick() {
  emit('close')
}

function handleModalClick(event: MouseEvent) {
  event.stopPropagation()
}

function handleClose() {
  emit('close')
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="visible"
      class="keyboard-help-overlay"
      data-testid="keyboard-help-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="keyboard-help-title"
    >
      <div
        class="backdrop"
        data-testid="keyboard-help-backdrop"
        @click="handleBackdropClick"
      />
      <div
        class="modal"
        data-testid="keyboard-help-modal"
        @click="handleModalClick"
      >
        <header class="modal-header">
          <h2 id="keyboard-help-title" class="modal-title">Keyboard Shortcuts</h2>
          <button
            class="close-button"
            data-testid="keyboard-help-close"
            type="button"
            aria-label="Close keyboard shortcuts help"
            @click="handleClose"
          >
            <span aria-hidden="true">&times;</span>
          </button>
        </header>

        <div class="modal-body">
          <section
            v-for="section in sections"
            :key="section.title"
            class="shortcut-section"
          >
            <h3 class="section-title">{{ section.title }}</h3>
            <ul class="shortcut-list">
              <li
                v-for="shortcut in section.shortcuts"
                :key="shortcut.description"
                class="shortcut-item"
              >
                <span class="shortcut-keys">
                  <kbd
                    v-for="(key, index) in shortcut.keys"
                    :key="key"
                  >{{ key }}<template v-if="index < shortcut.keys.length - 1"> + </template></kbd>
                </span>
                <span class="shortcut-description">{{ shortcut.description }}</span>
              </li>
            </ul>
          </section>
        </div>

        <footer class="modal-footer">
          <span class="footer-hint">Press <kbd>?</kbd> or <kbd>Esc</kbd> to close</span>
        </footer>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.keyboard-help-overlay {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.backdrop {
  position: absolute;
  inset: 0;
  background: rgba(5, 8, 15, 0.9);
  backdrop-filter: blur(4px);
}

.modal {
  position: relative;
  width: 90%;
  max-width: 480px;
  background: var(--bg-panel);
  border: 1px solid var(--border-glow);
  border-radius: 8px;
  box-shadow:
    0 0 40px rgba(0, 229, 255, 0.1),
    0 8px 32px rgba(0, 0, 0, 0.4);
  animation: modal-enter 0.2s ease-out;
}

@keyframes modal-enter {
  from {
    opacity: 0;
    transform: scale(0.95) translateY(-10px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md) var(--space-lg);
  border-bottom: 1px solid var(--border-subtle);
}

.modal-title {
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 1px;
  color: var(--glow-cyan);
  margin: 0;
}

.close-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  background: transparent;
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  color: var(--text-secondary);
  font-size: 20px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.close-button:hover {
  border-color: var(--glow-cyan);
  color: var(--glow-cyan);
  background: var(--glow-cyan-dim);
}

.modal-body {
  padding: var(--space-lg);
  display: flex;
  flex-direction: column;
  gap: var(--space-lg);
}

.shortcut-section {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}

.section-title {
  font-family: var(--font-display);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text-secondary);
  margin: 0;
  padding-bottom: var(--space-xs);
  border-bottom: 1px solid var(--border-subtle);
}

.shortcut-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}

.shortcut-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-md);
}

.shortcut-keys {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
}

kbd {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  height: 24px;
  padding: 0 var(--space-sm);
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--text-bright);
  white-space: nowrap;
}

.shortcut-description {
  font-size: 13px;
  color: var(--text-primary);
  text-align: right;
}

.modal-footer {
  padding: var(--space-sm) var(--space-lg);
  border-top: 1px solid var(--border-subtle);
  text-align: center;
}

.footer-hint {
  font-size: 11px;
  color: var(--text-secondary);
}

.footer-hint kbd {
  font-size: 10px;
  height: 20px;
  min-width: 20px;
}
</style>
