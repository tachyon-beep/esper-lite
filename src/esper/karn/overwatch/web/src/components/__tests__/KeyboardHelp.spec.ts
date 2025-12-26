// src/esper/karn/overwatch/web/src/components/__tests__/KeyboardHelp.spec.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mount, config, VueWrapper } from '@vue/test-utils'
import KeyboardHelp from '../KeyboardHelp.vue'

describe('KeyboardHelp', () => {
  let wrapper: VueWrapper

  // Configure Teleport stub globally for all tests in this describe block
  beforeEach(() => {
    // Stub Teleport to render in place instead of teleporting to body
    config.global.stubs = {
      Teleport: true
    }
  })

  afterEach(() => {
    wrapper?.unmount()
    config.global.stubs = {}
  })

  it('renders when visible prop is true', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    expect(wrapper.find('[data-testid="keyboard-help-overlay"]').exists()).toBe(true)
  })

  it('does not render when visible prop is false', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: false }
    })

    expect(wrapper.find('[data-testid="keyboard-help-overlay"]').exists()).toBe(false)
  })

  it('displays all global shortcut sections', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    // Check for section headers
    expect(wrapper.text()).toContain('Global')
    expect(wrapper.text()).toContain('Navigation')
  })

  it('displays number key shortcuts for environment selection', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    // Check for 1-9 shortcut description
    expect(wrapper.text()).toContain('1-9')
    expect(wrapper.text()).toContain('Select environment')
  })

  it('displays Escape shortcut for clearing selection', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    expect(wrapper.text()).toContain('Esc')
    expect(wrapper.text()).toContain('Clear selection')
  })

  it('displays ? shortcut for help overlay', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    expect(wrapper.text()).toContain('?')
    expect(wrapper.text()).toContain('Toggle help')
  })

  it('displays j/k shortcuts for vertical navigation', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    expect(wrapper.text()).toContain('j / k')
    expect(wrapper.text()).toContain('Navigate')
  })

  it('displays h/l shortcuts for horizontal navigation', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    expect(wrapper.text()).toContain('h / l')
    expect(wrapper.text()).toContain('panel')
  })

  it('emits close event when clicking backdrop', async () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    await wrapper.find('[data-testid="keyboard-help-backdrop"]').trigger('click')

    expect(wrapper.emitted('close')).toBeTruthy()
    expect(wrapper.emitted('close')).toHaveLength(1)
  })

  it('emits close event when clicking close button', async () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    await wrapper.find('[data-testid="keyboard-help-close"]').trigger('click')

    expect(wrapper.emitted('close')).toBeTruthy()
    expect(wrapper.emitted('close')).toHaveLength(1)
  })

  it('does not emit close when clicking modal content', async () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    await wrapper.find('[data-testid="keyboard-help-modal"]').trigger('click')

    expect(wrapper.emitted('close')).toBeFalsy()
  })

  it('has correct aria attributes for accessibility', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    const overlay = wrapper.find('[data-testid="keyboard-help-overlay"]')
    expect(overlay.attributes('role')).toBe('dialog')
    expect(overlay.attributes('aria-modal')).toBe('true')
    expect(overlay.attributes('aria-labelledby')).toBe('keyboard-help-title')
  })

  it('displays title in modal', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    const title = wrapper.find('#keyboard-help-title')
    expect(title.exists()).toBe(true)
    expect(title.text()).toContain('Keyboard Shortcuts')
  })

  it('renders shortcut keys with visual styling', () => {
    wrapper = mount(KeyboardHelp, {
      props: { visible: true }
    })

    // Check that kbd elements exist for keyboard styling
    const kbdElements = wrapper.findAll('kbd')
    expect(kbdElements.length).toBeGreaterThan(0)
  })
})
