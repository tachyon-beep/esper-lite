// src/esper/karn/overwatch/web/src/components/__tests__/StatusBar.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import StatusBar from '../StatusBar.vue'

describe('StatusBar', () => {
  it('displays connection state', () => {
    const wrapper = mount(StatusBar, {
      props: {
        connectionState: 'connected',
        staleness: 100,
        episode: 42,
        epoch: 10,
        batch: 5
      }
    })

    expect(wrapper.find('[data-testid="connection-status"]').text()).toContain('CONNECTED')
  })

  it('shows staleness warning when stale', () => {
    const wrapper = mount(StatusBar, {
      props: {
        connectionState: 'connected',
        staleness: 5000,  // 5 seconds
        episode: 42,
        epoch: 10,
        batch: 5
      }
    })

    expect(wrapper.find('[data-testid="staleness"]').classes()).toContain('warning')
  })

  it('displays episode and epoch', () => {
    const wrapper = mount(StatusBar, {
      props: {
        connectionState: 'connected',
        staleness: 100,
        episode: 42,
        epoch: 10,
        batch: 5
      }
    })

    expect(wrapper.text()).toContain('Ep 42')
    expect(wrapper.text()).toContain('Epoch 10')
  })
})
