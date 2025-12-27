// src/esper/karn/overwatch/web/src/components/__tests__/AnomalySidebar.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import AnomalySidebar from '../AnomalySidebar.vue'
import type { EventLogEntry } from '../../types/sanctum'

const createEvent = (
  overrides: Partial<EventLogEntry> = {}
): EventLogEntry => ({
  timestamp: '2024-01-15T10:30:00Z',
  event_type: 'epoch_completed',
  env_id: 0,
  message: 'Epoch completed successfully',
  episode: 42,
  relative_time: '5m ago',
  metadata: {},
  ...overrides
})

describe('AnomalySidebar', () => {
  it('filters and displays only anomaly events', () => {
    const events: EventLogEntry[] = [
      createEvent({ event_type: 'epoch_completed', message: 'Epoch done' }),
      createEvent({ event_type: 'anomaly_detected', message: 'Gradient spike detected', relative_time: '2m ago' }),
      createEvent({ event_type: 'seed_spawned', message: 'Seed spawned' }),
      createEvent({ event_type: 'nan_detected', message: 'NaN in loss', relative_time: '1m ago' }),
      createEvent({ event_type: 'gradient_explosion', message: 'Gradients exploded', relative_time: '30s ago' }),
      createEvent({ event_type: 'entropy_collapse', message: 'Entropy collapsed', relative_time: '10s ago' }),
      createEvent({ event_type: 'training_warning', message: 'Training unstable', relative_time: '5s ago' })
    ]

    const wrapper = mount(AnomalySidebar, {
      props: { events }
    })

    const items = wrapper.findAll('[data-testid="anomaly-item"]')
    expect(items.length).toBe(5)

    // Should not include non-anomaly events
    expect(wrapper.text()).not.toContain('Epoch done')
    expect(wrapper.text()).not.toContain('Seed spawned')

    // Should include all anomaly events
    expect(wrapper.text()).toContain('Gradient spike detected')
    expect(wrapper.text()).toContain('NaN in loss')
    expect(wrapper.text()).toContain('Gradients exploded')
    expect(wrapper.text()).toContain('Entropy collapsed')
    expect(wrapper.text()).toContain('Training unstable')
  })

  it('shows empty state when no anomalies', () => {
    const events: EventLogEntry[] = [
      createEvent({ event_type: 'epoch_completed', message: 'Epoch done' }),
      createEvent({ event_type: 'seed_spawned', message: 'Seed spawned' }),
      createEvent({ event_type: 'seed_fossilized', message: 'Seed fossilized' })
    ]

    const wrapper = mount(AnomalySidebar, {
      props: { events }
    })

    expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('No anomalies detected')
  })

  it('displays correct count in header', () => {
    const events: EventLogEntry[] = [
      createEvent({ event_type: 'anomaly_detected', message: 'Anomaly 1' }),
      createEvent({ event_type: 'nan_detected', message: 'Anomaly 2' }),
      createEvent({ event_type: 'gradient_explosion', message: 'Anomaly 3' }),
      createEvent({ event_type: 'epoch_completed', message: 'Not an anomaly' })
    ]

    const wrapper = mount(AnomalySidebar, {
      props: { events }
    })

    const header = wrapper.find('[data-testid="sidebar-header"]')
    expect(header.text()).toContain('Anomalies (3)')
  })

  it('respects maxItems prop', () => {
    const events: EventLogEntry[] = Array.from({ length: 30 }, (_, i) =>
      createEvent({
        event_type: 'anomaly_detected',
        message: `Anomaly ${i + 1}`,
        relative_time: `${i}m ago`
      })
    )

    const wrapper = mount(AnomalySidebar, {
      props: { events, maxItems: 10 }
    })

    const items = wrapper.findAll('[data-testid="anomaly-item"]')
    expect(items.length).toBe(10)
  })

  it('displays env ID badge when env_id is present', () => {
    const events: EventLogEntry[] = [
      createEvent({ event_type: 'anomaly_detected', message: 'Anomaly', env_id: 3 })
    ]

    const wrapper = mount(AnomalySidebar, {
      props: { events }
    })

    const envBadge = wrapper.find('[data-testid="env-badge"]')
    expect(envBadge.exists()).toBe(true)
    expect(envBadge.text()).toContain('3')
  })

  it('shows critical icon for explosion/nan events and warning for others', () => {
    const events: EventLogEntry[] = [
      createEvent({ event_type: 'nan_detected', message: 'NaN detected' }),
      createEvent({ event_type: 'gradient_explosion', message: 'Exploded' }),
      createEvent({ event_type: 'training_warning', message: 'Warning' })
    ]

    const wrapper = mount(AnomalySidebar, {
      props: { events }
    })

    const items = wrapper.findAll('[data-testid="anomaly-item"]')

    // First two should be critical (nan and explosion)
    expect(items[0].find('[data-testid="severity-icon"]').text()).toBe('!!')
    expect(items[1].find('[data-testid="severity-icon"]').text()).toBe('!!')

    // Third should be warning
    expect(items[2].find('[data-testid="severity-icon"]').text()).toBe('!')
  })

  it('displays relative time for each anomaly', () => {
    const events: EventLogEntry[] = [
      createEvent({ event_type: 'anomaly_detected', message: 'Test', relative_time: '2m ago' })
    ]

    const wrapper = mount(AnomalySidebar, {
      props: { events }
    })

    expect(wrapper.find('[data-testid="relative-time"]').text()).toBe('2m ago')
  })
})
