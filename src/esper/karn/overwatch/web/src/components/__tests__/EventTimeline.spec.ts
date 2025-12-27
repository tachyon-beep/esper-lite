// src/esper/karn/overwatch/web/src/components/__tests__/EventTimeline.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import EventTimeline from '../EventTimeline.vue'
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

describe('EventTimeline', () => {
  describe('Basic Rendering', () => {
    it('renders timeline container with correct structure', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Seed germinated' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      expect(wrapper.find('[data-testid="event-timeline"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="timeline-container"]').exists()).toBe(true)
    })

    it('renders all events as timeline items', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Seed 1 germinated' }),
        createEvent({ event_type: 'seed_fossilized', message: 'Seed 2 fossilized' }),
        createEvent({ event_type: 'episode_complete', message: 'Episode done' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const items = wrapper.findAll('[data-testid="timeline-item"]')
      expect(items.length).toBe(3)
    })

    it('displays events in order (newest first)', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'First event', relative_time: '10m ago' }),
        createEvent({ event_type: 'seed_fossilized', message: 'Second event', relative_time: '5m ago' }),
        createEvent({ event_type: 'episode_complete', message: 'Third event', relative_time: '1m ago' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const items = wrapper.findAll('[data-testid="timeline-item"]')
      // Events should render in original order (newest first is up to caller to sort)
      expect(items[0].text()).toContain('First event')
      expect(items[2].text()).toContain('Third event')
    })
  })

  describe('Empty State', () => {
    it('shows empty state when no events provided', () => {
      const wrapper = mount(EventTimeline, {
        props: { events: [] }
      })

      expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(true)
      expect(wrapper.text()).toContain('No events yet')
    })

    it('hides empty state when events are present', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Seed germinated' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(false)
    })
  })

  describe('Event Type Icons and Colors', () => {
    it('displays germinated icon for seed_germinated events', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Seed germinated' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const icon = wrapper.find('[data-testid="event-icon"]')
      expect(icon.text()).toContain('🌱')
      expect(icon.classes()).toContain('event-type-seed_germinated')
    })

    it('displays fossilized icon for seed_fossilized events', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_fossilized', message: 'Seed fossilized' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const icon = wrapper.find('[data-testid="event-icon"]')
      expect(icon.text()).toContain('🪨')
      expect(icon.classes()).toContain('event-type-seed_fossilized')
    })

    it('displays pruned icon for seed_pruned events', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_pruned', message: 'Seed pruned' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const icon = wrapper.find('[data-testid="event-icon"]')
      expect(icon.text()).toContain('✂️')
      expect(icon.classes()).toContain('event-type-seed_pruned')
    })

    it('displays anomaly icon for anomaly_detected events', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'anomaly_detected', message: 'Anomaly detected' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const icon = wrapper.find('[data-testid="event-icon"]')
      expect(icon.text()).toContain('⚠️')
      expect(icon.classes()).toContain('event-type-anomaly_detected')
    })

    it('displays episode icon for episode_complete events', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'episode_complete', message: 'Episode complete' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const icon = wrapper.find('[data-testid="event-icon"]')
      expect(icon.text()).toContain('🏁')
      expect(icon.classes()).toContain('event-type-episode_complete')
    })

    it('displays ppo icon for ppo_update events', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'ppo_update', message: 'PPO update' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const icon = wrapper.find('[data-testid="event-icon"]')
      expect(icon.text()).toContain('🧠')
      expect(icon.classes()).toContain('event-type-ppo_update')
    })

    it('displays default icon for unknown event types', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'unknown_event', message: 'Unknown event' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const icon = wrapper.find('[data-testid="event-icon"]')
      expect(icon.text()).toContain('📋')
      expect(icon.classes()).toContain('event-type-default')
    })
  })

  describe('Event Card Content', () => {
    it('displays relative time', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test', relative_time: '2m ago' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const time = wrapper.find('[data-testid="relative-time"]')
      expect(time.exists()).toBe(true)
      expect(time.text()).toBe('2m ago')
    })

    it('displays event type badge', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const badge = wrapper.find('[data-testid="event-type-badge"]')
      expect(badge.exists()).toBe(true)
      expect(badge.text()).toBe('seed_germinated')
    })

    it('displays message text', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Seed slot_0 germinated successfully' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const message = wrapper.find('[data-testid="event-message"]')
      expect(message.exists()).toBe(true)
      expect(message.text()).toBe('Seed slot_0 germinated successfully')
    })

    it('displays env ID chip when env_id is present', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test', env_id: 3 })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const envChip = wrapper.find('[data-testid="env-chip"]')
      expect(envChip.exists()).toBe(true)
      expect(envChip.text()).toContain('Env 3')
    })

    it('does not display env ID chip when env_id is null', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test', env_id: null })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      expect(wrapper.find('[data-testid="env-chip"]').exists()).toBe(false)
    })
  })

  describe('Scrollable Container', () => {
    it('applies default maxHeight when not specified', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const container = wrapper.find('[data-testid="timeline-container"]')
      expect(container.attributes('style')).toContain('max-height')
    })

    it('applies custom maxHeight when specified', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events, maxHeight: '500px' }
      })

      const container = wrapper.find('[data-testid="timeline-container"]')
      expect(container.attributes('style')).toContain('500px')
    })
  })

  describe('Filter Support', () => {
    it('shows all events when no filter is provided', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Germinated' }),
        createEvent({ event_type: 'seed_fossilized', message: 'Fossilized' }),
        createEvent({ event_type: 'anomaly_detected', message: 'Anomaly' }),
        createEvent({ event_type: 'episode_complete', message: 'Episode' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const items = wrapper.findAll('[data-testid="timeline-item"]')
      expect(items.length).toBe(4)
    })

    it('filters events when filter prop is provided', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Germinated' }),
        createEvent({ event_type: 'seed_fossilized', message: 'Fossilized' }),
        createEvent({ event_type: 'anomaly_detected', message: 'Anomaly' }),
        createEvent({ event_type: 'episode_complete', message: 'Episode' })
      ]

      const wrapper = mount(EventTimeline, {
        props: {
          events,
          filter: ['seed_germinated', 'seed_fossilized']
        }
      })

      const items = wrapper.findAll('[data-testid="timeline-item"]')
      expect(items.length).toBe(2)
      expect(wrapper.text()).toContain('Germinated')
      expect(wrapper.text()).toContain('Fossilized')
      expect(wrapper.text()).not.toContain('Anomaly')
      expect(wrapper.text()).not.toContain('Episode')
    })

    it('shows empty state when filter matches no events', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Germinated' }),
        createEvent({ event_type: 'seed_fossilized', message: 'Fossilized' })
      ]

      const wrapper = mount(EventTimeline, {
        props: {
          events,
          filter: ['anomaly_detected']
        }
      })

      expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(true)
    })

    it('supports single event type in filter', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Germinated' }),
        createEvent({ event_type: 'seed_fossilized', message: 'Fossilized' }),
        createEvent({ event_type: 'ppo_update', message: 'PPO' })
      ]

      const wrapper = mount(EventTimeline, {
        props: {
          events,
          filter: ['ppo_update']
        }
      })

      const items = wrapper.findAll('[data-testid="timeline-item"]')
      expect(items.length).toBe(1)
      expect(wrapper.text()).toContain('PPO')
    })
  })

  describe('Timeline Visual Elements', () => {
    it('renders timeline line element', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      expect(wrapper.find('[data-testid="timeline-line"]').exists()).toBe(true)
    })

    it('renders timeline header with event count', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test 1' }),
        createEvent({ event_type: 'seed_fossilized', message: 'Test 2' }),
        createEvent({ event_type: 'ppo_update', message: 'Test 3' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const header = wrapper.find('[data-testid="timeline-header"]')
      expect(header.exists()).toBe(true)
      expect(header.text()).toContain('Events')
      expect(header.text()).toContain('3')
    })

    it('shows filtered count when filter is active', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test 1' }),
        createEvent({ event_type: 'seed_fossilized', message: 'Test 2' }),
        createEvent({ event_type: 'ppo_update', message: 'Test 3' })
      ]

      const wrapper = mount(EventTimeline, {
        props: {
          events,
          filter: ['seed_germinated']
        }
      })

      const header = wrapper.find('[data-testid="timeline-header"]')
      expect(header.text()).toContain('1')
    })
  })

  describe('Accessibility', () => {
    it('timeline container is scrollable', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      const container = wrapper.find('[data-testid="timeline-container"]')
      // Check that overflow styling is applied via class or inline style
      expect(container.classes()).toContain('timeline-scroll')
    })

    it('renders with semantic list structure', () => {
      const events: EventLogEntry[] = [
        createEvent({ event_type: 'seed_germinated', message: 'Test 1' }),
        createEvent({ event_type: 'seed_fossilized', message: 'Test 2' })
      ]

      const wrapper = mount(EventTimeline, {
        props: { events }
      })

      // Uses ul/li for accessibility
      expect(wrapper.find('ul[data-testid="timeline-list"]').exists()).toBe(true)
      expect(wrapper.findAll('li[data-testid="timeline-item"]').length).toBe(2)
    })
  })
})
