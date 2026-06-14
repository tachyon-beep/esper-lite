// src/esper/karn/overwatch/web/src/components/__tests__/MorphologyCausalLogPanel.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import MorphologyCausalLogPanel from '../MorphologyCausalLogPanel.vue'
import { createSnapshot, createMorphologyCausalLogEntry } from './factories'

describe('MorphologyCausalLogPanel', () => {
  describe('Empty State', () => {
    it('shows empty state when there are no causal-log entries', () => {
      const snapshot = createSnapshot({ morphology_causal_log: [] })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(true)
      expect(wrapper.text()).toContain('No causal-log entries yet')
    })

    it('hides empty state when entries are present', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [createMorphologyCausalLogEntry()]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(false)
    })
  })

  describe('Rendering causal entries', () => {
    it('renders one item per causal-log entry', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry({ phase: 'proposal' }),
          createMorphologyCausalLogEntry({ phase: 'verdict' }),
          createMorphologyCausalLogEntry({ phase: 'mutation' })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      const items = wrapper.findAll('[data-testid="causal-item"]')
      expect(items.length).toBe(3)
    })

    it('renders the header with the entry count', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry(),
          createMorphologyCausalLogEntry({ phase: 'commit' })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      const header = wrapper.find('[data-testid="causal-header"]')
      expect(header.text()).toContain('Causal Log')
      expect(header.text()).toContain('2')
    })
  })

  describe('Causal / identity chain', () => {
    it('displays the action/proposal/verdict/mutation IDs', () => {
      const entry = createMorphologyCausalLogEntry({
        action_id: 'act-XYZ',
        proposal_id: 'prop-XYZ',
        verdict_id: 'ver-XYZ',
        mutation_id: 'mut-XYZ'
      })
      const snapshot = createSnapshot({ morphology_causal_log: [entry] })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      expect(wrapper.find('[data-testid="action-id"]').text()).toContain('act-XYZ')
      expect(wrapper.find('[data-testid="proposal-id"]').text()).toContain('prop-XYZ')
      expect(wrapper.find('[data-testid="verdict-id"]').text()).toContain('ver-XYZ')
      expect(wrapper.find('[data-testid="mutation-id"]').text()).toContain('mut-XYZ')
    })

    it('displays the phase badge', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [createMorphologyCausalLogEntry({ phase: 'fossilization' })]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      const badge = wrapper.find('[data-testid="phase-badge"]')
      expect(badge.text()).toBe('fossilization')
    })

    it('displays watch-window evidence', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry({ watch_window_evidence: 0.421 })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      expect(wrapper.find('[data-testid="watch-evidence"]').text()).toContain('0.421')
    })

    it('renders an em-dash when watch evidence is null', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry({ watch_window_evidence: null })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      expect(wrapper.find('[data-testid="watch-evidence"]').text()).toContain('—')
    })

    it('displays the linked terminal event id when present', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry({ linked_event_id: 'evt-link-1' })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      const linked = wrapper.find('[data-testid="linked-event"]')
      expect(linked.exists()).toBe(true)
      expect(linked.text()).toContain('evt-link-1')
    })

    it('omits the linked event chip when linked_event_id is null', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry({ linked_event_id: null })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      expect(wrapper.find('[data-testid="linked-event"]').exists()).toBe(false)
    })

    it('displays RNG provenance (stream and seed)', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry({ rng_stream: 'kasmina:prune', rng_seed: 999 })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      const rng = wrapper.find('[data-testid="rng-identity"]')
      expect(rng.text()).toContain('kasmina:prune')
      expect(rng.text()).toContain('999')
    })
  })

  describe('Governor verdict', () => {
    it('shows an approved verdict', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry({ governor_approved: true })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      const verdict = wrapper.find('[data-testid="governor-verdict"]')
      expect(verdict.text()).toBe('approved')
      expect(verdict.classes()).toContain('approved')
    })

    it('shows a blocked verdict with reason', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry({
            governor_approved: false,
            governor_reason: 'watch_evidence_insufficient'
          })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      const verdict = wrapper.find('[data-testid="governor-verdict"]')
      expect(verdict.text()).toBe('blocked')
      expect(verdict.classes()).toContain('blocked')
      expect(wrapper.find('[data-testid="governor-reason"]').text()).toBe(
        'watch_evidence_insufficient'
      )
    })
  })

  describe('Joinability', () => {
    it('keeps a stable action_id across phases so rows join', () => {
      const snapshot = createSnapshot({
        morphology_causal_log: [
          createMorphologyCausalLogEntry({ phase: 'proposal', action_id: 'act-join' }),
          createMorphologyCausalLogEntry({ phase: 'verdict', action_id: 'act-join' }),
          createMorphologyCausalLogEntry({ phase: 'commit', action_id: 'act-join' })
        ]
      })
      const wrapper = mount(MorphologyCausalLogPanel, { props: { snapshot } })

      const actionIds = wrapper.findAll('[data-testid="action-id"]')
      expect(actionIds.length).toBe(3)
      actionIds.forEach((el) => expect(el.text()).toContain('act-join'))
    })
  })
})
