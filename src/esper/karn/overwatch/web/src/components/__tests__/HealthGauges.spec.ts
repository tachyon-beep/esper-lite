// src/esper/karn/overwatch/web/src/components/__tests__/HealthGauges.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import HealthGauges from '../HealthGauges.vue'
import type { SystemVitals, TamiyoState } from '../../types/sanctum'
import {
  createSystemVitals,
  createTamiyoState
} from './factories'

function createVitals(overrides: Partial<SystemVitals> = {}): SystemVitals {
  return createSystemVitals({
    gpu_memory_used_gb: 8,
    gpu_memory_total_gb: 16,
    gpu_utilization: 75,
    gpu_temperature: 65,
    cpu_percent: 50,
    ram_used_gb: 16,
    ram_total_gb: 32,
    epochs_per_second: 1.5,
    batches_per_hour: 120,
    host_params: 100000,
    ...overrides
  })
}

function createTamiyo(overrides: Partial<TamiyoState> = {}): TamiyoState {
  return createTamiyoState({
    entropy: 0.5,
    clip_fraction: 0.1,
    explained_variance: 0.85,
    ...overrides
  })
}

describe('HealthGauges', () => {
  it('renders all gauge elements', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo()
      }
    })

    // Should have 5 gauges: GPU Util, GPU Memory, Entropy, Clip Fraction, Explained Variance
    expect(wrapper.find('[data-testid="gauge-gpu-util"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="gauge-gpu-memory"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="gauge-entropy"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="gauge-clip-fraction"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="gauge-explained-variance"]').exists()).toBe(true)

    // Each gauge should have an SVG with circles
    const gaugeContainers = wrapper.findAll('.gauge')
    expect(gaugeContainers.length).toBe(5)

    for (const gauge of gaugeContainers) {
      expect(gauge.find('svg').exists()).toBe(true)
      expect(gauge.find('circle').exists()).toBe(true)
    }
  })

  it('shows correct percentage for GPU utilization', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals({ gpu_utilization: 85 }),
        tamiyo: createTamiyo()
      }
    })

    const gpuGauge = wrapper.find('[data-testid="gauge-gpu-util"]')
    expect(gpuGauge.find('[data-testid="gauge-value"]').text()).toBe('85%')
  })

  it('shows correct percentage for GPU memory', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals({
          gpu_memory_used_gb: 12,
          gpu_memory_total_gb: 16
        }),
        tamiyo: createTamiyo()
      }
    })

    const memGauge = wrapper.find('[data-testid="gauge-gpu-memory"]')
    // 12/16 = 75%
    expect(memGauge.find('[data-testid="gauge-value"]').text()).toBe('75%')
  })

  it('colors gauge correctly based on health threshold - healthy GPU', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals({ gpu_utilization: 50, gpu_temperature: 60 }),
        tamiyo: createTamiyo()
      }
    })

    const gpuGauge = wrapper.find('[data-testid="gauge-gpu-util"]')
    expect(gpuGauge.classes()).toContain('health-good')
  })

  it('colors gauge with warning when GPU temperature exceeds 80C', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals({ gpu_temperature: 85 }),
        tamiyo: createTamiyo()
      }
    })

    const gpuGauge = wrapper.find('[data-testid="gauge-gpu-util"]')
    expect(gpuGauge.classes()).toContain('health-warning')
  })

  it('colors entropy gauge with warning when entropy is below 0.1', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ entropy: 0.05 })
      }
    })

    const entropyGauge = wrapper.find('[data-testid="gauge-entropy"]')
    expect(entropyGauge.classes()).toContain('health-critical')
  })

  it('colors clip fraction gauge with warning when above 0.2', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ clip_fraction: 0.3 })
      }
    })

    const clipGauge = wrapper.find('[data-testid="gauge-clip-fraction"]')
    expect(clipGauge.classes()).toContain('health-warning')
  })

  it('colors clip fraction gauge with critical when above 0.4', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ clip_fraction: 0.5 })
      }
    })

    const clipGauge = wrapper.find('[data-testid="gauge-clip-fraction"]')
    expect(clipGauge.classes()).toContain('health-critical')
  })

  it('colors explained variance gauge correctly for good values', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ explained_variance: 0.9 })
      }
    })

    const evGauge = wrapper.find('[data-testid="gauge-explained-variance"]')
    expect(evGauge.classes()).toContain('health-good')
  })

  it('colors explained variance gauge with warning for low values', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ explained_variance: 0.3 })
      }
    })

    const evGauge = wrapper.find('[data-testid="gauge-explained-variance"]')
    expect(evGauge.classes()).toContain('health-warning')
  })

  it('displays labels for each gauge', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo()
      }
    })

    expect(wrapper.find('[data-testid="gauge-gpu-util"] [data-testid="gauge-label"]').text()).toBe('GPU')
    expect(wrapper.find('[data-testid="gauge-gpu-memory"] [data-testid="gauge-label"]').text()).toBe('VRAM')
    expect(wrapper.find('[data-testid="gauge-entropy"] [data-testid="gauge-label"]').text()).toBe('Entropy')
    expect(wrapper.find('[data-testid="gauge-clip-fraction"] [data-testid="gauge-label"]').text()).toBe('Clip')
    expect(wrapper.find('[data-testid="gauge-explained-variance"] [data-testid="gauge-label"]').text()).toBe('ExpVar')
  })

  it('shows temperature indicator when GPU is hot', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals({ gpu_temperature: 85 }),
        tamiyo: createTamiyo()
      }
    })

    const tempIndicator = wrapper.find('[data-testid="temp-warning"]')
    expect(tempIndicator.exists()).toBe(true)
    expect(tempIndicator.text()).toContain('85')
  })

  it('does not show temperature indicator when GPU is cool', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals({ gpu_temperature: 65 }),
        tamiyo: createTamiyo()
      }
    })

    expect(wrapper.find('[data-testid="temp-warning"]').exists()).toBe(false)
  })

  it('renders GPU gauges as pending (—) when GPU stats not yet sampled', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        // gpu_data_present false, but the unmeasured-zero fields are present
        vitals: createVitals({ gpu_data_present: false, gpu_utilization: 0, gpu_memory_total_gb: 0 }),
        tamiyo: createTamiyo()
      }
    })

    const gpuGauge = wrapper.find('[data-testid="gauge-gpu-util"]')
    expect(gpuGauge.find('[data-testid="gauge-value"]').text()).toBe('—')
    expect(gpuGauge.classes()).toContain('gauge-pending')
    // Pending must not render a false-healthy/critical alarm.
    expect(gpuGauge.classes()).not.toContain('health-critical')

    const memGauge = wrapper.find('[data-testid="gauge-gpu-memory"]')
    expect(memGauge.find('[data-testid="gauge-value"]').text()).toBe('—')
  })

  it('renders PPO gauges as pending (—) before the first PPO update', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        // No PPO data yet: entropy default 0.0 would otherwise read as critical.
        tamiyo: createTamiyo({ ppo_data_received: false, entropy: 0 })
      }
    })

    const entropyGauge = wrapper.find('[data-testid="gauge-entropy"]')
    expect(entropyGauge.find('[data-testid="gauge-value"]').text()).toBe('—')
    expect(entropyGauge.classes()).toContain('gauge-pending')
    expect(entropyGauge.classes()).not.toContain('health-critical')
  })

  it('does not mask a genuinely negative explained variance as 0% (no display clamp)', () => {
    // EV-telemetry-robustness: the old Math.max(0, ...) clamp rendered a blown-out
    // negative EV as "0%", hiding the collapse. The honest gauge must show the true value.
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ explained_variance: -8.0 })
      }
    })

    const evGauge = wrapper.find('[data-testid="gauge-explained-variance"]')
    const valueText = evGauge.find('[data-testid="gauge-value"]').text()
    // Must NOT be masked to 0%.
    expect(valueText).not.toBe('0%')
    // True value surfaced (rounded -800%), and the status is honestly critical.
    expect(valueText).toBe('-800%')
    expect(evGauge.classes()).toContain('health-critical')
  })

  it('clamps only the EV progress arc geometry (never negative), keeping SVG valid', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ explained_variance: -8.0 })
      }
    })

    const evGauge = wrapper.find('[data-testid="gauge-explained-variance"]')
    const dash = evGauge.find('.gauge-progress').attributes('stroke-dasharray') ?? ''
    const progress = parseFloat(dash.split(' ')[0])
    // Arc geometry must be non-negative (negative dash length is invalid SVG).
    expect(progress).toBeGreaterThanOrEqual(0)
  })

  it('renders a low-return-variance badge when ev_low_return_variance is set', () => {
    // EV-telemetry-robustness: when the EV denominator is ill-conditioned, a deeply negative
    // EV is a denominator artifact, not a collapse. Surface a badge so the gauge is not read
    // as a real failure.
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ explained_variance: -8.0, ev_low_return_variance: true })
      }
    })

    expect(wrapper.find('[data-testid="ev-low-variance-badge"]').exists()).toBe(true)
  })

  it('does not style flagged low-return-variance EV as critical', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ explained_variance: -8.0, ev_low_return_variance: true })
      }
    })

    const evGauge = wrapper.find('[data-testid="gauge-explained-variance"]')
    expect(evGauge.find('[data-testid="gauge-value"]').text()).toBe('-800%')
    expect(wrapper.find('[data-testid="ev-low-variance-badge"]').exists()).toBe(true)
    expect(evGauge.classes()).not.toContain('health-critical')
  })

  it('does not render the low-return-variance badge when the flag is clear', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals(),
        tamiyo: createTamiyo({ explained_variance: 0.85, ev_low_return_variance: false })
      }
    })

    expect(wrapper.find('[data-testid="ev-low-variance-badge"]').exists()).toBe(false)
  })

  it('renders a measured GPU utilization of 0% (not pending) when present', () => {
    const wrapper = mount(HealthGauges, {
      props: {
        vitals: createVitals({ gpu_data_present: true, gpu_utilization: 0 }),
        tamiyo: createTamiyo()
      }
    })

    const gpuGauge = wrapper.find('[data-testid="gauge-gpu-util"]')
    // A genuinely measured 0 must render "0%", never pending.
    expect(gpuGauge.find('[data-testid="gauge-value"]').text()).toBe('0%')
    expect(gpuGauge.classes()).not.toContain('gauge-pending')
  })
})
