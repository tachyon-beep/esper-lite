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
})
