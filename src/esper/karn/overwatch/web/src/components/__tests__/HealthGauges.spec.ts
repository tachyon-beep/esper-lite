// src/esper/karn/overwatch/web/src/components/__tests__/HealthGauges.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import HealthGauges from '../HealthGauges.vue'
import type { SystemVitals, TamiyoState } from '../../types/sanctum'

function createVitals(overrides: Partial<SystemVitals> = {}): SystemVitals {
  return {
    gpu_stats: {},
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
  }
}

function createTamiyo(overrides: Partial<TamiyoState> = {}): TamiyoState {
  return {
    entropy: 0.5,
    clip_fraction: 0.1,
    kl_divergence: 0.01,
    explained_variance: 0.85,
    policy_loss: 0.1,
    value_loss: 0.2,
    entropy_loss: 0.05,
    grad_norm: 1.0,
    learning_rate: 0.0003,
    entropy_coef: 0.01,
    ratio_mean: 1.0,
    ratio_min: 0.9,
    ratio_max: 1.1,
    ratio_std: 0.05,
    advantage_mean: 0.0,
    advantage_std: 1.0,
    advantage_min: -2.0,
    advantage_max: 2.0,
    advantage_raw_mean: 0.0,
    advantage_raw_std: 1.0,
    dead_layers: 0,
    exploding_layers: 0,
    nan_grad_count: 0,
    layer_gradient_health: null,
    entropy_collapsed: false,
    update_time_ms: 100,
    early_stop_epoch: null,
    head_slot_entropy: 0.5,
    head_blueprint_entropy: 0.5,
    head_style_entropy: 0.5,
    head_tempo_entropy: 0.5,
    head_alpha_target_entropy: 0.5,
    head_alpha_speed_entropy: 0.5,
    head_alpha_curve_entropy: 0.5,
    head_op_entropy: 0.5,
    head_slot_grad_norm: 1.0,
    head_blueprint_grad_norm: 1.0,
    head_style_grad_norm: 1.0,
    head_tempo_grad_norm: 1.0,
    head_alpha_target_grad_norm: 1.0,
    head_alpha_speed_grad_norm: 1.0,
    head_alpha_curve_grad_norm: 1.0,
    head_op_grad_norm: 1.0,
    episode_return_history: [],
    current_episode_return: 0,
    current_episode: 0,
    policy_loss_history: [],
    value_loss_history: [],
    grad_norm_history: [],
    entropy_history: [],
    explained_variance_history: [],
    kl_divergence_history: [],
    clip_fraction_history: [],
    inner_epoch: 0,
    ppo_batch: 0,
    action_counts: {},
    total_actions: 0,
    ppo_data_received: true,
    recent_decisions: [],
    group_id: null,
    ...overrides
  }
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
