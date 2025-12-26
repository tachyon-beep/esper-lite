<!-- src/esper/karn/overwatch/web/src/components/ContributionWaterfall.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import type { RewardComponents } from '../types/sanctum'

const props = defineProps<{
  rewards: RewardComponents
}>()

// SVG dimensions and spacing
const WIDTH = 400
const BAR_HEIGHT = 24
const BAR_GAP = 8
const LABEL_WIDTH = 100
const VALUE_WIDTH = 60
const SECTION_GAP = 10 // Gap between label/chart/value sections
const TOP_PADDING = 20 // Padding above first bar
const BOTTOM_PADDING = 20 // Padding below last bar

// Computed layout values
const CHART_LEFT = LABEL_WIDTH + SECTION_GAP
const CHART_WIDTH = WIDTH - CHART_LEFT - VALUE_WIDTH - SECTION_GAP

interface WaterfallBar {
  id: string
  label: string
  value: number
  show: boolean
}

// Define the bars to display in order
// Note: bounded_attribution is intentionally excluded from visualization.
// It represents a clipped summary for normalization, not a separate contribution.
// The individual components (seed_contribution, base_acc_delta) are shown instead.
const bars = computed<WaterfallBar[]>(() => {
  const combinedPenalties =
    props.rewards.alpha_shock +
    props.rewards.ratio_penalty +
    props.rewards.blending_warning +
    props.rewards.holding_warning

  return [
    {
      id: 'base-acc-delta',
      label: 'Base Acc Delta',
      value: props.rewards.base_acc_delta,
      show: true
    },
    {
      id: 'seed-contribution',
      label: 'Seed Contrib',
      value: props.rewards.seed_contribution,
      show: true
    },
    {
      id: 'stage-bonus',
      label: 'Stage Bonus',
      value: props.rewards.stage_bonus,
      show: true
    },
    {
      id: 'fossilize-bonus',
      label: 'Fossilize Bonus',
      value: props.rewards.fossilize_terminal_bonus,
      show: props.rewards.fossilize_terminal_bonus > 0
    },
    {
      id: 'compute-rent',
      label: 'Compute Rent',
      value: props.rewards.compute_rent,
      show: true
    },
    {
      id: 'penalties',
      label: 'Penalties',
      value: combinedPenalties,
      show: true
    },
    {
      id: 'total',
      label: 'Total',
      value: props.rewards.total,
      show: true
    }
  ]
})

// Get visible bars
const visibleBars = computed(() => bars.value.filter(bar => bar.show))

// Calculate scale for bar widths
const scale = computed(() => {
  const allValues = visibleBars.value.map(b => Math.abs(b.value))
  const maxValue = Math.max(...allValues, 0.01) // Prevent division by zero
  return (CHART_WIDTH / 2) / maxValue
})

// Calculate Y position for each bar
function getBarY(index: number): number {
  return TOP_PADDING + index * (BAR_HEIGHT + BAR_GAP)
}

// Calculate bar width based on value
function getBarWidth(value: number): number {
  return Math.max(2, Math.abs(value) * scale.value)
}

// Calculate bar X position (centered at midpoint for waterfall effect)
function getBarX(value: number): number {
  const midpoint = CHART_LEFT + CHART_WIDTH / 2
  if (value >= 0) {
    return midpoint
  }
  return midpoint - getBarWidth(value)
}

// Format value for display
function formatValue(value: number): string {
  const formatted = Math.abs(value).toFixed(2)
  if (value > 0) return `+${formatted}`
  if (value < 0) return `-${formatted}`
  return formatted
}

// Determine bar class based on value
function getBarClass(bar: WaterfallBar): string[] {
  const classes: string[] = []
  if (bar.id === 'total') {
    classes.push('total')
  } else if (bar.value > 0) {
    classes.push('positive')
  } else if (bar.value < 0) {
    classes.push('negative')
  } else {
    classes.push('zero')
  }
  return classes
}

// Calculate SVG height based on number of visible bars
const svgHeight = computed(() => {
  return TOP_PADDING + BOTTOM_PADDING + visibleBars.value.length * (BAR_HEIGHT + BAR_GAP)
})

// Midpoint line X position
const midpointX = computed(() => CHART_LEFT + CHART_WIDTH / 2)
</script>

<template>
  <div class="contribution-waterfall">
    <svg
      :width="WIDTH"
      :height="svgHeight"
      :viewBox="`0 0 ${WIDTH} ${svgHeight}`"
      class="waterfall-svg"
    >
      <!-- Zero/midpoint line -->
      <line
        :x1="midpointX"
        y1="10"
        :x2="midpointX"
        :y2="svgHeight - 10"
        class="midpoint-line"
      />

      <!-- Bars and labels -->
      <g
        v-for="(bar, index) in visibleBars"
        :key="bar.id"
        :data-testid="`bar-${bar.id}`"
        :class="['bar-group', ...getBarClass(bar)]"
      >
        <!-- Bar label (left side) -->
        <text
          :x="CHART_LEFT - 10"
          :y="getBarY(index) + BAR_HEIGHT / 2"
          class="bar-label"
          text-anchor="end"
          dominant-baseline="middle"
        >
          {{ bar.label }}
        </text>

        <!-- The bar itself -->
        <rect
          :x="getBarX(bar.value)"
          :y="getBarY(index)"
          :width="getBarWidth(bar.value)"
          :height="BAR_HEIGHT"
          :class="['bar-rect', ...getBarClass(bar)]"
          rx="2"
        />

        <!-- Value label (right side) -->
        <text
          :data-testid="`label-${bar.id}`"
          :x="WIDTH - VALUE_WIDTH + 5"
          :y="getBarY(index) + BAR_HEIGHT / 2"
          :class="['value-label', ...getBarClass(bar)]"
          text-anchor="start"
          dominant-baseline="middle"
        >
          {{ formatValue(bar.value) }}
        </text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.contribution-waterfall {
  padding: var(--space-sm);
  background: var(--bg-panel);
  border-radius: 4px;
}

.waterfall-svg {
  display: block;
}

.midpoint-line {
  stroke: var(--border-subtle);
  stroke-width: 1;
  stroke-dasharray: 4 2;
}

.bar-label {
  font-size: 11px;
  fill: var(--text-secondary);
  font-family: var(--font-mono);
}

.value-label {
  font-size: 11px;
  font-family: var(--font-mono);
  font-weight: 600;
}

.value-label.positive {
  fill: var(--status-win);
}

.value-label.negative {
  fill: var(--status-loss);
}

.value-label.total {
  fill: var(--glow-cyan);
}

.value-label.zero {
  fill: var(--text-dim);
}

.bar-rect {
  transition: opacity 0.2s ease;
}

.bar-rect.positive {
  fill: var(--status-win);
  opacity: 0.8;
}

.bar-rect.negative {
  fill: var(--status-loss);
  opacity: 0.8;
}

.bar-rect.total {
  fill: var(--glow-cyan);
  opacity: 0.9;
}

.bar-rect.zero {
  fill: var(--text-dim);
  opacity: 0.5;
}

.bar-group:hover .bar-rect {
  opacity: 1;
}
</style>
