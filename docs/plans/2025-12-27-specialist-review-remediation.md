# Specialist Review Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix bugs, best practice deviations, and implement enhancements identified by UX, DRL, and PyTorch specialist reviews of the `ux-overwatch-refactor` branch.

**Architecture:** Remediate issues in priority order: high-priority bugs first (user-facing), then medium-priority best practices (stability/accessibility), then enhancements. Each task is atomic and independently testable.

**Tech Stack:** Vue 3 / TypeScript (Overwatch web), Python / Textual (TUI), PyTorch (training loop)

---

## Priority Legend

| Priority | Meaning |
|----------|---------|
| **P0** | Bug - fix before merge |
| **P1** | Best practice deviation - fix soon |
| **P2** | Enhancement - nice to have |

---

## Task 1: Fix KeyboardHelp j/k Documentation (P0)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/components/KeyboardHelp.vue:33`

**Step 1: Fix the documentation**

The help says "up/down" but vim-style is "down/up" (j=down, k=up).

Change line 33 from:
```typescript
{ keys: ['j / k'], description: 'Navigate up/down in leaderboard' }
```

To:
```typescript
{ keys: ['j / k'], description: 'Navigate down/up in leaderboard' }
```

**Step 2: Verify test still passes**

Run: `cd src/esper/karn/overwatch/web && npm test -- KeyboardHelp`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/web/src/components/KeyboardHelp.vue
git commit -m "fix(overwatch): correct j/k navigation documentation order"
```

---

## Task 2: Remove Duplicate data-testid Attribute (P0)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/components/EnvironmentGrid.vue:49`

**Step 1: Remove the duplicate attribute**

Delete line 49 which contains `data-testid-base="env-card"` (leftover from earlier implementation).

**Step 2: Run tests**

Run: `cd src/esper/karn/overwatch/web && npm test -- EnvironmentGrid`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/web/src/components/EnvironmentGrid.vue
git commit -m "fix(overwatch): remove duplicate data-testid-base attribute"
```

---

## Task 3: Fix ContributionWaterfall Viewport Overflow (P0)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/components/ContributionWaterfall.vue`

**Step 1: Add container ref and resize observer**

Add imports and refs:
```typescript
import { ref, computed, onMounted, onUnmounted } from 'vue'

const containerRef = ref<HTMLElement | null>(null)
const containerWidth = ref<number>(400)

onMounted(() => {
  if (containerRef.value) {
    containerWidth.value = containerRef.value.clientWidth
    const observer = new ResizeObserver((entries) => {
      containerWidth.value = entries[0].contentRect.width
    })
    observer.observe(containerRef.value)
    onUnmounted(() => observer.disconnect())
  }
})
```

**Step 2: Make WIDTH and CHART_WIDTH reactive**

Change all dependent constants to computed:
```typescript
// Before (constants)
const WIDTH = 400
const CHART_WIDTH = WIDTH - CHART_LEFT - VALUE_WIDTH - SECTION_GAP

// After (computed)
const WIDTH = computed(() => Math.min(400, containerWidth.value))
const CHART_WIDTH = computed(() => WIDTH.value - CHART_LEFT - VALUE_WIDTH - SECTION_GAP)
```

**Step 3: Update scale and midpointX to use .value**

```typescript
const scale = computed(() => {
  const allValues = visibleBars.value.map(b => Math.abs(b.value))
  const maxValue = Math.max(...allValues, 0.01)
  return (CHART_WIDTH.value / 2) / maxValue
})

const midpointX = computed(() => CHART_LEFT + CHART_WIDTH.value / 2)
```

**Step 4: Update getBarX function**

```typescript
function getBarX(value: number): number {
  const midpoint = CHART_LEFT + CHART_WIDTH.value / 2
  return value >= 0 ? midpoint : midpoint + value * scale.value
}
```

**Step 5: Update template**

```html
<div ref="containerRef" class="waterfall-container">
  <svg :width="WIDTH" :height="HEIGHT" ...>
```

**Step 6: Add test**

In `ContributionWaterfall.spec.ts`:
```typescript
it('limits SVG width to container width when narrow', async () => {
  const wrapper = mount(ContributionWaterfall, {
    props: { rewards: mockRewards },
    attachTo: document.createElement('div')
  })
  const svg = wrapper.find('svg')
  const width = parseInt(svg.attributes('width') || '0')
  expect(width).toBeLessThanOrEqual(400)
})
```

**Step 7: Run tests**

Run: `cd src/esper/karn/overwatch/web && npm test -- ContributionWaterfall`
Expected: PASS

**Step 8: Commit**

```bash
git add src/esper/karn/overwatch/web/src/components/ContributionWaterfall.vue
git add src/esper/karn/overwatch/web/src/components/__tests__/ContributionWaterfall.spec.ts
git commit -m "fix(overwatch): make ContributionWaterfall responsive to container width"
```

---

## Task 4: Add Enter Key Selection to LeaderboardTable (P0)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/composables/useKeyboardNav.ts`
- Modify: `src/esper/karn/overwatch/web/src/components/LeaderboardTable.vue`

**Note:** `useKeyboardNav` is a composable that uses callbacks, not Vue emit. Must add callback option.

**Step 1: Add callback option to UseKeyboardNavOptions interface**

In `useKeyboardNav.ts`, update the options interface:
```typescript
export interface UseKeyboardNavOptions {
  // ... existing options
  /** Callback when leaderboard row is selected via Enter */
  onLeaderboardSelect?: (rowIndex: number) => void
}
```

**Step 2: Add Enter key handler in handleKeydown**

In the `handleKeydown` function, add Enter case:
```typescript
case 'Enter':
  if (currentLeaderboardRow.value >= 0 && currentLeaderboardRow.value < leaderboardRowCount.value) {
    event.preventDefault()
    options.onLeaderboardSelect?.(currentLeaderboardRow.value)
  }
  return
```

**Step 3: Update LeaderboardTable to use new callback**

In `LeaderboardTable.vue`, add emit and wire callback:
```typescript
const emit = defineEmits<{
  select: [envId: number]
}>()

// When setting up keyboard nav, pass the callback:
useKeyboardNav({
  // ... existing options
  onLeaderboardSelect: (rowIndex: number) => {
    const env = props.environments[rowIndex]
    if (env) {
      emit('select', env.env_id)
    }
  }
})
```

**Step 4: Add test**

In `useKeyboardNav.spec.ts`:
```typescript
it('calls onLeaderboardSelect callback on Enter key', async () => {
  const onLeaderboardSelect = vi.fn()
  const { currentLeaderboardRow } = useKeyboardNav({
    leaderboardRowCount: ref(5),
    onLeaderboardSelect
  })

  // Select a row first
  currentLeaderboardRow.value = 2

  // Simulate Enter key
  const event = new KeyboardEvent('keydown', { key: 'Enter' })
  document.dispatchEvent(event)

  expect(onLeaderboardSelect).toHaveBeenCalledWith(2)
})
```

**Step 5: Run tests**

Run: `cd src/esper/karn/overwatch/web && npm test -- useKeyboardNav LeaderboardTable`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/overwatch/web/src/composables/useKeyboardNav.ts
git add src/esper/karn/overwatch/web/src/components/LeaderboardTable.vue
git add src/esper/karn/overwatch/web/src/composables/__tests__/useKeyboardNav.spec.ts
git commit -m "feat(overwatch): add Enter key selection to LeaderboardTable"
```

---

## Task 5: Add Connection Status Icons (P1 - Accessibility)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/components/StatusBar.vue`

**Note:** Existing code uses `status-indicator` class, not `connection-indicator`.

**Step 1: Add status icons**

Update the existing status indicator span to include an icon:

```html
<span
  class="status-indicator"
  :class="connectionState"
  data-testid="connection-status"
>
  <span class="status-icon" aria-hidden="true">
    <template v-if="connectionState === 'connecting'">&#8987;</template>
    <template v-else-if="connectionState === 'connected'">&#10003;</template>
    <template v-else>&#10007;</template>
  </span>
  {{ connectionState.toUpperCase() }}
</span>
```

Add CSS:
```css
.status-icon {
  margin-right: 0.25rem;
}
```

**Step 2: Update test**

In `StatusBar.spec.ts`:
```typescript
it('shows icon for each connection state', () => {
  const wrapper = mount(StatusBar, { props: { connectionState: 'connected' } })
  expect(wrapper.find('.status-icon').text()).toContain('✓')
})
```

**Step 3: Run tests**

Run: `cd src/esper/karn/overwatch/web && npm test -- StatusBar`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/karn/overwatch/web/src/components/StatusBar.vue
git add src/esper/karn/overwatch/web/src/components/__tests__/StatusBar.spec.ts
git commit -m "feat(overwatch): add icons to connection status indicator for accessibility"
```

---

## Task 6: Add Exponential Backoff to WebSocket Reconnection (P1)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/composables/useOverwatch.ts`

**Step 1: Add backoff state and helper function**

At the top of the composable, add:
```typescript
let reconnectAttempts = 0
const MAX_BACKOFF = 30000

const getBackoffDelay = () => {
  const backoff = Math.min(MAX_BACKOFF, 2000 * Math.pow(2, reconnectAttempts))
  const jitter = Math.random() * 1000
  return backoff + jitter
}
```

**Step 2: Update reconnection logic to use backoff**

In the error/close handler where reconnection is scheduled:
```typescript
// Replace fixed delay:
// reconnectTimeout = setTimeout(connect, 2000)

// With exponential backoff:
reconnectAttempts++
reconnectTimeout = setTimeout(connect, getBackoffDelay())
```

**Step 3: Reset attempts on successful connection**

In the `ws.onopen` handler, add reset:
```typescript
ws.onopen = () => {
  connectionState.value = 'connected'
  reconnectAttempts = 0  // Reset backoff on successful connection
}
```

**Step 4: Add test**

In `useOverwatch.spec.ts`:
```typescript
it('increases delay between reconnection attempts', async () => {
  vi.useFakeTimers()
  const delays: number[] = []
  const originalSetTimeout = setTimeout
  vi.spyOn(global, 'setTimeout').mockImplementation((fn, delay) => {
    if (delay && delay > 1000) delays.push(delay)
    return originalSetTimeout(fn, 0)
  })

  // Trigger multiple connection failures
  // ...

  // Verify delays increase (with some tolerance for jitter)
  expect(delays[1]).toBeGreaterThan(delays[0])
  vi.useRealTimers()
})
```

**Step 5: Run tests**

Run: `cd src/esper/karn/overwatch/web && npm test -- useOverwatch`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/overwatch/web/src/composables/useOverwatch.ts
git add src/esper/karn/overwatch/web/src/composables/__tests__/useOverwatch.spec.ts
git commit -m "feat(overwatch): add exponential backoff to WebSocket reconnection"
```

---

## Task 7: Fix Temperature Degree Symbol (P1)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/components/HealthGauges.vue:180-183`

**Step 1: Add degree symbol**

Change:
```html
{{ vitals.gpu_temperature }}C
```

To:
```html
{{ vitals.gpu_temperature }}°C
```

**Step 2: Run tests**

Run: `cd src/esper/karn/overwatch/web && npm test -- HealthGauges`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/web/src/components/HealthGauges.vue
git commit -m "fix(overwatch): add degree symbol to temperature display"
```

---

## Task 8: Add Reduced Motion Media Query (P1 - Accessibility)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/styles/theme.css`

**Step 1: Add media query at end of file**

```css
/* Accessibility: Respect user's motion preferences */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

**Step 2: Commit**

```bash
git add src/esper/karn/overwatch/web/src/styles/theme.css
git commit -m "feat(overwatch): add prefers-reduced-motion support for accessibility"
```

---

## Task 9: Fix Hindsight Credit Weight vs Cap Issue (P1 - DRL)

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:2710-2714`
- Modify: `src/esper/leyline/__init__.py` (add new constant)

**Step 1: Add per-scaffold credit weight constant**

In `src/esper/leyline/__init__.py`, add:
```python
# Hindsight credit: per-scaffold weight (allows multiple scaffolds to contribute)
HINDSIGHT_CREDIT_WEIGHT = 0.1  # Half of MAX to allow 2+ scaffolds to meaningfully contribute
```

**Step 2: Update vectorized.py to use new constant**

Change line ~2710:
```python
# Before
raw_credit = compute_scaffold_hindsight_credit(
    boost_given=boost_given,
    beneficiary_improvement=beneficiary_improvement,
    credit_weight=MAX_HINDSIGHT_CREDIT,
)

# After
from esper.leyline import HINDSIGHT_CREDIT_WEIGHT
raw_credit = compute_scaffold_hindsight_credit(
    boost_given=boost_given,
    beneficiary_improvement=beneficiary_improvement,
    credit_weight=HINDSIGHT_CREDIT_WEIGHT,
)
```

**Step 3: Update test to verify multi-scaffold contribution**

In `tests/simic/test_scaffold_hindsight.py`, add:
```python
def test_multiple_scaffolds_can_contribute_meaningfully():
    """Two scaffolds should each contribute, not have second discarded by cap."""
    from esper.leyline import MAX_HINDSIGHT_CREDIT, HINDSIGHT_CREDIT_WEIGHT
    from esper.simic.rewards.rewards import compute_scaffold_hindsight_credit

    # Two identical scaffolds providing boosts
    credit_1 = compute_scaffold_hindsight_credit(
        boost_given=5.0,
        beneficiary_improvement=2.0,
        credit_weight=HINDSIGHT_CREDIT_WEIGHT
    )
    credit_2 = compute_scaffold_hindsight_credit(
        boost_given=5.0,
        beneficiary_improvement=2.0,
        credit_weight=HINDSIGHT_CREDIT_WEIGHT
    )

    # Each should contribute, total should be more than single
    assert credit_1 > 0
    assert credit_2 > 0
    combined = min(credit_1 + credit_2, MAX_HINDSIGHT_CREDIT)
    assert combined > credit_1  # Second scaffold added value
    assert combined <= MAX_HINDSIGHT_CREDIT  # Still capped
```

**Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_scaffold_hindsight.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/__init__.py
git add src/esper/simic/training/vectorized.py
git add tests/simic/test_scaffold_hindsight.py
git commit -m "fix(simic): use separate weight constant for per-scaffold hindsight credit

Fixes issue where credit_weight = MAX_HINDSIGHT_CREDIT caused single scaffold
to saturate cap, discarding contributions from additional scaffolds.

New HINDSIGHT_CREDIT_WEIGHT = 0.1 allows 2+ scaffolds to contribute meaningfully
while still respecting MAX_HINDSIGHT_CREDIT cap."
```

---

## Task 10: Document Hindsight Credit Magic Constants (P2)

**Files:**
- Modify: `src/esper/simic/rewards/rewards.py` (near line 1261)

**Step 1: Add documentation comment**

Above the `compute_scaffold_hindsight_credit` function, add:
```python
def compute_scaffold_hindsight_credit(
    boost_given: float,
    beneficiary_improvement: float,
    credit_weight: float,
) -> float:
    """Compute hindsight credit for a scaffold that helped a beneficiary.

    The 0.1 scaling factor controls tanh saturation:
    - boost * improvement = 10 → tanh(1.0) = 0.76
    - boost * improvement = 50 → tanh(5.0) ≈ 1.0 (saturated)

    This ensures credit saturates gracefully for large contributions
    while remaining sensitive to smaller improvements.

    Args:
        boost_given: Amount of boost the scaffold provided
        beneficiary_improvement: Total improvement of the beneficiary seed
        credit_weight: Maximum credit per scaffold (typically HINDSIGHT_CREDIT_WEIGHT)

    Returns:
        Credit value in range [0, credit_weight]
    """
    raw_credit = math.tanh(boost_given * beneficiary_improvement * 0.1)
    return raw_credit * credit_weight
```

**Step 2: Commit**

```bash
git add src/esper/simic/rewards/rewards.py
git commit -m "docs(simic): document hindsight credit scaling factor rationale"
```

---

## Task 11: Use defaultdict for Scaffold Ledger (P2)

**Files:**
- Modify: `src/esper/simic/training/parallel_env_state.py:93`
- Modify: `src/esper/simic/training/vectorized.py` (ledger append sites + reset logic)

**Step 1: Change ledger type annotation and factory**

In `parallel_env_state.py`:
```python
from collections import defaultdict
from typing import DefaultDict

# Change from:
scaffold_boost_ledger: dict[str, list[tuple[float, str, int]]] = field(default_factory=dict)

# To (note: DefaultDict type annotation, not dict):
scaffold_boost_ledger: DefaultDict[str, list[tuple[float, str, int]]] = field(
    default_factory=lambda: defaultdict(list)
)
```

**Step 2: Simplify append sites in vectorized.py**

Remove the explicit initialization checks:
```python
# Before
if slot_a not in env_state.scaffold_boost_ledger:
    env_state.scaffold_boost_ledger[slot_a] = []
env_state.scaffold_boost_ledger[slot_a].append(...)

# After
env_state.scaffold_boost_ledger[slot_a].append(...)
```

**Step 3: Verify reset uses .clear() not reassignment**

In `vectorized.py`, any reset logic MUST use `.clear()` to preserve defaultdict behavior:
```python
# WRONG - loses defaultdict factory:
env_state.scaffold_boost_ledger = {}

# CORRECT - preserves defaultdict factory:
env_state.scaffold_boost_ledger.clear()
```

Search for any reassignments and fix them to use `.clear()`.

**Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_scaffold_hindsight.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/parallel_env_state.py
git add src/esper/simic/training/vectorized.py
git commit -m "refactor(simic): use defaultdict for scaffold_boost_ledger

Use DefaultDict type annotation and defaultdict factory for cleaner
append logic. Ensures reset uses .clear() to preserve factory behavior."
```

---

## Task 12: Add Seed Count Legend to EnvironmentGrid (P2)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/components/EnvironmentGrid.vue`

**Step 1: Add tooltip to seed counts**

Wrap the seed count display with a title attribute:
```html
<span
  class="seed-counts"
  title="Active / Fossilized / Pruned"
>
  <span class="seed-active">{{ env.active_seed_count }}</span>
  <span class="seed-separator">/</span>
  <span class="seed-fossilized">{{ env.fossilized_count }}</span>
  <span class="seed-separator">/</span>
  <span class="seed-pruned">{{ env.pruned_count }}</span>
</span>
```

**Step 2: Run tests**

Run: `cd src/esper/karn/overwatch/web && npm test -- EnvironmentGrid`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/web/src/components/EnvironmentGrid.vue
git commit -m "feat(overwatch): add tooltip legend for seed count display"
```

---

## Final Verification

**Step 1: Run all Overwatch web tests**

```bash
cd src/esper/karn/overwatch/web && npm test
```

**Step 2: Run all Python tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/test_scaffold_hindsight.py tests/karn/ -v
```

**Step 3: Type check**

```bash
cd src/esper/karn/overwatch/web && npm run type-check
```

---

## Summary

| Task | Priority | Type | Component |
|------|----------|------|-----------|
| 1 | P0 | Bug | KeyboardHelp j/k docs |
| 2 | P0 | Bug | EnvironmentGrid duplicate attr |
| 3 | P0 | Bug | ContributionWaterfall overflow |
| 4 | P0 | Bug | LeaderboardTable Enter key |
| 5 | P1 | A11y | StatusBar icons |
| 6 | P1 | Stability | WebSocket backoff |
| 7 | P1 | Polish | Temperature degree symbol |
| 8 | P1 | A11y | Reduced motion support |
| 9 | P1 | DRL | Hindsight credit weight |
| 10 | P2 | Docs | Magic constant documentation |
| 11 | P2 | Refactor | defaultdict for ledger |
| 12 | P2 | UX | Seed count legend |
