# Telemetry Audit: DecisionDetailScreen

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/decision_detail_screen.py`
**Purpose:** Modal screen showing full details for a single Tamiyo decision, including summary metadata, action details, value diagnostics, factored head choices, slot states, and action alternatives.

---

## Telemetry Fields Consumed

### Source: DecisionSnapshot (path: `snapshot.tamiyo.recent_decisions[i]`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `decision_id` | `str` | `""` | Displayed as "Decision ID" in Summary section |
| `timestamp` | `datetime` | required | Displayed as ISO timestamp; also used to compute age |
| `env_id` | `int` | `0` | Displayed as "Env" in Summary section |
| `epoch` | `int` | `0` | Displayed as "Epoch" in Summary section |
| `batch` | `int` | `0` | Displayed as "Batch" in Summary section |
| `chosen_action` | `str` | required | Displayed as "Op" in Action section |
| `chosen_slot` | `str \| None` | `None` | Displayed as "Slot" in Action section; shows "--" if None |
| `confidence` | `float` | required | Displayed as percentage in Action section |
| `decision_entropy` | `float` | `0.0` | Displayed as "Entropy" in Action section |
| `expected_value` | `float` | required | Displayed as "V(s)" in Values section |
| `actual_reward` | `float \| None` | `None` | Displayed as "Reward" in Values section; shows "pending" if None |
| `value_residual` | `float` | `0.0` | Displayed as "Residual" in Values section |
| `td_advantage` | `float \| None` | `None` | Displayed as "TD(0)" in Values section; shows "pending" if None |
| `chosen_blueprint` | `str \| None` | `None` | Displayed in Factored Heads section; shows "--" if None |
| `blueprint_confidence` | `float` | `0.0` | Shown as probability percentage with blueprint |
| `blueprint_entropy` | `float` | `0.0` | Shown as entropy value with blueprint |
| `chosen_tempo` | `str \| None` | `None` | Displayed in Factored Heads section; shows "--" if None |
| `tempo_confidence` | `float` | `0.0` | Shown as probability percentage with tempo |
| `tempo_entropy` | `float` | `0.0` | Shown as entropy value with tempo |
| `chosen_style` | `str \| None` | `None` | Displayed in Factored Heads section; shows "--" if None |
| `style_confidence` | `float` | `0.0` | Shown as probability percentage with style |
| `style_entropy` | `float` | `0.0` | Shown as entropy value with style |
| `chosen_curve` | `str \| None` | `None` | Displayed in Factored Heads section; shows "--" if None |
| `curve_confidence` | `float` | `0.0` | Shown as probability percentage with curve |
| `curve_entropy` | `float` | `0.0` | Shown as entropy value with curve (note: schema field is `curve_entropy` but widget accesses it incorrectly - see bug below) |
| `chosen_alpha_target` | `str \| None` | `None` | Displayed in Factored Heads section; shows "--" if None |
| `alpha_target_confidence` | `float` | `0.0` | Shown as probability percentage with alpha target |
| `alpha_target_entropy` | `float` | `0.0` | Shown as entropy value with alpha target |
| `chosen_alpha_speed` | `str \| None` | `None` | Displayed in Factored Heads section; shows "--" if None |
| `alpha_speed_confidence` | `float` | `0.0` | Shown as probability percentage with alpha speed |
| `alpha_speed_entropy` | `float` | `0.0` | Shown as entropy value with alpha speed |
| `slot_states` | `dict[str, str]` | `{}` | Displayed as key-value pairs in Slot State section |
| `alternatives` | `list[tuple[str, float]]` | `[]` | Displayed as action-probability pairs in Alternatives section |

### Additional Context (passed to constructor)

| Field | Type | Usage |
|-------|------|-------|
| `group_id` | `str` | Displayed in title bar for A/B test identification |

---

## Thresholds and Color Coding

This widget uses minimal color coding - all data is displayed with `dim` style for content rows:

| Element | Style | Meaning |
|---------|-------|---------|
| Section headers | `bold cyan` | "Summary", "Action", "Values", "Factored Heads", "Slot State", "Alternatives" |
| Title | `bold` | "TAMIYO DECISION DETAIL" |
| Group ID in title | `dim` | A/B test group identifier |
| All data rows | `dim` | Uniform styling for all field values |

**No conditional thresholds are applied.** Unlike other widgets that use color coding for health indicators, this detail screen displays raw values without judgment.

---

## Rendering Logic

1. **Title Bar**: Renders "TAMIYO DECISION DETAIL" with the group_id in brackets

2. **Summary Section**: Shows decision metadata
   - decision_id, timestamp (ISO format), age (computed from current time)
   - env_id, epoch, batch

3. **Action Section**: Shows the chosen action
   - Operation name, target slot (or "--")
   - Confidence as percentage, entropy as 3-decimal float

4. **Values Section**: Shows value function diagnostics
   - V(s) expected value with sign
   - Actual reward (or "pending" if None)
   - Value residual (r - V(s)) with sign
   - TD(0) advantage (or "pending" if None)

5. **Factored Heads Section**: Shows per-head sub-decisions
   - Each head displays: chosen value, confidence %, and entropy
   - Heads: Blueprint, Tempo, Style, Curve, Alpha Target, Alpha Speed

6. **Slot State Section**: Shows slot_id -> state mapping
   - Sorted by slot_id
   - Falls back to "(no slot state captured)" if empty

7. **Alternatives Section**: Shows alternative action probabilities
   - Each alternative shows action name and probability %
   - Falls back to "(none captured)" if empty

---

## Additional Notes

### Fields Not Displayed

The following DecisionSnapshot fields exist in the schema but are NOT displayed by this widget:

- `op_confidence` — Confidence for the op head specifically
- `op_entropy` — Entropy for the op head
- `slot_confidence` — Confidence for slot selection
- `slot_entropy` — Entropy for slot selection
- `host_accuracy` — Host model accuracy at decision time

These op/slot-level metrics could be valuable additions to the Factored Heads section.

### Formatting Details

- Section separators use em-dash (`—`) characters
- All floating-point values displayed with sign prefix where applicable
- "pending" displayed for None values in reward/advantage fields
