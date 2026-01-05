# Telemetry Audit: decisions_column.py

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo/decisions_column.py`

**Purpose:** Displays a vertical stack of decision cards showing Tamiyo's recent action decisions with throttled updates (one card every 30 seconds).

---

## 1. Telemetry Fields Consumed

### Source: `SanctumSnapshot.tamiyo.recent_decisions` -> `list[DecisionSnapshot]`

The widget receives decisions via `update_snapshot()` and accesses the `tamiyo.recent_decisions` list containing `DecisionSnapshot` objects.

---

## 2. DecisionSnapshot Fields Used

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `decision_id` | `str` | `""` | Unique ID for card tracking and preventing duplicate display. |
| `timestamp` | `datetime` | Required | Used to select the newest candidate decision when refreshing cards. |
| `chosen_action` | `str` | Required | Primary display: "GERMINATE", "ADVANCE", "SET_ALPHA_TARGET", "PRUNE", "FOSSILIZE", "WAIT". Determines card color and context note. |
| `chosen_slot` | `str \| None` | `None` | Displayed as slot number (last character). Used in context notes for PRUNE/FOSSILIZE. |
| `confidence` | `float` | Required | Displayed as percentage (e.g., "confidence:87%"). Used in `_infer_why()` logic. |
| `epoch` | `int` | `0` | Displayed as "epoch:{value}" in training context line. |
| `env_id` | `int` | `0` | Displayed as "env:{value}" in training context line. |
| `batch` | `int` | `0` | Displayed as "batch:{value}" in training context line. |
| `host_accuracy` | `float` | Required | Displayed as "host:{value}%". Used in `_infer_why()` for low accuracy detection. |
| `decision_entropy` | `float` | `0.0` | Displayed as "entropy:{value:.2f}". Used to compute entropy label. |
| `expected_value` | `float` | Required | Displayed as "expect:{value:+.2f}". Used for outcome badge calculation. |
| `actual_reward` | `float \| None` | `None` | Displayed as "reward:{value:+.2f}" or "reward:--" if None. Used for outcome badge. |
| `td_advantage` | `float \| None` | `None` | Displayed as "TD:{value:+.2f}" or "TD:--" if None. |
| `alternatives` | `list[tuple[str, float]]` | `[]` | Displayed as "also: {action} {prob%}" for up to 2 alternatives. |
| `slot_states` | `dict[str, str]` | Required | Maps slot_id to state string (e.g., "Training 12%", "Empty"). Used in `_infer_why()` and `_entropy_label()`. |

### GERMINATE-Specific Fields

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `chosen_blueprint` | `str \| None` | `None` | Displayed truncated to 10 chars for GERMINATE actions. |
| `blueprint_confidence` | `float` | `0.0` | Displayed as percentage after blueprint name (e.g., "(87%)"). |
| `chosen_tempo` | `str \| None` | `None` | Mapped to glyphs: FAST=">>>", STANDARD=">>", SLOW=">". |
| `chosen_style` | `str \| None` | `None` | Mapped to abbreviations: LINEAR_ADD="lin+add", LINEAR_MULTIPLY="lin x mul", SIGMOID_ADD="sig+add", GATED_GATE="gate(circle)". |
| `chosen_curve` | `str \| None` | `None` | Mapped to glyphs via `ALPHA_CURVE_GLYPHS`: LINEAR="/", COSINE="~", SIGMOID_GENTLE=")", SIGMOID=")", SIGMOID_SHARP="D". |

---

## 3. Field Paths (from SanctumSnapshot)

```
SanctumSnapshot
  |-- tamiyo: TamiyoState
        |-- recent_decisions: list[DecisionSnapshot]
        |     |-- decision_id: str
        |     |-- timestamp: datetime
        |     |-- chosen_action: str
        |     |-- chosen_slot: str | None
        |     |-- confidence: float
        |     |-- epoch: int
        |     |-- env_id: int
        |     |-- batch: int
        |     |-- host_accuracy: float
        |     |-- decision_entropy: float
        |     |-- expected_value: float
        |     |-- actual_reward: float | None
        |     |-- td_advantage: float | None
        |     |-- alternatives: list[tuple[str, float]]
        |     |-- slot_states: dict[str, str]
        |     |-- chosen_blueprint: str | None
        |     |-- blueprint_confidence: float
        |     |-- chosen_tempo: str | None
        |     |-- chosen_style: str | None
        |     |-- chosen_curve: str | None
        |-- group_id: str | None  (used for _group_id assignment)
```

---

## 4. Rendering Logic

### 4.1 Card Structure (DecisionCard.render)

Each card renders as a multi-line Rich Text block:

**Line 1: Action + Age**
```
#{index+1} {chosen_action}                 {age}
```
- Age computed from display timestamp (when card was added), not decision timestamp
- Age formatted as "{seconds}s" if <60s, else "{minutes}:{seconds:02d}"

**Line 2: Training Context**
```
epoch:{epoch}  env:{env_id}  batch:{batch}
```

**Line 3: Slot + Confidence**
```
slot:{slot_num}  confidence:{confidence:.0%}
```

**Line 4: Blueprint Info (GERMINATE) or Context Note**
- GERMINATE: `blueprint:{name[:10]}{confidence%}  {tempo_glyph}  {style_abbrev}  {curve_glyph}`
- Other actions: Context note from `_action_context_note()`

**Separator Line**
```
----------------------------------------------
```

**Line 5: WHY Reasoning**
```
WHY: {inferred_reason}
```

**Line 6: Host + Entropy + Badge**
```
host:{accuracy:.0f}%  entropy:{entropy:.2f} {label}        {badge}
```

**Line 7: Expect + Reward + TD**
```
expect:{expected_value:+.2f}  reward:{actual_reward:+.2f}  TD:{td_advantage:+.2f}
```
- If actual_reward is None: "reward:--  TD:--"

**Line 8: Alternatives**
```
also: {alt_action1} {prob1%}  {alt_action2} {prob2%}
```

### 4.2 Action Context Notes (`_action_context_note`)

| Action | Note |
|--------|------|
| GERMINATE | (empty - blueprint info shown instead) |
| WAIT | "(waiting for training progress)" |
| PRUNE | "(removing underperformer from {slot})" |
| FOSSILIZE | "(fusing trained module in {slot})" |
| SET_ALPHA_TARGET | "(adjusting blend parameters)" |

---

## 5. Thresholds and Color Coding

### 5.1 Action Colors (`ACTION_COLORS`)

| Action | Color |
|--------|-------|
| GERMINATE | green |
| SET_ALPHA_TARGET | cyan |
| FOSSILIZE | blue |
| PRUNE | red |
| WAIT | dim |
| ADVANCE | cyan |

### 5.2 Entropy Labels (`_entropy_label`)

Context-aware entropy classification distinguishing legitimate low-entropy from policy collapse:

| Condition | Label | Style |
|-----------|-------|-------|
| entropy < 0.3 + legitimate context | "[checkmark]" | green |
| entropy < 0.3 + concerning context | "[collapsing]" | red |
| 0.3 <= entropy < 0.7 | "[confident]" | yellow |
| 0.7 <= entropy < 1.2 | "[balanced]" | green |
| entropy >= 1.2 | "[exploring]" | cyan |

**Legitimate Low Entropy Conditions:**
- WAIT with no dormant slots OR training in progress
- GERMINATE with confidence > 0.8 AND dormant slot available
- FOSSILIZE/PRUNE with confidence > 0.8

### 5.3 Outcome Badges (`_outcome_badge`)

Compares expected_value to actual_reward:

| Condition | Badge | Style |
|-----------|-------|-------|
| actual_reward is None | "[...]" | dim |
| \|reward - expect\| < 0.1 | "[HIT]" | bright_green |
| \|reward - expect\| < 0.3 | "[~OK]" | yellow |
| \|reward - expect\| >= 0.3 | "[MISS]" | red |

### 5.4 WHY Inference (`_infer_why`)

Generates reasoning text based on decision context:

**GERMINATE:**
- dormant slot available
- host accuracy low (<30) -> "host accuracy low"
- host accuracy <50 -> "host needs help"
- confidence > 0.8 -> "high confidence"
- Fallback: "opportunity to grow"

**WAIT:**
- training_count > 0 -> "{n} slot(s) still training"
- entropy > 1.0 -> "high uncertainty, gathering data"
- dormant_count == 0 -> "all slots occupied"
- confidence > 0.7 -> "deliberate pause"
- Fallback: "monitoring progress"

**PRUNE:**
- confidence > 0.8 -> "clear underperformer"
- Fallback: "removing low contributor"

**FOSSILIZE:**
- slot_state contains "Blending" -> "module ready to fuse"
- slot_state contains "Holding" -> "stable, ready to commit"
- Fallback: "matured module"

**SET_ALPHA_TARGET:**
- confidence > 0.8 -> "alpha adjustment needed"
- Fallback: "tuning blend ratio"

### 5.5 CSS Classes

| Class | Applied When |
|-------|--------------|
| `newest` | index == 0 |
| `oldest` | index == total_cards - 1 |

---

## 6. Throttling Logic

### 6.1 Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `CARD_SWAP_INTERVAL` | 30.0 seconds | Minimum time between card additions/swaps |
| `MAX_CARDS` | 4 | Maximum cards displayed simultaneously |

### 6.2 Firehose Model

- Cards display with age starting at 0 when added (not decision timestamp)
- No backlog/queueing - always grab most recent decision
- Displayed decision IDs tracked to prevent duplicates
- Growing phase: add to front until MAX_CARDS reached
- Steady state: swap oldest card with newest decision

---

## 7. Display Mappings

### 7.1 Tempo Glyphs

| Tempo | Display |
|-------|---------|
| FAST | ">>>" |
| STANDARD | ">>" |
| SLOW | ">" |

### 7.2 Style Abbreviations

| Style | Display |
|-------|---------|
| LINEAR_ADD | "lin+add" |
| LINEAR_MULTIPLY | "lin x mul" |
| SIGMOID_ADD | "sig+add" |
| GATED_GATE | "gate(circle)" |

### 7.3 Alpha Curve Glyphs (from leyline)

| Curve | Glyph |
|-------|-------|
| LINEAR | "/" |
| COSINE | "~" |
| SIGMOID_GENTLE | ")" |
| SIGMOID | ")" |
| SIGMOID_SHARP | "D" |

---

## 8. Interactive Features

### 8.1 Click Handlers

| Click Type | Action |
|------------|--------|
| Click | Posts `DecisionDetailRequested` message with group_id and decision |

### 8.2 Messages Emitted

| Message | Fields | Purpose |
|---------|--------|---------|
| `DecisionDetailRequested` | group_id, decision (DecisionSnapshot) | Open drill-down view |

---

## 9. State Management

### 9.1 DecisionsColumn State

| Field | Type | Purpose |
|-------|------|---------|
| `_snapshot` | `SanctumSnapshot \| None` | Latest snapshot |
| `_displayed_decisions` | `list[DecisionSnapshot]` | Currently visible decisions |
| `_display_timestamps` | `dict[str, datetime]` | Maps decision_id to when card was added |
| `_last_card_swap_time` | `float` | Timestamp of last swap (for throttling) |
| `_rendering` | `bool` | Guard against concurrent renders |
| `_render_generation` | `int` | Unique ID suffix for each render cycle |
| `_group_id` | `str` | A/B test cohort ID from snapshot |

---

## 10. Unused DecisionSnapshot Fields

The following DecisionSnapshot fields are defined but NOT consumed by decisions_column.py:

| Field | Type | Notes |
|-------|------|-------|
| `value_residual` | `float` | r - V(s) |
| `op_confidence` | `float` | Probability of chosen operation |
| `slot_confidence` | `float` | Probability of chosen slot |
| `style_confidence` | `float` | Probability of chosen style |
| `tempo_confidence` | `float` | Probability of chosen tempo |
| `alpha_target_confidence` | `float` | Probability of chosen alpha target |
| `alpha_speed_confidence` | `float` | Probability of chosen alpha speed |
| `curve_confidence` | `float` | Probability of chosen curve |
| `op_entropy` | `float` | Entropy of operation distribution |
| `slot_entropy` | `float` | Entropy of slot distribution |
| `blueprint_entropy` | `float` | Entropy of blueprint distribution |
| `style_entropy` | `float` | Entropy of style distribution |
| `tempo_entropy` | `float` | Entropy of tempo distribution |
| `alpha_target_entropy` | `float` | Entropy of alpha target distribution |
| `alpha_speed_entropy` | `float` | Entropy of alpha speed distribution |
| `curve_entropy` | `float` | Entropy of curve distribution |
| `chosen_alpha_target` | `str \| None` | Target alpha amplitude |
| `chosen_alpha_speed` | `str \| None` | Ramp speed |
