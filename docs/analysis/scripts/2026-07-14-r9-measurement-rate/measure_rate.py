#!/usr/bin/env python3
"""r9 HOLDING per-epoch counterfactual measurement-rate read (task esper-lite-5e9affe207).

READ-ONLY streaming analysis of sealed archival telemetry. One pass per file.

Operationalization (grounded in code, see r9_measurement_rate_results.md):
  A VALID counterfactual (LOO) measurement of a HOLDING seed at (env, episode, slot, epoch)
  is recorded IFF that slot appears in the `slot_ids` of the COUNTERFACTUAL_MATRIX_COMPUTED
  event belonging to that epoch, with finite config accuracies. This is exactly the condition
  under which the trainer resets `epochs_since_counterfactual -> 0` and records a FRESH
  measurement (vectorized_trainer.py:1368-1402); i.e. ContributionState.measured_this_epoch.

Epoch clock:      EPOCH_COMPLETED.epoch (within-episode 1..max_epochs). The top-level `epoch`
                  on SEED_* events is a stuck constant (12) and is IGNORED.
Ordering:         COUNTERFACTUAL_MATRIX_COMPUTED is emitted immediately BEFORE its epoch's
                  EPOCH_COMPLETED -> flush pending measurements at each EPOCH_COMPLETED.
HOLDING census:   EPOCH_COMPLETED.data.seeds[slot].stage == 'HOLDING' (throttle-immune,
                  authoritative per-(env,epoch)).
Lifecycle id:     (env, episode, slot, seed_id); seed_id tracked from SEED_GERMINATED /
                  SEED_STAGE_CHANGED top-level seed_id.
"""
import json
import sys
import math
from collections import defaultdict


def is_finite_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def process_file(path, max_lines=None):
    # partition state keyed by (env, episode)
    slot_seed = defaultdict(dict)       # p -> {slot: seed_id}
    pending = defaultdict(set)          # p -> {slot measured for upcoming epoch}
    # partition epoch bookkeeping (throttle sanity)
    part_epochs = defaultdict(lambda: [None, None, 0])  # p -> [min, max, count]
    # lifecycle: L=(env,ep,slot,seed_id) -> list of (epoch, measured_bool)
    life = defaultdict(list)
    life_gen_unknown = 0

    regime = None
    n_epoch_completed = 0
    n_cf = 0
    n_cf_invalid = 0
    n = 0

    with open(path) as f:
        for line in f:
            n += 1
            if max_lines and n > max_lines:
                break
            # regime stamp from first TRAINING_STARTED
            if regime is None and '"TRAINING_STARTED"' in line:
                e = json.loads(line)
                regime = e['data']
                continue
            # fast substring gate
            if '"EPOCH_COMPLETED"' in line:
                e = json.loads(line)
                if e.get('event_type') != 'EPOCH_COMPLETED':
                    continue
                d = e['data']
                env = d.get('env_id'); ep = d.get('episode_idx')
                if env is None or ep is None:
                    continue
                p = (env, ep)
                epoch = e.get('epoch')
                pe = part_epochs[p]
                if pe[0] is None or epoch < pe[0]:
                    pe[0] = epoch
                if pe[1] is None or epoch > pe[1]:
                    pe[1] = epoch
                pe[2] += 1
                n_epoch_completed += 1
                seeds = d.get('seeds') or {}
                measured_slots = pending[p]
                for slot, v in seeds.items():
                    if v.get('stage') != 'HOLDING':
                        continue
                    seed_id = slot_seed[p].get(slot)
                    if seed_id is None:
                        seed_id = f'{slot}:UNKNOWN'
                    L = (env, ep, slot, seed_id)
                    life[L].append((epoch, slot in measured_slots))
                pending[p] = set()
            elif '"COUNTERFACTUAL_MATRIX_COMPUTED"' in line:
                e = json.loads(line)
                if e.get('event_type') != 'COUNTERFACTUAL_MATRIX_COMPUTED':
                    continue
                d = e['data']
                env = d.get('env_id'); ep = d.get('episode_idx')
                if env is None or ep is None:
                    continue
                p = (env, ep)
                n_cf += 1
                # validity: all config accuracies finite
                cfgs = d.get('configs') or []
                accs_ok = all(is_finite_num(c.get('accuracy')) for c in cfgs) and len(cfgs) > 0
                if not accs_ok:
                    n_cf_invalid += 1
                for slot in (d.get('slot_ids') or []):
                    if accs_ok:
                        pending[p].add(slot)
            elif '"SEED_STAGE_CHANGED"' in line or '"SEED_GERMINATED"' in line:
                e = json.loads(line)
                et = e.get('event_type')
                if et not in ('SEED_STAGE_CHANGED', 'SEED_GERMINATED'):
                    continue
                d = e['data']
                env = d.get('env_id'); ep = d.get('episode_idx')
                slot = d.get('slot_id')
                sid = e.get('seed_id')
                if env is None or ep is None or slot is None or sid is None:
                    continue
                slot_seed[(env, ep)][slot] = sid

    # count unknown-generation HOLDING epochs
    for L in life:
        if str(L[3]).endswith(':UNKNOWN'):
            life_gen_unknown += 1

    return {
        'regime': regime,
        'n_lines': n,
        'n_epoch_completed': n_epoch_completed,
        'n_cf': n_cf,
        'n_cf_invalid': n_cf_invalid,
        'life': life,
        'part_epochs': part_epochs,
        'life_gen_unknown': life_gen_unknown,
    }


def pctl(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx)); hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def contiguous_runs(epoch_measured):
    """Split a lifecycle's HOLDING (epoch, measured) list into maximal runs of
    consecutive epoch integers. Returns list of runs; each run is list of measured bools
    ordered by epoch."""
    epoch_measured = sorted(epoch_measured, key=lambda x: x[0])
    runs = []
    cur = []
    prev = None
    for ep, m in epoch_measured:
        if prev is not None and ep != prev + 1:
            runs.append(cur); cur = []
        cur.append(m); prev = ep
    if cur:
        runs.append(cur)
    return runs


def analyze(res, label):
    life = res['life']
    total_hold_epochs = 0
    total_measured = 0
    gap_lengths = []            # unmeasured-run lengths within contiguous HOLDING runs
    span_lengths = []          # contiguous HOLDING span lengths
    n_lifecycles = 0
    n_lifecycles_hold = 0
    win5_total = 0
    win5_full = 0             # windows with all 5 measured
    # additional: per-lifecycle measured fraction, first-epoch measured
    for L, em in life.items():
        n_lifecycles_hold += 1
        for ep, m in em:
            total_hold_epochs += 1
            if m:
                total_measured += 1
        for run in contiguous_runs(em):
            span_lengths.append(len(run))
            # gaps: maximal runs of False
            gl = 0
            for m in run:
                if not m:
                    gl += 1
                else:
                    if gl > 0:
                        gap_lengths.append(gl)
                    gl = 0
            if gl > 0:
                gap_lengths.append(gl)
            # 5-in-5 windows within this contiguous span
            if len(run) >= 5:
                for i in range(0, len(run) - 4):
                    win5_total += 1
                    if sum(run[i:i + 5]) >= 5:
                        win5_full += 1

    gap_lengths_sorted = sorted(gap_lengths)
    span_sorted = sorted(span_lengths)
    frac = (total_measured / total_hold_epochs) if total_hold_epochs else None

    # span-length distribution buckets
    from collections import Counter
    span_hist = Counter(span_lengths)
    n_spans_ge5 = sum(1 for s in span_lengths if s >= 5)

    out = {
        'label': label,
        'n_lifecycles_hold': n_lifecycles_hold,
        'total_hold_epochs': total_hold_epochs,
        'total_measured': total_measured,
        'measured_fraction': frac,
        'n_gaps': len(gap_lengths),
        'gap_P50': pctl(gap_lengths_sorted, 0.50),
        'gap_P90': pctl(gap_lengths_sorted, 0.90),
        'gap_P99': pctl(gap_lengths_sorted, 0.99),
        'gap_max': (gap_lengths_sorted[-1] if gap_lengths_sorted else None),
        'unmeasured_hold_epochs': total_hold_epochs - total_measured,
        'n_spans': len(span_lengths),
        'span_P50': pctl(span_sorted, 0.50),
        'span_P90': pctl(span_sorted, 0.90),
        'span_max': (span_sorted[-1] if span_sorted else None),
        'n_spans_ge5': n_spans_ge5,
        'win5_total': win5_total,
        'win5_full': win5_full,
        'win5_full_fraction': (win5_full / win5_total) if win5_total else None,
        'span_hist_small': {k: span_hist[k] for k in sorted(span_hist)[:12]},
        'n_cf': res['n_cf'],
        'n_cf_invalid': res['n_cf_invalid'],
        'n_epoch_completed': res['n_epoch_completed'],
        'life_gen_unknown': res['life_gen_unknown'],
    }
    # throttle sanity: partitions where EPOCH_COMPLETED count < (max-min+1) => a dropped epoch
    part_gap = 0
    for p, (mn, mx, cnt) in res['part_epochs'].items():
        if mn is not None and cnt < (mx - mn + 1):
            part_gap += 1
    out['partitions_with_epoch_number_gaps'] = part_gap
    out['n_partitions'] = len(res['part_epochs'])
    return out


if __name__ == '__main__':
    path = sys.argv[1]
    label = sys.argv[2]
    max_lines = int(sys.argv[3]) if len(sys.argv) > 3 else None
    res = process_file(path, max_lines)
    summary = analyze(res, label)
    # merge regime stamp (subset of keys) for assertion
    reg = res['regime'] or {}
    stamp_keys = ['recurrent_n_epochs', 'reward_mode', 'chunk_length', 'max_epochs',
                  'per_head_advantage_norm', 'max_seeds', 'gamma', 'task', 'n_envs',
                  'gae_lambda', 'reward_family']
    summary['regime_stamp'] = {k: reg.get(k) for k in stamp_keys}
    summary['regime_has_obs_v4_key'] = any('obs_v4' in k for k in reg.keys())
    print(json.dumps(summary, indent=2))
