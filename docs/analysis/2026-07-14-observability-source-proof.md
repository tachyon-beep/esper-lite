# Observability source-proof: pending-committed HOLDING is ALIASED in the observation

Date: 2026-07-14. Tracker: esper-lite-8ca95a0ecf. Feeds OWNER FORK F2 (pending-visibility, PDR-0091 #5).
Read-only source proof executed by a dispatched agent; findings condensed here verbatim-in-substance with
citations (the agent wrote no files by instruction; this document is the durable record, main-session reviewed).

**Scope:** every observation feature reaching the network is enumerated in
`src/esper/tamiyo/policy/features.py::batch_obs_to_features` (base dims 0–23 at :837–869; per-slot dims at
:872–992). The LSTM input is exactly `cat([obs, blueprint_embeddings])` (`factored_lstm.py:820–826` forward,
`:1357–1363` evaluate_actions). Masks are separate kwargs. No reward feature and no `committed` field exist
anywhere in the obs.

## The four questions (gpt-prime's minimal-delta rule), answered from source

| Q | Verdict | Evidence |
|---|---|---|
| Q1: is committed/pending state encoded in any obs feature (v3 or v4)? | **ABSENT — aliased** | Per-slot block :872–992: stage one-hot + alpha identical (neither changes at request); alpha-scaffolding dims (+15..22) already read "schedule complete" for ORDINARY HOLDING (`slot.py:917, :1838` full-amplitude; `:541–547, :474–483`) so freezing adds no delta; the obs-v4 `cf_frozen` flag (+33) fires only at ACTUAL fossilization (`handlers/fossilize.py:206` → `contribution_state.py:108–119`), not at a commit — 0 for both seeds; cf value/fresh/age dims alias too because the settlement keeps measuring the committed seed (nothing branches cf-measurement on `committed`) |
| Q2: are masks network INPUT? | **ABSENT — logit-only** | masks passed as kwargs (`lstm_bundle.py:113–121, :253–260`) and applied via `masked_fill` AFTER the forward pass (`factored_lstm.py:849–864`, `:1531`); never concatenated into state — the committed slot's masking is invisible to the network input |
| Q3: is epoch observed / countdown derivable? | **ESTABLISHED** | `epoch_norm` at base dim 0 (`features.py:832, :837`), normalized on fixed `max_epochs` → epoch exactly recoverable → time-to-next-boundary = W − (epoch mod W) is a deterministic function of an observed feature (learnability of "mod 10" is a separate question). Caveat: holds for the fixed-cadence schedule; a commit-relative schedule would need memory of the commit epoch |
| Q4: is the previous action in the recurrent input? | **ESTABLISHED but partial** | `last_action_op`/`last_action_success` set at `action_execution.py:1415–1416`, enter base dims 16–22 (`features.py:862–868`) — OP-ONLY (sampled slot never fed back), global per-env, one step, overwritten by the next non-WAIT op; rewards are never inputs |

## Overall verdict

**Committed-membership is genuinely aliased; the countdown is observable.** Two identical-telemetry HOLDING
seeds — one pending, one not — produce byte-identical obs at every epoch except the single step after the
commit op (`last_action_op` echo). The only channel to "know" committed is the LSTM latching that one-step,
op-only, non-slot-attributed echo across ~W epochs of intervening WAITs with no reinforcing input and no
reward signal — fragile and ambiguous, not a clean state feature.

## Minimal fields to close the gap (if the owner rules for legibility)

1. **(Sufficient) ONE per-slot boolean `pending_settlement`** (e.g. +35 under obs-v4). With dim 0 already
   supplying the epoch, this is the WHOLE gap — identity + derivable countdown.
2. The explicit countdown dim is information-redundant given #1 + dim 0 (a learnability convenience only);
   per gpt-prime's own rule ("do not duplicate information merely for convenience"), skip it.
3. Any schema change = re-warm/retrain + an explicit obs-schema version stamp; arm A emits a constant 0
   (schema identical across arms).
4. **Latent accidental channel flagged:** if the settlement ever STOPPED measuring committed seeds, the
   cf-staleness dims would become an implicit discriminator — accidental, non-clean observability; the
   explicit bit is the designed fix. (The current spec keeps measuring — §2.0 — so this channel is inert.)

## Consequence for fork F2 (PDR-0093)

The PDR-0091 reversal branch "observability already established → F2 collapses to the interpretation-note
resolution" did NOT fire — the proof went the other way. The fork remains owner-open (the primes still
diverge), but the factual ground has moved: gpt-prime's state-aliasing concern is structurally confirmed,
and the remedy is narrowed to one bit + the existing epoch feature. The aliasing-cost telemetry + symmetry
check (claude-prime) remain valuable under EITHER ruling — as the verification that the bit fixes the thing,
or as the instrument if invisibility is retained.
