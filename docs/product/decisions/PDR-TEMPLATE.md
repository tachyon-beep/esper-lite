# PDR-NNNN — <short imperative title>

Date: YYYY-MM-DD   Status: accepted | proposed | superseded
Supersedes: <PDR-NNNN | —>   Related: <files / PDRs>

## Context
<Why this decision is on the table now. The situation, the trigger.>

## What does this buy, and how is that measured?  (REQUIRED — PDR-0068)
<The downstream PRODUCT outcome this decision is meant to produce, and the metric/observation
that would show it did. If the point of the work is to ESTABLISH that relationship, say so
explicitly ("establishing this is the experiment"). A decision that introduces an optimization
target, reward term, or metric-driven epic MUST answer this — an unanswerable box is the
"strong engineering, no empirical baseline" failure mode, made visible before it is banked.>

## Options considered
<2–4 real options with pros/cons.>

## The call
<What was decided, and the rationale. Mark owner-gated items as proposed.>

## Before-a-run check  (when this authorizes a GPU run — PDR-0067 G1 / PDR-0068)
<What ALREADY-COLLECTED data bears on the question and why it is insufficient. For a
decision-bearing run, the pre-committed reading must be independently reviewed for what the
metric can STRUCTURALLY not show, with one synthetic counterexample (world where A holds,
world where B holds, the read separates them).>

## Reversal trigger
<The pre-committed, metric-bound condition under which this decision reopens. Tie it to
metrics.md so it fires on data, not mood.>
