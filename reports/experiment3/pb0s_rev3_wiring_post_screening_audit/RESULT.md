# PB0-S REV3 Wiring-post Screening Assumption Audit

Verdict: `PB0S_REV3_SCREENING_ASSUMPTIONS_AUDITED`

Formal PB0 verdict remains: `PB0_WIRING_POST_NO_NATURAL_CCDA_PAIRS`

## Formal PB0 funnel

- t+10-eligible same-time pairs: 11642624
- Chamfer pass: 9465367
- EE pass: 9286773
- Original hidden pass: 27569
- Original t+10 ratio pass: 0

## Full-time hidden-pair discovery for horizon audit

- Original-hidden pairs across all current-state times: 27569
- Global-descriptor-only pairs: 0

Each horizon uses its own eligible current-state time range.

## Chamfer metric audit

- Search exhaustive: False
- Positive candidate count: 0
- Zero is non-conclusive.

## Cross-time audit

- Search exhaustive: False
- Positive candidate count: 0
- Zero is non-conclusive.

## Findings

- `PB0S_ORIGINAL_FUNNEL_CHARACTERIZED`
- `PB0S_ORIGINAL_HIDDEN_PAIRS_EXIST_BEFORE_T10_FILTER`

## Next action

Pre-register snapshot same-action branching on top original-descriptor pairs. Evaluate the future at multiple horizons and compare branch divergence against deterministic repeat/uncertainty at each horizon. Do not reuse future/current>=2 as formal Gate 4.

## Terminology

REV3 reports `horizon_of_maximum_sampled_future_distance`; it does not use or claim a physical `peak horizon`.

