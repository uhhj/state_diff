# CCDA V9.5 — PB3 Wrapping Snapshot Same-Action Causal Bifurcation Audit

> Start main SHA: `bc2baa7aef5e0e0f593ef26d4a2c11debb83236b`  
> DLO-Lab gitlink: `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`

## Scientific question

PB2-C is complete and produced 5,534 frozen discovery candidates from 123 official-valid rollouts. PB3 now tests only CCDA Condition 4:

> Do hidden-winding-different branch states undergo different deformation motion under the identical future qpos suffix, beyond deterministic snapshot/replay uncertainty?

No new pair discovery is allowed.

## Formal shortlist

Freeze 10 pairs / 20 unique rollouts before any PB3 future simulation.

Selection rule:

```text
scan PB2-C saved top candidates in frozen rank order
accept iff neither rollout has appeared previously
stop after 10
```

This rule uses no future, force, sensor, or reward information.

The frozen shortlist contains three `t=13` pairs with `[0,0,0] ↔ [0,1,0]` and seven `t=20` pairs with `[0,1,0] ↔ [1,1,0]`.

## Targeted replay alignment

Reuse:

```text
/data/Experiment3/data/pb2c_dlolab_wrapping/rollouts_initial.npz
```

Every branch state must be replayed from its exact batch/seed/env/time and aligned to frozen PB2-C data before causal interpretation.

Engineering identity limits:

```text
rope max abs      <= 5e-5 m
EE max abs        <= 5e-5 m
motor-qpos maxabs <= 5e-5 rad
winding index     exact
```

Failure verdict:

```text
PB3_TARGETED_REPLAY_ALIGNMENT_FAILED
```

## Snapshot protocol

Use pinned Genesis:

```python
snapshot = env.scene.get_state()
env.scene.reset(state=snapshot)
```

Pinned Genesis `reset(state=...)` replaces the registered initial state. Therefore preserve one pristine built state and explicitly restore it before creating each new randomized source batch.

For each branch state:

```text
snapshot repeats = 3
future horizons  = 1,2,5,10,20,40
```

Snapshot restore rope error must remain <= `5e-5 m`.

Formal selected repeats must stay finite and inside the published stretch-validity condition.

## Primary Gate-4 metric

Raw full-rope distance is not the formal metric, because hidden full geometry already differs at `t`.

Use differential deformation:

\[
\Delta X_A(h)=X_A(t+h)-X_A(t)
\]

\[
\Delta X_B(h)=X_B(t+h)-X_B(t)
\]

and

\[
D_\Delta(h)=
\sqrt{
\frac{1}{N}
\sum_i
\|
\Delta x_{A,i}(h)-\Delta x_{B,i}(h)
\|_2^2
}.
\]

This tests whether the same future command induces different material-point motion rather than merely detecting the already-present hidden topology.

With 3 repeats per side, each horizon has 9 cross-branch values. Report minimum/median/maximum. Formal Gate 4 uses the minimum.

## Repeat floor

Within each side, compare all 3 choose 2 repeat pairs with the same displacement-field metric.

\[
F(h)=\max(F_A(h),F_B(h)).
\]

Use the maximum repeat divergence, not a percentile.

## Frozen formal criterion

A sampled horizon passes iff:

\[
\min D_\Delta(h)
\ge
\max(1\text{ mm},5F(h)).
\]

The 1 mm absolute effect is fixed as 0.1 × the published 10 mm rope radius.

A pair passes if at least one sampled horizon passes.

PB3 positive requires:

```text
passing pairs >= 3
AND
passing winding strata >= 2
```

Positive verdict:

```text
PB3_CAUSAL_FUTURE_BIFURCATION_CONFIRMED
```

Negative verdict:

```text
PB3_CAUSAL_FUTURE_BIFURCATION_NOT_CONFIRMED
```

Do not change `1 mm`, `5×`, `3 pairs`, or `2 strata` after seeing results.

## Diagnostics only

Report, but never use as Gate 4:

```text
raw ordered full-rope position RMSE
artificial partial-state future Chamfer
partial Chamfer growth from h=0
signed winding L-inf difference
```

The PB2-C 10 mm discovery threshold is not reused.

Use the term:

```text
horizon_of_maximum_sampled_displacement_divergence
```

not `peak horizon`.

## Outputs

Raw:

```text
/data/Experiment3/data/pb3_dlolab_wrapping/
  PB3_TRAJECTORIES.npz
  AUDIT.json
```

Committed:

```text
reports/experiment3/pb3_dlolab_wrapping/
  RESULT.md
  EVIDENCE.json
  PAIR_METRICS.json
```

## Next action

If PB3 is positive, stop branch engineering and enter PB4 control relevance/action regret. Do not train StateDiff until PB4 is established.

If PB3 is negative, do not rescue it by changing formal criteria or replacing failed pairs with future-selected candidates.


## REV1 shortlist-validation correction

The shortlist file is not trusted merely because its pairs are members of the
PB2-C top-50, use unique rollouts, and have monotonically increasing
`source_rank`.

PB3 now **recomputes the greedy rank-order scan from committed
`CANDIDATES.json`**:

```text
used = {}
expected = []

for source_rank, candidate in enumerate(top_candidates, start=1):
    if candidate.rollout_a in used:
        continue
    if candidate.rollout_b in used:
        continue

    expected.append(candidate)
    used += {rollout_a, rollout_b}

    if len(expected) == 10:
        break
```

The supplied shortlist must then equal this recomputed sequence pair-by-pair,
including:

```text
source_rank
rollout A/B
time index
winding indices
differing posts
PB2-C visible-history Chamfer
winding integer residual
replay batch/env/seed metadata
```

A shortlist that is merely monotonic and rollout-disjoint but skips an
earlier admissible candidate is rejected.

`EVIDENCE.json` records:

```text
shortlist_validation.rank_order_scan_verified = true
derived_source_ranks
actual_source_ranks
target_pair_count
unique_rollout_count
```
