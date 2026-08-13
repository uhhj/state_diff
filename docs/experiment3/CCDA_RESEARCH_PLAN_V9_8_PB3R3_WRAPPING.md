# CCDA V9.8 — PB3-R3 Independent Calibration-Based Alignment Re-Preregistration

> Start main SHA: `2d7324106bf15622f07ed2990870ec0f7f754919`
> DLO-Lab gitlink: `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`

## 1. Entry condition

PB3-R2 is complete:

```text
PB3R2_REPLAY_FLOOR_CALIBRATED
```

Independent calibration:

```text
20 states
3 fresh replays/state
60 frozen-reconstruction samples
60 fresh-process repeat samples
all 60 target samples finite
formal PB3 overlap = 0
```

Observed rope max-abs distributions:

```text
frozen acquisition -> fresh replay:
  median 24.47 um
  P95    395.31 um
  max    516.74 um

fresh replay -> fresh replay:
  median 0.00095 um
  P95    0.0596 um
  max    16.53 um
```

Original 50 um rope max-abs reference:

```text
36 / 60 pass
24 exceedances
```

PB3-R2 did not choose a new threshold.

PB3-R3 is CPU-only and exists only to freeze the alignment handling before any formal PB3 Resume.

---

## 2. Scientific interpretation boundary

PB3-R2 supports:

```text
fresh replay is highly repeatable
but
original PB2-C frozen acquisition -> fresh replay
has state-dependent reconstruction offsets
```

It does not support treating 516.74 um as ordinary fresh-run stochastic noise.

Therefore PB3-R3 must not simply choose:

```text
400 um max-abs
520 um max-abs
```

from P95/max.

The new targeted-replay rope engineering gate uses whole-rope coordinate-wise RMSE instead of single-coordinate max-abs.

---

## 3. New targeted-replay rope metric

Formal engineering metric:

\[
E_{coord}
=
\sqrt{\frac{1}{3N}
\sum_{i=1}^N
\|X_i^{live}-X_i^{frozen}\|_2^2}
\]

Implementation name:

```text
coordinate_rmse_m
```

Important:

```text
sqrt(mean(all XYZ coordinate squared errors))
```

This is not the same as the existing PB3 ordered per-vertex Euclidean RMSE used for Gate 4.

`rope_max_abs_coordinate_m` remains stored as a diagnostic only.

---

## 4. Avoid pseudo-replication

The 60 reconstruction rows are not treated as 60 independent calibration states.

For each of the 20 independent states:

```text
three fresh-process coordinate_rmse_m values
→ state median
```

giving:

```text
20 independent state-level medians
```

The preregistered threshold is:

\[
\tau_{rope,R3}
=
\max_{j=1..20}
\operatorname{median}_{r=1..3}
E_{j,r}
\]

No multiplier.

No percentile chosen after seeing formal PB3.

No formal PB3 outcome is read by threshold derivation.

The exact numeric threshold is generated from retained PB3-R2 worker JSONs on the server and then committed before any PB3 Resume.

---

## 5. Raw-source validation before deriving threshold

PB3-R3 must load exactly the 12 canonical logical worker JSON files:

```text
batch0_repeat0..2
batch1_repeat0..2
batch2_repeat0..2
batch3_repeat0..2
```

Attempt logs are not calibration realizations and must not be loaded.

Require:

```text
12 / 12 worker_verdict = PB3R2_WORKER_COMPLETE
future_steps_executed = 20
5 target rows / worker
all_target_samples_finite = true

60 unique (calibration_id, repeat_index) rows
20 calibration IDs
repeat indices [0,1,2] for every state
```

The 60 raw `coordinate_rmse_m` values must reproduce the committed PB3-R2 coordinate-RMSE summary:

```text
count
min
median
P95
P99
max
```

before threshold derivation proceeds.

---

## 6. Retain robot-state alignment

Do not relax unrelated channels.

Future formal branch targeted replay keeps:

```text
EE max abs <= 5e-5 m
motor qpos max abs <= 5e-5 rad
live winding index == frozen winding index
```

Only the rope frozen-reconstruction gate changes from:

```text
max-abs <= 5e-5 m
```

to:

```text
coordinate_rmse_m <= tau_rope_R3
```

---

## 7. Snapshot restore gate is a separate process

PB3-R2 calibrated:

```text
original PB2-C acquisition
vs
fresh targeted replay
```

It did not calibrate:

```text
snapshot capture
vs
snapshot restore
```

Therefore retain the original PB3 snapshot-restore rule unchanged:

```text
rope max-abs <= 5e-5 m
```

Do not apply the new R3 targeted-replay threshold to snapshot restore.

---

## 8. Live PB2-C pair revalidation

Passing frozen-reconstruction alignment is not enough.

Before any future suffix, the later PB3 Resume must re-evaluate the actual live reconstructed pair using the original PB2-C 3-frame semantics.

For each of the 10 formal pairs, using frames:

```text
t-2
t-1
t
```

require:

### Partial rope

Use original PB2-C artificial post-local occlusion surrogate.

```text
occlusion radius = 0.05 m
visible history symmetric 3D Chamfer <= 0.01 m
nonempty visible rope on both sides at every frame
```

### Robot history

```text
dual-arm mean EE-position history <= 0.01 m

sign-invariant quaternion-geodesic history mean
<= 0.08726646259971647 rad

dual-arm motor-qpos history RMS
<= 0.05 rad
```

### Hidden state

At branch time:

```text
rounded signed winding index vectors differ
```

### Action/time

```text
same time
same common official qpos action history
```

These are original PB2-C pair-admission semantics, not new criteria.

---

## 9. No survivor bias

All frozen formal content remains fixed:

```text
10 formal pairs
20 unique rollouts
```

PB3 Resume must not:

```text
drop a failing formal pair
replace a pair
select a later PB2-C candidate
re-rank using live reconstruction
```

Every one of the 20 branch states must pass engineering alignment.

Every one of the 10 live pairs must pass live PB2-C revalidation.

Any failure before future suffix gives:

```text
PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED
```

and records a diagnostic failure component such as:

```text
targeted_replay_alignment
live_pair_revalidation
```

No future causal interpretation is allowed.

---

## 10. Future PB3 execution order is frozen now

Later PB3 Resume must execute:

```text
1. replay all required branch histories through branch time
2. collect t-2,t-1,t histories
3. validate all 20 targeted-replay engineering alignments
4. revalidate all 10 live PB2-C pair semantics
5. if any failure -> stop
6. only then run snapshot restore + future suffix repeats
7. apply unchanged Gate 4
```

This order is important: no future suffix may be generated before all formal live pairs are revalidated.

---

## 11. Gate 4 remains unchanged

PB3-R3 must preserve:

```text
primary:
ordered rope displacement-field divergence

absolute effect:
1 mm

repeat-relative effect:
5 x per-horizon repeat floor

phase positive:
>= 3 formal pairs pass
AND
>= 2 winding strata pass
```

PB3-R3 is not a Gate-4 recalibration phase.

---

## 12. PB3-R3 formal verdict

Successful CPU-only derivation and rule freeze:

```text
PB3R3_ALIGNMENT_RULE_PREREGISTERED
```

Required evidence:

```text
Genesis/GPU worker executed = false
future suffix executed = false
Gate 4 executed = false
formal PB3 outcomes used for threshold = false

20 state-level calibration medians
exact governing state
exact tau_rope_R3

snapshot restore changed = false
Gate 4 changed = false
pair drop allowed = false
pair replacement allowed = false
```

---

## 13. Generated outputs

Generated from retained PB3-R2 raw JSON and committed:

```text
configs/experiment3/published_benchmark/
  dlolab_wrapping_pb3r3_alignment_rule.json

reports/experiment3/pb3r3_alignment_preregistration/
  RESULT.md
  EVIDENCE.json
  STATE_LEVEL_CALIBRATION.json
```

The rule JSON must be generated, validated, and committed before any PB3 GPU Resume.

---

## 14. Stop condition

After:

```text
PB3R3_ALIGNMENT_RULE_PREREGISTERED
```

STOP.

Do not run PB3 Resume in the same phase.

The next phase will be a separate implementation/execution step that applies the already committed R3 rule to the frozen formal 10-pair PB3 audit.
