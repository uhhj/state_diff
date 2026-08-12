# CCDA V9.4 — PB2-C REV1 DLO-Lab Wrapping Natural Pair Discovery

> Main start SHA: `4ee307741b57a4d5cbfdea4ef1cdfcd0700d7d05`  
> DLO-Lab gitlink: `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`

## 1. Scientific question

PB2-A/B has passed. PB2-C asks only:

> Under the unchanged published Wrapping task and repository-provided rope-position randomization, are there naturally occurring same-time states whose **artificial partial-state observation histories** and **published robot-state histories** are near, while their **task-native signed winding indices** differ?

PB2-C is a discovery phase. It does not establish future bifurcation, control relevance, sensing observability, or a deployable perception stack.

---

## 2. REV1 corrections

This revision fixes four issues before the formal PB2-C run:

1. the collector inherits the official Wrapping rollout-validity logic;
2. winding integer residual is stored in every candidate;
3. the rope observation is explicitly called an **artificial partial-state discovery observation**, not a deployable observation;
4. robot observation is explicitly defined from the published Wrapping `compute_observation()` robot component.

---

## 3. Official rollout validity

The published Wrapping evaluator marks a rollout failed when either:

```text
geodesic_distance(control_idx[0], control_idx[1])
/
control_dist_init
> 1.2
```

or rope vertices become NaN during the rollout.

A surviving final state with NaN reward is also not usable as a valid completed rollout.

PB2-C therefore records for every rollout:

```text
official_rollout_valid
official_first_fail_step
official_failed_stretch
official_failed_rope_nan
official_final_reward_nan
official_max_stretch_ratio
```

Raw failed rollouts are preserved, but **only full official-valid rollouts may enter pair mining**.

Do not rescue pre-failure states from a rollout that the published evaluator would reject.

---

## 4. Collection protocol

Use the completed official `best_qpos.npy` as the common action sequence.

Activate only the pinned repository Wrapping position randomization:

```text
pos_bound = (-0.025, -0.01, 0.025, 0.01)
```

No mass/radius/stiffness/friction randomization is added.

Budget:

```text
initial:
32 env × 4 batches = 128 rollouts

if discovery candidates < 5:
collect only 12 additional batches

maximum:
32 env × 16 batches = 512 rollouts
```

The initial 128 are not recollected during scale-up.

---

## 5. Action equivalence

Only same-time states are compared:

```text
rollout A @ t
rollout B @ t
```

All rollouts replay the identical `best_qpos.npy`, so action history is equal by construction.

No cross-time mining is performed in PB2-C.

---

## 6. Artificial partial-state rope observation

PB2-C does not have a camera reconstruction pipeline. Therefore do not call this a deployable observation.

Use the term:

> **artificial partial-state discovery observation**

The rope component is:

```text
3-frame history
of simulator rope-vertex XYZ
with post-local vertices hidden
```

Published geometry:

```text
post radius = 0.015 m
rope radius = 0.010 m
```

Frozen local occlusion radius:

```text
2 × (0.015 + 0.010) = 0.050 m
```

A rope vertex is hidden when its XY center lies within 50 mm of any post center.

Rope velocity is stored in raw data but is **not** part of PB2-C pair selection.

Visible-rope discovery metric:

```text
mean 3-frame symmetric XYZ Chamfer <= 0.010 m
```

The 10 mm threshold is a discovery tolerance, not formal Gate 1.

---

## 7. Robot observation definition

Mirror the published Wrapping robot component of `compute_observation()`.

Per Franka:

```text
EE position       3
EE quaternion     4
motor joint qpos  7
-------------------
14 dimensions / arm
```

Both arms are included.

Do not use finger qpos, motor force, controller force, reward, or hidden winding as robot observation.

PB2-C uses three interpretable history metrics:

```text
A. dual-arm mean EE position-history distance
   <= 0.010 m

B. dual-arm mean sign-invariant quaternion geodesic history distance
   <= 5 deg
   = 0.08726646259971647 rad

C. dual-arm mean motor-joint-qpos history RMS
   <= 0.050 rad
```

Quaternion distance:

```text
2 * acos(abs(dot(normalize(qA), normalize(qB))))
```

The three fixed post states are shared task context. Since PB2-C randomizes only the rope, no separate post-state near threshold is needed.

These robot thresholds are discovery tolerances only.

---

## 8. Hidden privileged state

Use PB2-A/B-validated signed task-native winding:

```text
W = [w1, w2, w3]
I = round(W)
```

Candidate hidden-state difference:

```text
I_A != I_B
```

on at least one post.

Do not reuse PB2-B `0.25 / 0.75` semantic bands as PB2-C pair thresholds.

---

## 9. Winding integer residual

For every state/post:

```text
r_j = |w_j - round(w_j)|
```

Every saved candidate must include:

```text
winding_integer_residual_a [3]
winding_integer_residual_b [3]

winding_integer_residual_linf_a
winding_integer_residual_linf_b

pair_max_winding_integer_residual
```

Residual is diagnostic evidence about the cleanliness of the integer topology label.

PB2-C REV1 does not invent a post-hoc residual pass threshold.

---

## 10. Exact same-time pair search

Only official-valid full rollouts are included.

Search all same-time hidden-index-different pairs exactly. Do not use kNN as the formal search.

Filter order:

```text
official-valid full rollout
→ same time
→ winding index different
→ non-empty 3-frame partial rope observation
→ robot EE-position near
→ robot EE-quaternion near
→ robot motor-qpos near
→ visible-rope history Chamfer near
→ discovery candidate
```

Future divergence, robot force, sensor separability, and reward are prohibited from selection.

A safe computational shortcut may reject a current-frame Chamfer larger than `history × final_mean_threshold`; this cannot create false negatives for the final non-negative history mean and is not a scientific gate.

---

## 11. Funnel

Report exact counts:

```text
total_rollouts
official_valid_rollouts
official_invalid_rollouts

total_valid_same_time_pair_comparisons
hidden_winding_index_different
nonempty_partial_rope_history

robot_ee_position_pass
robot_ee_quaternion_pass
robot_motor_qpos_pass
robot_observation_pass

visible_history_chamfer_pass
candidate_count
```

Also report mixed winding-index class histograms and valid-state global residual statistics.

---

## 12. Candidate ranking

Rank top 50 by:

```text
1. visible-rope history Chamfer ascending
2. EE-position history distance ascending
3. EE-quaternion history distance ascending
4. motor-qpos RMS ascending
5. number of differing winding posts descending
```

Integer residual is saved but is not used to manufacture a pass.

---

## 13. Decision

If candidate count >= 5:

```text
PB2C_NATURAL_WINDING_PAIRS_FOUND
```

Stop PB2-C and proceed directly to PB3.

If initial 128 gives <5, perform the single precommitted scale-up to 512 total.

If 512 total still gives <5:

```text
PB2C_INSUFFICIENT_NATURAL_PAIRS_UNDER_FROZEN_PROTOCOL
```

Do not lower thresholds, enlarge the mask, modify Wrapping physics, or invent another hidden proxy.

This does not prove Wrapping contains no CCDA.

---

## 14. PB3 boundary

PB3, not PB2-C, tests:

```text
snapshot A/B
same future command
multi-horizon divergence
deterministic repeat / uncertainty floor
```

No `future/current >= 2`.

Do not start StateDiff / CFPM / IDM training before PB3 and later control-relevance validation.
