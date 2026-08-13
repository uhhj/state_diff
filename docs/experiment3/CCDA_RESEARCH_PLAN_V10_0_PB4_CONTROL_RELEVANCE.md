# CCDA V10.0 — PB4 Wrapping Control Relevance / Action Regret

> Start main SHA: `8e076445cc95caa8b4015ed8d7ab5900034af4d8`  
> DLO-Lab gitlink: `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`

## 1. Entry condition

PB3-B2 is complete:

```text
PB3_CAUSAL_FUTURE_BIFURCATION_CONFIRMED
```

The frozen prospective PB3-B1 cohort has:

```text
10 pairs
20 unique rollouts
2 winding strata
10/10 fresh live revalidation
20/20 branch snapshots
60/60 restore records
10/10 Gate-4 passing pairs
```

PB3 therefore establishes CCDA Condition 4:

```text
same observable/action history
+ hidden winding difference
+ same future action
→ repeat-robust future deformation bifurcation
```

PB4 must now test Condition 5:

```text
hidden branch
→ different control preference / significant action regret
```

PB4 does not test sensing or model performance.

---

## 2. Scientific question

For each frozen PB3 pair A/B, evaluate the same finite action library on both
hidden branches.

Let:

\[
J_s(a)
\]

be the branch-local task score for branch `s` and action `a`.

PB4 asks whether:

\[
a_A^* \neq a_B^*
\]

and whether using the other branch's preferred action incurs robust regret:

\[
R_A = J_A(a_A^*) - J_A(a_B^*)
\]

\[
R_B = J_B(a_B^*) - J_B(a_A^*)
\]

The action library and regret thresholds must be committed before GPU action
evaluation.

---

## 3. Why PB4 uses the verified qpos replay interface

The published Wrapping CMA-ES action is 12D:

```text
arm1 xyz
arm2 xyz
arm1 rotation
arm2 rotation
```

and DLO-Lab converts the optimized trajectory into `best_qpos.npy`.

The published `eval_traj(..., qpos=...)` path directly applies the resulting
joint-position sequence to the two Frankas.

The current CCDA Wrapping chain already validated this exact `best_qpos`
position-control replay in PB2/PB3.

PB4 therefore uses the verified qpos actuator interface and does not introduce a
new IK/control stack after snapshot restore.

---

## 4. Frozen action library

At branch time `t`, let:

```text
q_t = official best_qpos[t]
q_nom(r) = official best_qpos[t+r]
```

Each Franka occupies one 9-DOF qpos block.

For arm `i`, define:

\[
q_i^a(r)
=
q_i(t)
+
m_i(a)
[
q_i(t+r)-q_i(t)
]
\]

with `m_i in {0,1}`.

This creates exactly four actions:

```text
A0 both_continue
   arm1 mask = 1
   arm2 mask = 1

A1 arm1_continue_arm2_hold
   arm1 mask = 1
   arm2 mask = 0

A2 arm1_hold_arm2_continue
   arm1 mask = 0
   arm2 mask = 1

A3 hold_both
   arm1 mask = 0
   arm2 mask = 0
```

This set is symmetric, finite, and defined from the published two-arm control
structure only.

It does not use:

```text
hidden winding index
PB3 pair future divergence magnitude
PB4 score
pair identity
winding stratum
```

No actions may be added/removed after GPU execution.

---

## 5. Formal action horizon

Freeze:

```text
H = 20 qpos microsteps
```

DLO-Lab Wrapping uses:

```text
n_steps_sub = 10
```

so this is exactly:

```text
2 published CMA-ES macro-step equivalents
```

The same H is used for all pairs.

No multi-horizon cherry-picking is allowed in PB4.

---

## 6. Formal task score

Use the published Wrapping reward function directly via:

```python
env.reward()
```

The published CMA-ES default uses cumulative reward, not last-state reward.

PB4 uses a fixed-horizon normalized cumulative score:

\[
J(a)
=
\frac1H
\sum_{h=1}^H
c_h
\]

where:

```text
if branch is alive and reward is finite after microstep h:
    c_h = reward_h + 1
else:
    c_h = 0
```

This mirrors the contribution used by published Wrapping `eval_traj()` for
cumulative reward, normalized by fixed H so the score remains in per-step reward
units.

Higher is better.

### Physical failure

A branch becomes permanently inactive only from the first microstep with:

```text
rope NaN
stretch ratio > 1.2
```

A `reward NaN` by itself contributes zero for that microstep but does **not**
permanently clear `alive`, matching the pinned Wrapping `eval_traj()` cumulative
reward semantics.

A physically failing action is **not removed**. After physical failure, its
remaining score contributions are zero, making failure a legitimate control
consequence.

---

## 7. Why the absolute regret threshold is 0.05

The published Wrapping reward contains:

```text
winding_reward = 1 - winding_loss
```

whose natural winding component has unit scale.

Freeze:

```text
absolute_regret_min = 0.05
```

i.e. 5% of one winding-reward unit on the normalized per-step score scale.

This prevents deterministic-but-microscopic action preferences from being
called control relevant.

No post-GPU threshold change.

---

## 8. Repeats and uncertainty

For every:

```text
20 branch states
x 4 actions
x 3 snapshot repeats
```

evaluate:

```text
240 formal branch-action-repeat records
```

For each branch/action:

```text
median score = median of 3 repeats

repeat floor =
max pairwise absolute score difference among the 3 repeats
```

For each branch:

```text
branch repeat floor =
max action repeat floor across the 4 actions
```

---

## 9. Unique best action

For each branch, rank the four actions by median task score.

Let:

```text
best median
second-best median
```

Freeze branch threshold:

\[
\tau_s
=
\max(
0.05,
5 F_s
)
\]

where `F_s` is the branch repeat floor.

The branch has a formal unique best action iff:

\[
J_{best,median}
-
J_{second,median}
\ge
\tau_s
\]

A nominal argmax with a tiny gap is not sufficient.

---

## 10. Robust cross-branch regret

For a pair, if the two unique best action IDs differ, define:

\[
R_A^{robust}
=
\min_r J_A(a_A^*,r)
-
\max_r J_A(a_B^*,r)
\]

\[
R_B^{robust}
=
\min_r J_B(a_B^*,r)
-
\max_r J_B(a_A^*,r)
\]

This is deliberately stronger than median regret: every repeat of the
branch-optimal action must outperform every repeat of the other branch's
preferred action by the formal margin.

Pair threshold:

\[
\tau_{pair}
=
\max(
0.05,
5\max(F_A,F_B)
)
\]

A pair passes PB4 iff:

```text
branch A has a unique best action
branch B has a unique best action
best action A != best action B
robust regret A >= tau_pair
robust regret B >= tau_pair
```

---

## 11. Phase criterion

Mirror the PB3 anti-one-off structure:

```text
passing pairs >= 3
AND
passing PB4 fresh-live winding strata >= 2
```

Positive:

```text
PB4_CONTROL_RELEVANCE_CONFIRMED
```

Negative:

```text
PB4_CONTROL_RELEVANCE_NOT_CONFIRMED
```

A negative result means:

> control relevance was not confirmed under this frozen finite action library.

It does not prove no possible action space could reveal relevance.

---

## 12. Pre-action global barriers

PB4 must use the exact same frozen PB3-B1 cohort.

Before action evaluation:

### Barrier A — fresh live pair revalidation

Require:

```text
10/10 pairs
```

under the already-frozen live PB2-C semantics **plus explicit PB3-B1 physical
prefix validity through branch time**.

For every branch at time `t`, PB4 must record:

```text
prefix_valid_through_t
first_fail_step
failed_stretch_through_t
failed_rope_nan_through_t
```

and require:

```text
no stretch / rope-NaN failure at or before t
```

equivalently:

```text
first_fail_step is absent
OR
first_fail_step > t
```

Any failure:

```text
PB4_LIVE_COHORT_REVALIDATION_FAILED
```

No action evaluation begins.

### Barrier B — native snapshot restore

Capture the same 20 branch states.

Use:

```text
3 restores / branch
50 um rope max-abs restore threshold
```

Require:

```text
60/60 restore records
```

Any failure:

```text
PB4_SNAPSHOT_RESTORE_FAILED
```

No action evaluation begins.

No pair drop/replacement.

---

## 13. Execution matrix

After both barriers:

```text
20 branches
x
4 fixed actions
x
3 repeats
=
240 action evaluations
```

Each action run starts from the native branch snapshot.

The action sequence is identical for A and B when pair/time/action are the same.

No branch-specific action construction is permitted.

---

## 14. Outputs

Pre-GPU committed:

```text
configs/experiment3/published_benchmark/
  dlolab_wrapping_pb4.json
  dlolab_wrapping_pb4_protocol.json

scripts/experiment3/dlolab_wrapping/
  control_relevance_pb4.py
  run_pb4.sh

tests/experiment3/dlolab_wrapping/
  test_control_relevance_pb4.py
```

Raw server evidence:

```text
/data/Experiment3/data/pb4_control_relevance/
  ACTION_EVALUATIONS.json
  ACTION_SCORES.npz
  AUDIT.json
```

Committed result:

```text
reports/experiment3/pb4_control_relevance/
  RESULT.md
  EVIDENCE.json
  PAIR_REGRET.json
```

---

## 15. PB4 success boundary

If PB4 is positive, the task-mechanism chain is complete through Condition 5:

```text
observable equivalence        PASS
past-action equivalence       PASS
hidden winding difference     PASS
same-action future branch     PASS
control relevance             PASS
```

At that point stop task/control-mechanism engineering.

The next scientific requirement becomes:

```text
deployable sensor observability
```

Only after deployable sensing is established should the project prepare
StateDiff B0/B1 and CFPM training data.

PB4 success alone does not authorize model training.

---

## 16. Hard stop

After PB4 result is committed:

```text
STOP
```

Do not start:

```text
sensor experiment
StateDiff training
StateDiff-FT training
CFPM
PB5
```

inside the same execution phase.


---

## 17. PB4 live stratum accounting

The phase-level winding-stratum count must come from **PB4 fresh live
revalidation**, not the historical PB3-B1 frozen winding indices.

For every pair report both:

```text
pb3b1_frozen_winding_stratum
pb4_live_winding_stratum
stratum_match
```

but only:

```text
pb4_live_winding_stratum
```

is used by the formal:

```text
passing winding strata >= 2
```

criterion.

PB4 does not require the fresh live stratum to equal the historical frozen
stratum; that historical equality is diagnostic only.

---

## 18. Score reconstructability

Every branch-action-repeat record must retain exactly 20:

```text
score_contributions
```

values.

CPU result validation recomputes:

```text
task_score = sum(score_contributions) / 20
```

and requires agreement with the stored task score.

This allows the primary PB4 score, action medians, repeat floors and regret to be
reconstructed from retained formal artifacts.

---

## 19. Blocked-verdict validation

`validate-results` must support both scientific results and pre-action/action
blocked results.

Formal completed verdicts:

```text
PB4_CONTROL_RELEVANCE_CONFIRMED
PB4_CONTROL_RELEVANCE_NOT_CONFIRMED
```

require:

```text
10/10 live barrier
60/60 snapshot restore
240 action records
non-null regret audit
```

Blocked verdicts are validated by their execution boundary:

```text
PB4_LIVE_COHORT_REVALIDATION_FAILED
  → action records = 0
  → snapshot restore records = 0
  → audit = null

PB4_SNAPSHOT_RESTORE_FAILED
  → live barrier passed
  → action records = 0
  → audit = null

PB4_ACTION_EVALUATION_FAILED
  → live barrier passed
  → snapshot barrier passed
  → audit = null
  → validate retained partial action-record boundary
```

A blocked result must not be forced through the full regret analyzer.
