# CCDA V10.2 — PB4-C0 Feasibility-Normalized Signed Qpos-Target Family

> Start main SHA: `795ce37d308a9651db26cb9d6d83de05e154c0c9`  
> DLO-Lab gitlink: `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`

## 1. Entry state

The frozen formal PB4 result remains:

```text
PB4_CONTROL_RELEVANCE_NOT_CONFIRMED
```

The original four-action experiment was a complete scientific negative:
10/10 live pairs passed, 60/60 snapshot restores passed, and all 240
branch-action-repeat records were evaluated.

A later CPU-only signed qpos-target preflight for the full unit grid
`{-1,0,+1}²` did not reach protocol freeze because some mirrored qpos targets
violated the pinned Panda hard position limits. No GPU, protocol commit, or
scientific PB4-B result was produced.

PB4-C0 is therefore **not** a PB4 retry and does not run a control experiment.

It is a CPU-only engineering derivation whose only purpose is to freeze one
globally feasible symmetric qpos-target family without consulting reward,
regret, winding labels, or future divergence.

---

## 2. Scientific/engineering question

Let the verified nominal qpos suffix be:

\[
d_i(t,r)
=
q_i^{nom}(t+r)-q_i(t)
\]

for arm `i`.

Instead of fixing the signed mask amplitude to `1`, derive the largest globally
feasible symmetric amplitude from the pinned hard joint limits.

For each qpos DOF:

\[
q^a(t,r)
=
q(t)
+
m d(t,r),
\]

with a future family:

\[
m\in\{-\alpha,0,+\alpha\}.
\]

The scale must be **global**:

```text
same alpha
for every pair
for both arms
for every joint
for every r=1..20
```

No per-joint, per-arm, per-pair, or outcome-dependent scaling is permitted.

---

## 3. Frozen derivation rule

For a branch-time target `q_t`, hard limits `[l,u]`, and nominal delta `d != 0`,
both signs must be feasible:

\[
l
\le
q_t\pm\alpha d
\le
u.
\]

Therefore the largest symmetric scale allowed by that scalar DOF is:

\[
\alpha_{limit}
=
\frac{\min(q_t-l, u-q_t)}
{|d|}.
\]

The global hard-limit scale is:

\[
\alpha_{max}
=
\min_{t,r,arm,j}
\alpha_{limit}.
\]

Zero nominal deltas impose no scale constraint.

This uses only:

```text
frozen PB3-B1/PB4 branch times
official best_qpos.npy
pinned Panda URDF hard position limits
H = 20
```

It does not use:

```text
PB4 scores
PB4 regret
PB4 best-action identities
PB3 branch-distance magnitudes
winding labels
sensor data
```

---

## 4. Cap and feasibility margin

Freeze before derivation:

```text
symmetric_scale_cap = 1.0
feasibility_safety_factor = 0.95
```

Then:

\[
\alpha_{reference}
=
\min(\alpha_{max},1)
\]

and:

\[
\boxed{
\alpha_{formal}
=
0.95\alpha_{reference}
}.
\]

The cap forbids amplification beyond the original nominal suffix.

The 0.95 factor is an engineering feasibility margin, not a scientific outcome
threshold. It is fixed before `alpha_max` is computed.

Do not change `0.95` after seeing the derived scale.

---

## 5. Position-limit source

Use exactly the Panda asset loaded by pinned Wrapping:

```text
external/dlo-lab/genesis/assets/urdf/panda_bullet/panda.urdf
```

The 9 qpos entries per arm are:

```text
panda_joint1          revolute, rad
panda_joint2          revolute, rad
panda_joint3          revolute, rad
panda_joint4          revolute, rad
panda_joint5          revolute, rad
panda_joint6          revolute, rad
panda_joint7          revolute, rad
panda_finger_joint1   prismatic, m
panda_finger_joint2   prismatic, m
```

Parse the hard XML `<limit lower=... upper=...>` values.

Comparison tolerance:

```text
1e-12 native units
```

No clipping is allowed.

---

## 6. Unit-scale diagnostic

PB4-C0 must independently regenerate the unit grid:

```text
{-1,0,+1}²
```

for:

```text
all unique frozen branch times
× r=1..20
```

and count:

```text
generated_command_count
out_of_limit_command_count
joint_limit_violation_event_count
first_violation
```

This is diagnostic/provenance only.

The known PB4-B preflight blocker should be reproducible from the same frozen
inputs, but the exact count is **not hard-coded** into the derivation rule.

The static rule requires:

```text
unit-scale grid has at least one violation
```

because PB4-C0 exists specifically to normalize an infeasible unit-scale
family.

If the unit grid unexpectedly has zero violations:

```text
STOP
```

because the previous feasibility diagnosis is inconsistent and should be
resolved before changing the action family.

---

## 7. Limiting constraint evidence

The derivation must report the exact scalar constraint that determines
`alpha_max`:

```text
time_index
relative_step
arm_index
joint_index
joint_name
joint_type
unit

base_qpos
nominal_qpos
nominal_delta

lower
upper
lower_slack
upper_slack

limiting_bound
limiting_unit_mask_sign
alpha_limit
```

This makes `alpha_formal` mechanically reproducible.

---

## 8. Frozen formal family

After deriving `alpha_formal`, generate exactly the 3×3 grid:

```text
both_negative
  (-alpha,-alpha)

arm1_negative_arm2_hold
  (-alpha, 0)

arm1_negative_arm2_positive
  (-alpha,+alpha)

arm1_hold_arm2_negative
  (0,-alpha)

hold_both
  (0,0)

arm1_hold_arm2_positive
  (0,+alpha)

arm1_positive_arm2_negative
  (+alpha,-alpha)

arm1_positive_arm2_hold
  (+alpha,0)

both_positive
  (+alpha,+alpha)
```

Terminology is qpos-target-specific:

```text
negative = sign reversal of nominal joint-target displacement
positive = nominal sign
```

Do not claim exact Cartesian end-effector motion reversal.

---

## 9. Formal-family validation

Re-enumerate:

```text
all unique branch times
× 9 actions
× r=1..20
```

and require:

```text
out_of_limit_command_count = 0
```

Record:

```text
generated_command_count
minimum_revolute_joint_limit_margin_rad
minimum_prismatic_finger_limit_margin_m
```

No clipping or saturation repair.

If any formal command is outside hard limits:

```text
PB4C0_FEASIBILITY_NORMALIZED_FAMILY_DERIVATION_FAILED
STOP
```

Do not adjust alpha, safety factor, horizon, or individual joints inside the
same phase.

---

## 10. Two-commit preregistration structure

PB4-C0 is CPU-only, but the alpha rule must still be frozen before deriving the
actual alpha.

### Commit A — derivation rule

Implement code/config/tests and run:

```text
freeze-rule
```

which generates:

```text
dlolab_wrapping_pb4c0_feasibility_rule.json
```

This rule contains:

```text
H=20
global alpha only
cap=1.0
safety factor=0.95
formula
hard-limit source
no clipping
no reward/regret/winding/future inputs
```

Commit and push this rule **before** running `derive-family`.

### Commit B — derived family result

Only after Commit A:

```text
derive-family
validate-result
```

Generate and commit the actual `alpha_max`, `alpha_formal`, action grid and CPU
evidence.

No GPU occurs in either commit.

---

## 11. Verdicts

Rule freeze:

```text
PB4C0_FEASIBILITY_RULE_PREREGISTERED
```

Successful derivation:

```text
PB4C0_FEASIBILITY_NORMALIZED_FAMILY_FROZEN
```

Failure before a family can be frozen:

```text
PB4C0_FEASIBILITY_NORMALIZED_FAMILY_DERIVATION_FAILED
```

PB4-C0 has no control-relevance positive/negative verdict.

---

## 12. Outputs

Static committed input:

```text
configs/experiment3/published_benchmark/
  dlolab_wrapping_pb4c0_feasibility.json
```

Generated rule, committed before derivation:

```text
configs/experiment3/published_benchmark/
  dlolab_wrapping_pb4c0_feasibility_rule.json
```

Generated successful family:

```text
configs/experiment3/published_benchmark/
  dlolab_wrapping_pb4c_family.json
```

Committed report:

```text
reports/experiment3/pb4c0_feasibility_normalized_family/
  RESULT.md
  EVIDENCE.json
```

No raw GPU directory is created.

---

## 13. Success boundary

If:

```text
PB4C0_FEASIBILITY_NORMALIZED_FAMILY_FROZEN
```

then STOP.

Do not automatically start PB4-C1.

The next phase, if pursued, must separately preregister an exploratory GPU
audit using exactly:

```text
alpha_formal from dlolab_wrapping_pb4c_family.json
same 9 actions
H=20
3 repeats
0.05 regret floor
5× repeat floor
3 pairs / 2 fresh-live strata
```

PB4-C1 must not recompute or tune alpha.

---

## 14. Failure boundary

If PB4-C0 cannot freeze a feasible global symmetric family:

```text
STOP
```

Do not introduce inside this phase:

```text
per-joint alpha
per-arm alpha
per-pair alpha
clipping
different H
different safety factor
Cartesian rescue action
```

Any such proposal would require a new separately justified action family.

---

## 15. Project boundary

PB4-C0 remains pre-Condition-5.

Even on success:

```text
Condition 5                 UNCONFIRMED
deployable sensing          UNTESTED
StateDiff/CFPM training     NOT STARTED
```

No sensing or model training is authorized by this CPU feasibility result.
