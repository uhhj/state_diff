# CCDA V10.3 — PB4-C1 Feasibility-Normalized Signed Qpos-Target Exploration

> Start main SHA: `c95e8211179b9e65a5a85996fdfaa0d1883deae8`  
> DLO-Lab gitlink: `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`

## 1. Entry state

Frozen scientific state:

```text
PB3-B2  same-action future bifurcation     CONFIRMED
PB4     4-action control relevance         NOT CONFIRMED
PB4-C0  normalized qpos family             FROZEN / CPU-ONLY
Condition 5                                UNCONFIRMED
```

PB4-C0 froze exactly:

```text
alpha_max_hard_limits = 0.26808133907545595
alpha_formal          = 0.25467727212168312

unit {-1,0,+1}² grid:
  98 violating commands
  114 joint-limit events

formal {-alpha,0,+alpha}² grid:
  360 commands
  0 hard-limit violations
```

PB4-C1 is allowed to use only the already-frozen family in:

```text
configs/experiment3/published_benchmark/dlolab_wrapping_pb4c_family.json
```

It must not recompute alpha.

---

## 2. Scientific role

PB4-C1 is an **exploratory** control-relevance audit.

It reuses the same prospective PB3-B1 cohort after the original PB4 outcome is
already known. Therefore PB4-C1 can test whether the feasibility-normalized
signed qpos-target hypothesis shows a strong signal, but it cannot provide
confirmatory Condition-5 evidence.

Required methodological invariant:

```text
classification = EXPLORATORY
formal_condition5_confirmed = false
condition5_confirmation_allowed = false
```

A positive result only motivates a new independent prospective cohort.

---

## 3. Frozen intervention family

PB4-C1 reads the family JSON verbatim.

The frozen global scalar is:

\[
\alpha = 0.25467727212168312.
\]

The per-arm mask set is:

\[
\{-\alpha,0,+\alpha\}.
\]

For branch time `t`, relative microstep `r`, and arm `i`:

\[
q_i^a(r)
=
q_i(t)
+
m_i(a)
\left[q_i^{nom}(t+r)-q_i(t)\right].
\]

The exact nine actions are:

```text
both_negative
  (-alpha,-alpha)

arm1_negative_arm2_hold
  (-alpha,0)

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

Terminology remains qpos-target-specific. `negative` means sign reversal of the
nominal joint-target displacement, not guaranteed Cartesian end-effector motion
reversal.

---

## 4. Family lock

PB4-C1 must not:

```text
recompute alpha
round alpha
rescale alpha
clip qpos targets
change action IDs
change action order
change H
introduce per-joint alpha
introduce per-arm alpha
introduce per-pair alpha
```

Before protocol freeze, PB4-C1 revalidates the exact frozen mask values against
the current pinned Panda hard limits and frozen `best_qpos`.

This is only a drift check. It must not call the PB4-C0 alpha derivation.

Require:

```text
pre_gpu_hard_limit_validation = true
out_of_limit_command_count = 0
```

If this fails:

```text
STOP BEFORE GPU
```

---

## 5. Formal horizon and repeat structure

Keep unchanged:

```text
H = 20 qpos microsteps
n_steps_sub = 10
3 repeats per branch-action
```

Formal matrix:

```text
20 branch states
× 9 actions
× 3 repeats
= 540 action-repeat records
```

No secondary horizon is formal.

---

## 6. Task score

Use the exact corrected PB4 published-score semantics.

At each microstep:

```text
rope NaN or stretch > 1.2
→ branch permanently inactive
→ current and later contributions = 0

reward NaN only
→ current contribution = 0
→ alive unchanged

otherwise
→ contribution = env.reward() + 1
```

Then:

\[
J(a)=\frac1{20}\sum_{h=1}^{20}c_h.
\]

Every action record retains the 20 `score_contributions` so CPU validation can
reconstruct the primary score.

---

## 7. Live and snapshot barriers

PB4-C1 must use the exact same frozen 10-pair cohort.

### Barrier A — fresh live pair revalidation

Require:

```text
10/10 pairs valid
```

including the existing PB4 prefix-validity condition:

```text
no stretch / rope-NaN failure at or before branch time t
```

Failure verdict:

```text
PB4C1_LIVE_COHORT_REVALIDATION_FAILED
```

No pair drop/replacement.

### Barrier B — native snapshot restore

Require:

```text
20 branch states
3 restore checks / branch
60/60 valid
50 um rope max-abs threshold
```

Failure verdict:

```text
PB4C1_SNAPSHOT_RESTORE_FAILED
```

Action evaluation begins only after both global barriers pass.

---

## 8. Control-relevance statistics

All formal PB4 statistics remain unchanged.

For each branch/action:

```text
median score = median of 3 repeats
repeat floor = max pairwise absolute repeat difference
```

For each branch:

```text
branch repeat floor = max repeat floor over 9 actions
```

Unique-best threshold:

\[
\tau_s=\max(0.05,5F_s).
\]

Unique best requires:

\[
J_{best,median}-J_{second,median}\ge\tau_s.
\]

For pair A/B with different unique best actions:

\[
R_A^{robust}
=
\min_r J_A(a_A^*,r)
-
\max_r J_A(a_B^*,r),
\]

and analogously for B.

Pair threshold:

\[
\tau_{pair}
=
\max(0.05,5\max(F_A,F_B)).
\]

A pair passes iff:

```text
A unique best
B unique best
best A != best B
robust regret A >= tau_pair
robust regret B >= tau_pair
```

---

## 9. Exploratory signal criterion

Keep unchanged:

```text
passing pairs >= 3
AND
passing PB4-C1 fresh-live winding strata >= 2
```

Positive exploratory verdict:

```text
PB4C1_NORMALIZED_QPOS_SIGNAL_FOUND
```

Complete negative exploratory verdict:

```text
PB4C1_NORMALIZED_QPOS_SIGNAL_NOT_FOUND
```

Blocked execution verdicts are not scientific negatives.

---

## 10. Winding-stratum semantics

Formal phase counting must use the winding indices from the PB4-C1 fresh live
revalidation at branch time.

Historical PB3-B1 winding strata may remain diagnostic only.

Do not require fresh live winding indices to equal the historical frozen
indices.

---

## 11. Evidence-preserving orchestration

PB4-C1 must keep caller-owned mutable buffers:

```text
live_records
restore_records
action_records
```

If a later barrier blocks, already acquired evidence must remain available.

Required behavior:

```text
live failure
→ keep actual live records

snapshot failure
→ keep live + actual restore records

action failure
→ keep live + 60 restore + partial action records
```

Do not use tuple assignment from the old PB4 `execute()` if an exception before
return could discard accumulated evidence.

---

## 12. Protocol preregistration

Before GPU, generate:

```text
configs/experiment3/published_benchmark/
  dlolab_wrapping_pb4c1_protocol.json
```

It must contain:

```text
classification = EXPLORATORY
formal_condition5_confirmed = false
condition5_confirmation_allowed = false

family source = dlolab_wrapping_pb4c_family.json
alpha_formal = exact frozen JSON value
9 exact action IDs/masks
H = 20

pre-GPU hard-limit violations = 0

repeat = 3
absolute regret floor = 0.05
repeat multiplier = 5
minimum passing pairs = 3
minimum passing fresh-live strata = 2

expected action records = 540
```

Protocol must be committed and pushed before formal GPU execution.

---

## 13. Outputs

Raw server evidence:

```text
/data/Experiment3/data/pb4c1_normalized_qpos_exploration/
  ACTION_EVALUATIONS.json
  ACTION_SCORES.npz
  AUDIT.json
```

Committed result:

```text
reports/experiment3/pb4c1_normalized_qpos_exploration/
  RESULT.md
  EVIDENCE.json
  PAIR_REGRET.json
```

All methodology artifacts must state:

```text
classification = EXPLORATORY
formal_condition5_confirmed = false
```

---

## 14. Positive stop rule

If:

```text
PB4C1_NORMALIZED_QPOS_SIGNAL_FOUND
```

STOP.

Allowed interpretation:

> The frozen feasibility-normalized qpos-target family produced an exploratory
> branch-dependent control signal on the already-observed cohort.

Not allowed:

```text
Condition 5 PASS
control relevance confirmed
```

Next allowed scientific step:

> build a new independent prospective cohort without using PB4-C1 action
> outcomes, then run a confirmatory audit with the exact same frozen family.

Do not create that cohort in PB4-C1.

---

## 15. Negative stop rule

If:

```text
PB4C1_NORMALIZED_QPOS_SIGNAL_NOT_FOUND
```

STOP the Wrapping control-relevance route.

Do not add:

```text
new alpha values
per-joint/per-arm/per-pair scales
new horizon
clipping
Cartesian rescue actions
hand-designed actions
```

At that point either narrow the paper's Wrapping claim to future-dynamics
ambiguity or move Condition 5 to a different published task.

---

## 16. Project boundary

PB4-C1 does not authorize:

```text
deployable sensing
StateDiff
StateDiff-FT
CFPM
training
```

If PB4-C1 is positive, confirmation still comes first. If PB4-C1 is negative,
Wrapping control-relevance engineering stops.
