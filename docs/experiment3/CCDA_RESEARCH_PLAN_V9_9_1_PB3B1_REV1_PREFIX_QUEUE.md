# CCDA V9.9 — PB3-B1 Prospective Live-Pair Cohort Discovery

> Start main SHA: `1545b89224bb55cc5b396048dba986f53de49b1e`  
> DLO-Lab gitlink: `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`

## 1. Why the historical PB3 cohort is closed

The current historical formal PB3 cohort must remain archived.

Formal result:

```text
PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED
failure_component = targeted_replay_alignment
```

All 20 formal branch states were evaluated. Seventeen passed the independently
pre-registered R3 engineering rule; rollouts 39, 55 and 97 at t=20 failed.
Rollout 55 also changed its rounded winding index.

No live-pair revalidation, branch snapshot, future suffix or Gate 4 was run.

Therefore the historical frozen cohort produced no causal future result. It is
not scientifically valid to:
- raise the R3 threshold using these formal failures;
- drop 39/55/97;
- replace them with later PB2-C candidates;
- call the result a negative Gate-4 result.

The next experiment must be prospective.

---

## 2. Scientific question

PB3-B1 asks only:

> Under a fresh replay of the unchanged published Wrapping task, can we
> prospectively construct a rollout-disjoint cohort of 10 live branch pairs
> that satisfy the original PB2-C observation/history and task-native winding
> conditions, without using any future divergence information?

PB3-B1 does not test causal future bifurcation.

The desired causal experiment remains PB3-B2.

---

## 3. Key design change

Historical route:

```text
old PB2-C frozen acquisition
→ later reconstruct branch state
→ frozen/live engineering alignment
→ causal suffix
```

Prospective route:

```text
precommitted frozen candidate ordering
→ fresh live replay
→ evaluate live t-2:t pair semantics
→ freeze first 10 live-valid rollout-disjoint pairs
→ STOP
```

PB3-B2 will later replay this prospective cohort, revalidate the live
conditions again, and only then create snapshots / future suffixes.

The R3 `61.945756736 µm` historical reconstruction rule remains archived and is
**not** a PB3-B1 admission gate.

---

## 4. Independence from already observed formal outcomes

PB3-B1 excludes two fixed 20-rollout sets before any GPU live screening.

### Historical formal PB3 rollouts

The 20 rollouts in the frozen 10-pair PB3 shortlist.

Reason:

```text
their formal replay-alignment outcomes are already observed
```

### PB3-R2 calibration rollouts

The 20 independent rollouts used to calibrate the R3 reconstruction envelope.

Reason:

```text
keep the prospective causal cohort independent of threshold-calibration data
```

These two sets are already disjoint, so:

```text
fixed exclusion union = 40 rollouts
```

No outcome from the blocked historical PB3 is used to rank the remaining
candidates.

---

## 5. Candidate source and ranking

Do not use only the committed PB2-C top-50 shortlist.

PB3-B1 REV1 must not call historical `mine_pairs()` to construct its
prospective queue: that miner first filters by full-rollout
`official_rollout_valid`, which conditions candidate admission on survival
after the candidate branch time.

Instead, rebuild a PB2-C-style queue directly from the frozen 128-rollout raw
data with time-local prefix validity. For candidate time `t`, rollout `i` is
eligible only when all selector fields in frames `t-2:t` are finite and it has
never failed or its first rope-NaN/stretch failure step is strictly after `t`.
A rollout that fails after `t` remains eligible at `t`.

Queue admission and ranking must ignore `official_rollout_valid`, final reward,
post-`t` failure, future winding, and future distance/divergence. Existing
PB2-C helpers for visibility, winding, robot history, Chamfer, and winding
integer residual are reused; the original PB2-C ranking key is retained.

This produces a `prefix_candidate_count`, which is not required to equal
5,534. Remove every candidate containing one of the fixed 40 excluded
rollouts, preserve the prefix-only source rank, and do not re-rank the
remaining candidates.

The historical full-survivor subset remains a regression diagnostic only:
filter the prefix-only candidates to pairs whose two rollouts have
`official_rollout_valid == true`, then require historical candidate count
5,534 and matching committed top-50 signatures. That validated subset must
never be used as the prospective queue.

PB2-C ranking remains:

```text
1. visible rope history Chamfer ascending
2. dual EE-position history distance ascending
3. quaternion geodesic history ascending
4. motor-qpos history RMS ascending
5. differing-post count descending
```

No future, force or sensor field is introduced into ranking.

---

## 6. Pre-GPU protocol freeze

PB3-B1 uses a two-commit workflow, not an extra scientific phase.

Before GPU live screening, CPU-only code generates:

```text
configs/experiment3/published_benchmark/
  dlolab_wrapping_pb3b1_protocol.json
```

The protocol records:

```text
prefix_candidate_count (no fixed equality requirement)
selection_used_full_rollout_survival = false
selection_used_final_reward = false
historical_survivor_subset_validation:
  candidate_count = 5534
  top_k_validated = true
eligible candidate count after fixed exclusions
exact 40 rollout exclusions
candidate ranking rule
live admission thresholds
first-10 greedy cohort rule
two-stratum proceed requirement
no drop/replacement
future/sensor/force/reward non-use
```

This protocol must be committed before the fresh replay.

The screening command recomputes the protocol and requires exact equality.

---

## 7. Fresh live collection

Use the unchanged published Wrapping setup:

```text
n_envs = 32
n_steps_sub = 10
batch indices = 0,1,2,3
seed = 123 + batch
repo Wrapping pos_bound only
same official best_qpos.npy
```

Reuse the committed PB2-C `replay_batch()` implementation directly.

One fresh Python process may replay the four batches sequentially.

This produces:

```text
128 fresh live rollouts
```

No physics/reward/geometry modification.

Raw live screen data:

```text
/data/Experiment3/data/pb3b1_prospective_live_cohort/
  LIVE_SCREEN_ROLLOUTS.npz
```

---

## 8. Important future-information boundary

`replay_batch()` naturally produces the full common-qpos trajectory, but
PB3-B1 pair admission may access only information at or before each candidate
branch time.

For a candidate at time `t`, formal admission uses only:

```text
t-2
t-1
t
```

plus failure status up to `t`.

A fresh rollout that fails stretch/rope-NaN **after** the candidate branch time
does not invalidate the PB3-B1 branch point.

Formal live branch validity is:

```text
selection fields finite through t
AND
no rope-NaN/stretch failure at or before t
```

PB3-B1 explicitly does **not** use:

```text
post-branch divergence
post-branch pair distance
post-branch winding evolution
final reward
future success
force
sensor
```

for cohort admission.

This avoids choosing the cohort using the future quantity PB3-B2 is meant to
test.

---

## 9. Live pair admission

For each eligible frozen-ranked candidate, evaluate the fresh live pair at the
candidate's frozen candidate time.

### Same time / action

```text
same time: required
common official qpos action history: equal by construction
```

### Live task-native hidden state

At branch time:

```text
round(signed winding turns A)
!=
round(signed winding turns B)
```

No historical frozen/live winding equality is required.

No winding-residual threshold is added.

### Artificial partial-rope history

Reuse original PB2-C semantics:

```text
history = 3 frames
occlusion radius = 0.05 m
nonempty visible rope on both branches at every history frame
mean symmetric 3D visible-history Chamfer <= 0.01 m
```

This remains a discovery surrogate, not a deployable perception claim.

### Robot history

Reuse original PB2-C thresholds:

```text
dual EE-position history mean <= 0.01 m
sign-invariant quaternion geodesic history mean <= 0.08726646259971647 rad
dual motor-qpos history RMS <= 0.05 rad
```

### Historical alignment

The following may not gate PB3-B1:

```text
R3 coordinate-RMSE <= 61.945756736 µm
historical rope max-abs
historical frozen winding equality
```

The new experiment is prospective.

---

## 10. Frozen cohort-selection algorithm

Scan eligible candidates in preserved PB2-C source-rank order.

For each candidate:

```text
if either rollout is already used by an accepted pair:
    skip

evaluate fresh live admission

if live admission passes:
    accept
    mark both rollouts used
```

Stop immediately when:

```text
10 pairs
20 unique rollouts
```

have been accepted.

Do not continue scanning to improve the cohort after the first 10.

This is crucial.

The cohort is therefore exactly:

> first 10 fresh-live-valid, rollout-disjoint pairs under the precommitted
> frozen ranking and fixed exclusions.

---

## 11. Winding-stratum rule

After the first 10 accepted pairs are frozen, count their live winding strata.

A stratum is the unordered pair of rounded winding-index vectors.

To make the unchanged PB3 phase criterion testable, PB3-B2 requires at least:

```text
2 live winding strata
```

However PB3-B1 must not continue scanning after the first 10 merely to rescue
stratum diversity.

Therefore:

```text
10 pairs and >=2 strata
→ PB3B1_PROSPECTIVE_LIVE_COHORT_FROZEN

<10 pairs
→ PB3B1_PROSPECTIVE_LIVE_COHORT_INSUFFICIENT

10 pairs but <2 strata
→ PB3B1_PROSPECTIVE_LIVE_COHORT_STRATA_INSUFFICIENT
```

The latter two are bounded screening outcomes, not claims that Wrapping lacks
CCDA.

---

## 12. Outputs

Pre-GPU committed:

```text
configs/experiment3/published_benchmark/
  dlolab_wrapping_pb3b1_protocol.json
```

Raw GPU evidence:

```text
/data/Experiment3/data/pb3b1_prospective_live_cohort/
  LIVE_SCREEN_ROLLOUTS.npz
  SCREEN_LOG.json
```

Post-screen committed:

```text
configs/experiment3/published_benchmark/
  dlolab_wrapping_pb3b1_cohort.json

reports/experiment3/pb3b1_prospective_live_cohort/
  RESULT.md
  EVIDENCE.json
```

No future trajectories or Gate-4 pair metrics are generated.

---

## 13. Independent post-screen validation

After live screening, CPU-only validation reloads:

```text
LIVE_SCREEN_ROLLOUTS.npz
frozen PB2-C source data/config
committed PB3-B1 protocol
```

It deterministically re-runs the candidate scan and requires exact equality
with:

```text
dlolab_wrapping_pb3b1_cohort.json
```

This is one targeted validation, not a new audit phase.

---

## 14. Hard stop

If verdict is:

```text
PB3B1_PROSPECTIVE_LIVE_COHORT_FROZEN
```

commit the cohort and stop.

Do not run snapshots, future suffixes or Gate 4 in PB3-B1.

Next:

```text
PB3-B2
→ replay the same frozen prospective cohort
→ revalidate all 10 live pairs again
→ snapshot
→ 3 same-action repeats
→ horizons 1,2,5,10,20,40
→ unchanged Gate 4
```

If PB3-B1 is insufficient, do not lower PB2-C thresholds in the same phase.


# REV1 — Remove future-survival bias from the candidate queue

The initial PB3-B1 draft still inherited one future-conditioning path:

```text
PB2-C mine_pairs()
→ official_rollout_valid only
→ candidate queue
```

`official_rollout_valid` is a full-rollout survival label. Therefore even if
fresh live admission reads only `t-2:t`, the source queue has already removed
rollouts that are valid at branch time but fail later.

This is not acceptable for a prospective branch-point cohort.

## Corrected frozen queue

PB3-B1 REV1 must NOT call historical `mine_pairs()` to form its prospective
queue.

Instead, rebuild a PB2-C-style queue directly from the frozen 128-rollout raw
data with **time-local prefix validity**.

For candidate time `t`, rollout `i` is eligible iff:

```text
history frames t-2:t are finite on all selector fields

AND

(no stretch / rope-NaN failure)
OR
(first failure step > t)
```

A rollout that fails after `t` remains eligible at `t`.

The queue must ignore:

```text
official_rollout_valid
official_final_reward_nan
final reward
survival after t
future winding after t
future distance/divergence
```

for admission and ranking.

`official_rollout_valid` may be stored only as a diagnostic field.

## Prefix-only pair semantics

At each `t`, among prefix-valid rollouts only, reuse the original PB2-C
same-time pair semantics:

```text
live/frozen history length = 3
winding index differs at t
nonempty post-occluded visible rope in all three frames
robot-history thresholds unchanged
visible-history Chamfer threshold unchanged
```

Then sort with the original PB2-C ranking key.

This produces a new:

```text
prefix_candidate_count
```

which is allowed to be larger than historical `5534`.

Do not require the prospective queue count to equal 5534.

## Regression check without reintroducing bias

For code validation only:

```text
take the prefix-only candidate list
filter it to pairs where both rollouts are historical full-rollout survivors
```

That diagnostic survivor subset must reproduce:

```text
historical candidate_count = 5534
historical committed top-50
```

This check validates metric/ranking compatibility.

It must NOT be used as the prospective queue.

## Fixed exclusions and live screening remain unchanged

After prefix-only ranking:

```text
exclude historical formal PB3 20 rollout IDs
exclude PB3-R2 calibration 20 rollout IDs
preserve prefix-only source rank
```

Then fresh live screening still admits candidates using only current `t-2:t`
history and validity through `t`.

The historical R3 reconstruction threshold remains non-gating.

No future suffix or Gate 4 is run in PB3-B1.
