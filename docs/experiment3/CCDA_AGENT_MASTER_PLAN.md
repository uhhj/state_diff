# CCDA Agent Master Plan — Published-Benchmark Route

> **Canonical repository path**: `docs/experiment3/CCDA_AGENT_MASTER_PLAN.md`  
> **Purpose**: persistent research guardrail for Agent/Codex. This file supersedes the OHJ-specific active guardrail.  
> **Current primary benchmark**: **DLO-Lab / Wiring-post**.  
> **Current main-repo migration start**: `a39b5fe5d612df22ddbbf10f6fbee73b23a8ec19`.  
> **Official DLO-Lab revision pinned for PB0**: `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`.

# 1. Scientific target

CCDA = **Contact-Conditioned Dynamics Ambiguity**.

The project studies deformable-object manipulation states where deployable observations are nearly indistinguishable but hidden interaction physics differs:

\[
H_t^{obs,1}\approx H_t^{obs,2},
\qquad
z_t^1\neq z_t^2
\]

and therefore:

\[
p(S_{t+1:t+H}\mid H_t^{obs},z_t^1)
\neq
p(S_{t+1:t+H}\mid H_t^{obs},z_t^2).
\]

The ambiguity matters only if it propagates to control:

\[
a_t^*(z_t^1)\neq a_t^*(z_t^2)
\]

or creates a material difference in control regret / closed-loop task success.

The final claim is **not** “force/torque classifies a hidden label.”

The intended claim is:

> **Deployable interaction sensing helps bind an ambiguous observed state to a physically compatible future-state branch, improving future prediction, IDM executability and ultimately closed-loop manipulation.**

# 2. What the previous benchmark work established

## 2.1 OHJ

OHJ provided useful mechanism evidence:

- observable ambiguity could be maintained;
- deployable sensing became strongly branch-specific;
- same-action future differences exceeded deterministic repeat variability.

But Phase 0E ended with:

```text
OHJ_CONTROL_STRUCTURE_INSUFFICIENT
```

The hidden condition did not induce the required useful best-action switch.

Therefore OHJ is historical evidence, not the paper's primary benchmark.

## 2.2 DHR

DHR was tested with a complete 2×3 action matrix and ended with:

```text
PHASE0F0_DHR_ACTION_SWITCH_FAIL
```

The intended hidden pocket had zero contact in all six rollouts; FREE and JAM-R did not switch best action.

Therefore DHR is also rejected as the paper's primary benchmark.

## 2.3 Consequence

Do **not** continue OHJ/DHR geometry tuning.

The benchmark policy is now:

> **Use a published deformable-manipulation task whose mechanics have already been demonstrated by its authors. Discover naturally occurring CCDA states inside that task rather than building bespoke hidden traps.**

# 3. Primary benchmark — DLO-Lab Wiring-post

Use official DLO-Lab `wiring_post` unchanged.

At the pinned upstream revision the task contains:

```text
30-vertex rope
Franka robot
two fixed cylindrical posts
DLO/rigid coupling
published target and reward
official CMA-ES trajectory optimization
best_traj.npy / best_qpos.npy replay path
```

This task already contains physical routing around fixed posts and is therefore a stronger substrate for CCDA than adding new posts/hooks/pockets ourselves.

## Benchmark-integrity rule

For primary CCDA experiments do not modify Wiring-post:

```text
rigid geometry
post positions/radii
rope physics for the purpose of creating ambiguity
reward
target
contact solver
task semantics
```

Allowed CCDA additions are outside task physics:

```text
partial-observation definition
occlusion/masking used by the model
logging/instrumentation
robot force/proprioceptive sensing extraction
oracle-only hidden routing descriptors
state-pair mining
snapshot/replay branching
StateDiff observation wrapper
sensor temporal encoder
CFPM
IDM and closed-loop evaluation
```

If Wiring-post does not naturally contain suitable pairs after the precommitted collection route, switch to another **published task**. Do not “repair” Wiring-post physics.

# 4. Strategy — mine pairs, do not build pairs

The active research route is:

```text
published Wiring-post
    ↓
reproduce official task
    ↓
collect natural rollouts
    ↓
mine observation-near / hidden-different state pairs
    ↓
same-command future audit
    ↓
control branch audit
    ↓
formal CCDA dataset
    ↓
B0 StateDiff
    ↓
B1 sensor-conditioned StateDiff
    ↓
CFPM
    ↓
IDM
    ↓
closed-loop
```

No new custom hidden-state mechanism should be introduced unless the published-benchmark route itself is formally abandoned.

# 5. PB0 — published-benchmark migration

PB0 has three jobs only.

## PB0-A — reproduce official benchmark

Integrate official DLO-Lab and reproduce `wiring_post` CMA-ES.

Sufficient reproduction evidence:

```text
environment builds
official task runs
best_traj.npy exists
best_qpos.npy exists
best reward is finite
saved qpos replay completes
```

Do not block progress while trying to exactly reproduce a paper table percentage unless a discrepancy indicates a real implementation problem.

## PB0-B — collect natural rollouts

Replay the official optimized qpos sequence over batches of the repository's own Wiring-post initialization randomization.

Do not modify reward or physics.

Save compact arrays:

```text
rope_xyz
rope_vel
EE pose
robot joint position
actual robot DOF force
controller DOF force
reward
post poses
common replay qpos
```

Robot force signals are **candidate deployable sensing**, not a Gate-3 result yet.

## PB0-C — pair mining

At the same replay time, search different rollouts for:

1. close visible rope history after masking local post neighborhoods;
2. close EE history;
3. different oracle-only hidden routing/contact proxy;
4. different future visible evolution under the identical future qpos suffix.

The oracle descriptor may use:

```text
post-wise wrap angle
minimum rope-post surface clearance
nearest rope vertex
contact-like clearance proxy
```

Oracle descriptors are for mining/audit only and must never be model input.

# 6. Observation protocol

PB0 must not feed the complete simulator state to StateDiff.

For pair mining:

- hide rope vertices inside a fixed local disk around each post;
- compare remaining rope point sets using symmetric Chamfer distance;
- include EE position distance;
- compare a short history instead of a single frame.

The mask is an **observation definition**, not a physics modification.

Do not tune it after seeing the result merely to obtain candidates. A single principled correction is allowed only if the first mask is clearly degenerate (for example, almost all rope becomes invisible).

# 7. Hidden descriptor semantics

PB0 does not depend on a privileged DLO-rigid contact-force API.

Compute an oracle routing/contact proxy from full rope geometry and known published post geometry.

Use exact language:

```text
oracle routing descriptor
minimum surface-clearance proxy
contact-like clearance proxy
wrap angle
```

Do not call geometric clearance a measured contact force.

If the pinned simulator later exposes a clean DLO-rigid contact signal, it can validate the oracle descriptor. It still remains privileged audit information.

# 8. Sensor strategy

Long-term sensor variable:

\[
Z_t.
\]

Initial DLO-Lab candidate deployable signals:

```text
Franka actual joint/DOF force
Franka controller force
their temporal residual/difference
optional gripper/finger contact sensing if supported cleanly
```

Do not train a FREE/JAM classifier.

Preferred later representation:

```text
causal sensor window
→ lightweight Conv1D / temporal encoder
→ sensor embedding
```

B1 and CFPM must see the same causal sensor history for a fair comparison.

# 9. Generalized five scientific gates

The old probe-specific OHJ language is generalized for a natural published task.

## Gate 1 — Observable-history equivalence

\[
H_t^{obs,1}\approx H_t^{obs,2}.
\]

Visible DLO + robot observation history must be close.

## Gate 2 — Action/history equivalence

Recent action histories must be equal or sufficiently close.

For same-timestep pairs under a common `best_qpos` replay this is satisfied by construction; verify it once rather than repeatedly auditing it.

## Gate 3 — Sensor observability

Deployable sensor history must contain branch-relevant information beyond repeat/noise variability.

## Gate 4 — Same-action future divergence

With identical future command suffix:

\[
A_{t:t+H}^1=A_{t:t+H}^2
\]

future observable deformable states must diverge beyond a repeat/uncertainty floor.

Visible deformation is evidence, not the research objective.

## Gate 5 — Control relevance

Require:

\[
a_1^*\neq a_2^*
\]

or a robust equivalent cross-condition control-regret / success difference under a shared action set and objective.

A weak comparison against only one alternative action is not sufficient.

# 10. StateDiff route

Do not freeze the final Wiring-post StateDiff dimensionality before PB0 proves natural pairs exist.

After PB1 admission define:

\[
S_t=[D_t,R_t]
\]

where:

- \(D_t\): deployably observable DLO keypoints/features;
- \(R_t\): minimal robot state needed for future prediction/control.

Never include as B0/B1 deployable input:

```text
oracle wrap label
contact-like proxy
masked hidden vertices
future state
hidden condition ID
```

# 11. Baselines and method

After published-benchmark feasibility is established:

## B0 — StateDiff

```text
observable state history
→ future-state diffusion
```

No interaction sensor.

## B1 — StateDiff + temporal interaction sensing

```text
observable state history
+ causal Z history
→ future-state diffusion
```

This is the first critical scientific comparison.

## B2 — StateDiff + CFPM

CFPM learns future compatibility:

\[
G_\phi(H_t^S,Z_t,Y_k,k)
\]

and guides reverse future-state diffusion.

## B3 — B1 + CFPM

Tests complementarity of direct sensor conditioning and inference-time guidance.

## Oracle

Privileged hidden state/future is an upper bound only.

# 12. Required sensor ablations later

Once B1 exists, include:

```text
correct Z
zero Z
wrong-episode Z
time-shuffled Z
```

A convincing result should improve with physically correct sensing and degrade with wrong/shuffled sensing.

# 13. Evaluation priorities

## Primary

```text
closed-loop task success
control regret
```

## Secondary

```text
correct-branch / wrong-branch rate
future-state physical compatibility
future prediction error
IDM executability
```

## Diagnostic only

```text
raw deformation magnitude
curvature
strain
raw force magnitude
wrap angle by itself
```

Do not promote an easy diagnostic to the headline contribution.

# 14. PB0 decision rule

Initial collection:

```text
32 environments × 4 batches
```

If pair mining finds at least 5 discovery candidates under the frozen PB0 filter, proceed to PB1.

If fewer than 5 are found, one larger collection is allowed:

```text
32 environments × 16 batches
```

using the same published environment, replay sequence, observation protocol and filters.

If that still fails:

> stop Wiring-post as the primary CCDA benchmark and evaluate another published task.

Do not move posts, alter friction, change rope stiffness, or tune the target.

# 15. PB1 and model route

If PB0 finds pairs:

## PB1

Snapshot/restore the mined state and branch controls.

Audit:

```text
same official replay suffix
alternative shared local actions
Gate 4
Gate 5
```

Only after control relevance is established should a larger CCDA dataset be generated.

## Dataset

Store:

```text
H_obs
causal Z history
future observable state
action
oracle descriptor only as audit metadata
```

## Models

Proceed:

```text
B0
→ B1
→ CFPM
→ B2/B3
→ IDM
→ closed-loop
```

# 16. Engineering rules — mandatory

## 16.1 Progress over security theater

Do not block research because a local development script contains a plaintext local password, temporary token, internal path/IP or simple development credential.

Security review is not the research task.

Only intervene for a concrete practical problem such as destructive behavior or an active long-lived external credential about to be publicly committed.

Do not build secret scanners or credential-management frameworks unless they become genuinely necessary.

## 16.2 No over-auditing

Keep only reproducibility evidence that matters:

```text
RESULT.md
EVIDENCE.json
final config
normal Git commit history
key raw outputs needed to reproduce conclusions
```

Do not build:

```text
per-file provenance databases
nested audit manifests
duplicate evidence trees
an audit stage for every small edit
```

Once an unchanged component has been established, do not repeatedly audit it.

## 16.3 No complex execution contracts

Each phase should have:

```text
one scientific question
few outputs
clear continue/stop decision
```

Do not create dozens of gates, elaborate resume contracts or large state machines.

Use ordinary scripts and ordinary Git.

## 16.4 No over-defensive programming

Handle realistic failures:

```text
required asset missing
array shape mismatch
NaN rollout
expected optimizer output missing
```

Do not write fallback branches for every hypothetical future upstream format.

## 16.5 Do not defend against effectively impossible cases

At the pinned published revision, use documented/stable shapes and APIs directly.

If upstream changes later, adapt later.

Do not pre-write compatibility layers for imagined future versions.

## 16.6 No excessive hashing

Normal Git commit pins are enough:

```text
main commit
DLO-Lab submodule commit
```

Do not SHA256 every config, array, directory or dataset unless a concrete integrity issue appears.

## 16.7 Scientific rigor with forward progress

Never lower a criterion after seeing a result merely to pass.

Never modify published benchmark physics to manufacture CCDA.

But also do not spend days polishing a diagnostic that cannot change the next research decision.

Prefer the shortest end-to-end experiment that tests the next scientific hypothesis.

## 16.8 End-to-end executability is the engineering priority

Once a conclusion is credible, move forward.

The project must converge toward:

\[
H_t^S,Z_t
\rightarrow
StateDiff/CFPM
\rightarrow
S_{future}
\rightarrow
IDM
\rightarrow
closed\text{-}loop.
\]

Do not optimize isolated subsystems indefinitely.

## 16.9 Regular cleanup is mandatory

At the end of every phase delete:

```text
temporary clones
cache directories
failed scratch outputs
obsolete duplicate configs
superseded generated instructions
one-off debugging files
unused plots/videos
```

Preserve:

```text
final RESULT.md
final EVIDENCE.json
final config
code required to reproduce the conclusion
small canonical artifacts
```

Historical OHJ/DHR final reports stay. Redundant active planning documents should be archived or removed once the DLO-Lab route is stable.

# 17. Canonical active repository layout

```text
docs/experiment3/CCDA_AGENT_MASTER_PLAN.md

external/dlo-lab/

configs/experiment3/published_benchmark/

scripts/experiment3/dlolab_wiring_post/

tests/experiment3/dlolab_wiring_post/

reports/experiment3/pb0_dlolab_wiring_post/
```

Keep `external/deformable-ravens` temporarily for historical reproducibility. It is no longer the active primary simulator.

# 18. Agent decision checklist

Before substantial work, ask:

1. Does this advance a published-benchmark CCDA pair, model or closed-loop result?
2. Am I changing benchmark physics when I should only change observation/modeling?
3. Am I adding an audit/contract rather than answering a scientific question?
4. Can the question be tested by a smaller end-to-end smoke first?
5. Is this file/output still necessary?

If the work does not push the causal chain forward, lower its priority.

# 19. Immediate task

Execute:

> **PB0 — DLO-Lab Wiring-post reproduction and natural CCDA pair mining.**

Do not resume OHJ/DHR geometry work.

Do not start B0/B1 until PB0/PB1 establish a credible natural hidden-state/control structure.
