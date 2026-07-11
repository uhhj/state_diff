# GOALS.md — CCDA State-Diffusion Research Roadmap

> Repository: `uhhj/state_diff`, branch `Experiment1`
> Simulation submodule: `uhhj/deformable-ravens`, branch `ccda-cable`
> Current research checkpoint: Phase3.12d-r2.3 completed
> Current formal decision: **do not run Phase4/CPS yet**

---

## 1. Research objective

This project studies **contact-conditioned deformation branch ambiguity (CCDA)** in deformable-object manipulation.

The target phenomenon is:

1. the robot has the same or nearly indistinguishable observable history;
2. past executable actions are the same or nearly indistinguishable;
3. the deformable object is subject to a different latent contact condition;
4. the latent condition produces different future deformation branches;
5. those branches produce different task outcomes or success probabilities.

The proposed method is:

- predict future deformable states using a state-space diffusion model;
- guide denoising with a **Contact Physical Score (CPS)** so sampled futures remain consistent with contact, tactile and proprioceptive evidence;
- use an action-feasibility component with inverse dynamics so physically plausible states do not produce infeasible robot actions.

The method is scientifically meaningful only after the benchmark satisfies the CCDA premise and the baseline pipeline is verified.

---

## 2. Formal observation contract

The formal deployable observation must contain only signals available to the intended system.

### Allowed

- deformable-object position or visual state history;
- robot proprioception;
- past executable robot actions;
- later, explicitly declared tactile/contact measurements.

### Not allowed as ordinary visual input

- hidden-condition labels;
- constraint IDs;
- breakaway state or release metadata;
- future reward, success or future state;
- simulator-only bead velocity unless the experiment is explicitly marked privileged;
- source-condition metadata copied into features.

### Proposed state schema

The current production schema, `state-v1`, contains simulator bead velocity and is not suitable for the final benchmark.

The proposed schema is:

```text
ccda_state_v2_position_proprio

24 beads × XY                 = 48
robot proprio/proxy           = 39
state dimension               = 87

history length                = 3
paper-state input             = 3 × 87 = 261
past executable action input  = 3 × 14 = 42
state-action input            = 303
```

Motion information must be represented by real causal position history, not direct simulator velocity.

---

## 3. Evidence established so far

### Phase 0 — repository and benchmark setup

Completed:

- reproduced the State Diffusion repository environment;
- connected the DeformableRavens submodule;
- verified the cable task and dataset fields;
- established separate generation and training environments.

### Phase 1 — hidden-contact task fork

Completed:

- created `hidden-contact-cable-line`;
- added paired hidden conditions;
- logged ordered bead state and robot proxy;
- added visualization and condition metadata.

### Phase 2 — CCDA data audit

Completed:

- found paired examples with similar initial state/action history;
- found different future cable configurations and success outcomes;
- established that the benchmark can produce branch divergence.

Limitation:

- early environments and datasets used legacy hidden-contact semantics.

### Phase 3 — baseline integrity and execution audits

Completed:

- aligned paper-state and state-action inputs;
- repaired the executable 14-D action codec;
- removed camera configuration from the action target;
- audited inverse dynamics;
- added grouped evaluation and matched resets;
- identified sparse progress quantization;
- implemented query-local PyBullet snapshot execution;
- repaired per-physics-step breakaway checks;
- repaired offline/live state extraction parity;
- audited dataset split leakage, duplicate windows and checkpoint dimensions.

### Phase3.12d-r2.1 — paired-horizon environment audit

Established:

- free replicate trajectory is deterministic;
- hidden-unarmed and free trajectories match;
- the rigid world-anchor condition causes a small condition-specific no-action motion.

### Phase3.12d-r2.2 — observation-space leakage audit

Established:

- XY position alone is close to chance;
- simulator bead velocity strongly decodes the hidden condition;
- the leakage reaches the actual `model_x`;
- negative controls pass.

### Phase3.12d-r2.3 — observation-contract audit

Established:

- simulator velocity is a strong privileged leakage channel;
- robot proprioception does not leak the condition;
- true causal finite differences computed from XY history also decode the condition;
- therefore the rigid world-anchor task leaks through legally observable motion;
- merely deleting simulator velocity is insufficient.

---

## 4. Current blockers

### Blocker A — benchmark validity

The current condition:

```text
hidden_breakaway_pin
```

uses a rigid world point constraint. Even when installed with zero initial position error, it changes the cable's natural dynamics and produces condition-specific no-action motion.

It is retired from formal CCDA claims and retained only for historical diagnostics.

### Blocker B — production state schema

The current legacy dataset and checkpoints use simulator bead velocity.

They are allowed only for bridge diagnostics. They cannot support final architecture-level negative or positive claims.

### Blocker C — candidate oracle

Candidate-set headroom has not been validly measured under a benchmark and observation contract that both satisfy CCDA requirements.

### Blocker D — CPS authorization

CPS development is blocked until:

- a valid latent-contact task passes hiddenness and actionability gates;
- the state-v2 dataset is regenerated;
- StateDiff and inverse dynamics are retrained;
- candidate-set oracle headroom is demonstrated.

---

## 5. Roadmap

## Phase3.12d-r2.4 — Versioned slack-breakaway environment

### Goal

Create a new condition:

```text
hidden_slack_breakaway_pin_v2
```

using a unilateral deadband tether without a persistent PyBullet constraint.

### Required properties

- no force below a declared slack distance;
- no hidden body or constraint;
- no-action free/hidden trajectories are indistinguishable;
- sub-deadband actions do not engage the tether;
- supra-deadband actions engage it;
- stronger actions cause deterministic release;
- after release, force is exactly zero;
- snapshot restore reproduces engagement and release;
- all hidden metadata remains outside observations.

### Gate to pass

```text
phase312d_r24_slack_breakaway_v2_environment_supported
```

Failure variants must identify whether the problem is:

- no-action parity;
- observation leakage;
- deadband actionability;
- breakaway release;
- snapshot restore;
- numerical instability.

### Forbidden

- model training;
- candidate matrix;
- Phase4;
- CPS.

---

## Phase 3.13 — State-v2 dataset regeneration

Starts only after Phase3.12d-r2.4 passes.

### Goal

Generate a new versioned dataset from the slack-breakaway environment using the state-v2 observation contract.

### Requirements

- no simulator bead velocity in model inputs;
- no copying free inputs onto hidden targets;
- each condition keeps its real observation history;
- paired visible seeds remain in the same split;
- train/validation/test splits are grouped by visible seed;
- raw source episodes can reconstruct every window;
- manifest records repository commits, environment semantics and schema version;
- action codec remains 14-D pose0/pose1 only.

### Deliverables

```text
data/phase3_state_v2_slack/
checkpoints/phase3_state_v2_slack/
reports/phase3_13_*
```

---

## Phase 3.14 — Retrained baseline and candidate-set oracle

### Goal

Retrain:

- future state diffusion;
- repaired inverse dynamics;
- state-only and state-action baselines.

Then measure candidate-set realized-effect oracle headroom using query-local snapshots.

### Decision

- useful candidates exist, selectors fail → selection/guidance problem;
- no useful candidates exist → future generation or inverse dynamics problem;
- dense progress improves but sparse fraction does not → metric masking problem.

CPS is not authorized unless useful candidate headroom exists.

---

## Phase 4 — Contact Physical Score prototype

Starts only after Phase 3.14 supports candidate headroom.

### Goal

Implement CPS as a state-space diffusion guidance term.

### Initial variants

- contact-consistency score;
- tactile/history consistency;
- local deformation/contact compatibility;
- uncertainty-aware guidance strength.

### Required comparisons

- unguided diffusion;
- condition-label upper bound;
- nearest-state/retrieval baselines;
- classifier guidance;
- CPS guidance;
- oracle best-of-K.

---

## Phase 5 — Action feasibility

### Goal

Prevent a plausible predicted future from generating an infeasible robot action.

Components:

- inverse dynamics confidence;
- action OOD score;
- geometry feasibility;
- execution discriminator;
- optional feasibility-guided resampling.

---

## Phase 6 — Full evaluation

Required evaluation axes:

- hidden contact types;
- object stiffness and friction;
- action magnitudes;
- unseen visible seeds;
- observation noise;
- partial occlusion;
- history length;
- CPS weight;
- number of diffusion samples;
- inverse dynamics uncertainty;
- task success and dense physical progress.

---

## 6. Hard gates

| Gate | Required before |
|---|---|
| No-action hiddenness | Any formal CCDA dataset |
| Causal action divergence | Dataset generation |
| Breakaway recoverability | Dataset generation |
| State-v2 observation contract | Formal model training |
| Grouped split integrity | Any reported model metric |
| Action codec integrity | Inverse dynamics training |
| Query-local snapshot integrity | Candidate oracle |
| Candidate oracle headroom | CPS implementation |
| CPS improvement on held-out seeds | Final method claim |

---

## 7. Reproducibility rules

Every formal experiment must record:

- main repository commit;
- submodule commit;
- Python, PyTorch, NumPy, SciPy and PyBullet versions;
- condition and observation schema versions;
- seed list and seed-list SHA256;
- data/checkpoint SHA256;
- feature provenance;
- split groups;
- exact thresholds;
- report root cause;
- explicit no-Phase4/no-CPS confirmation when blocked.

JSON reports must reject NaN and Infinity.

No result may silently reuse an artifact produced under a different environment or observation semantics.

---

## 8. Artifact status

### Legacy-only

- rigid `hidden_breakaway_pin`;
- legacy Phase3 windows;
- legacy state-v1 checkpoints;
- old candidate-oracle outputs generated before snapshot and environment repairs.

### Current valid diagnostics

- r2.1 paired-horizon audit;
- r2.2 code/data and observation leakage audit;
- r2.3 observation-contract and privileged-state ablation.

### Next formal artifact

```text
Phase3.12d-r2.4 slack-breakaway-v2 environment audit
```

---

## 9. Current decision

```text
Environment redesign: required
State-v2 activation: not yet
Dataset regeneration: blocked on r2.4
Candidate matrix: blocked
Phase4/CPS: blocked
```

The immediate next task is to implement and audit the versioned unilateral slack-breakaway condition without altering the historical rigid condition.
