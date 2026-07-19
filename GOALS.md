# GOALS.md — CCDA State-Diffusion Research Roadmap

> Repository: `uhhj/state_diff`, branch `Experiment1`
> Simulation submodule: `uhhj/deformable-ravens`, branch `ccda-cable`
> Current research checkpoint: Phase3.14b-r2.5.8 Stage H implementation committed at `0203b66`; candidate-descriptor identifiability audit is in progress, with Stage G as the latest finalized evidence
> Current formal decision: **Stage G execution passed, but feasibility progress was not identifiable. No configuration was selected; formal test, formal training, Phase3.14c, candidate execution, and Phase4/CPS remain blocked**

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

### Phase3.12d-r2.4 and Phase3.13 — valid environment and formal state-v2 data

Established:

- replaced the rigid world-anchor benchmark with the versioned slack-breakaway environment;
- passed no-action parity, actionability, deterministic release and snapshot-restore gates;
- regenerated and atomically promoted the grouped formal state-v2 dataset;
- retained a 14-D executable action codec and excluded hidden metadata from model inputs;
- produced 448 visible seeds, 896 episodes and 4256 windows.

### Phase3.14a–r2.5.5 — cache, DDPM and target-schema diagnosis

Established:

- built the immutable cache and passed deterministic future learnability;
- completed the DDPM candidate matrix, scheduler-equivalence and ordered-geometry audits;
- separated exact reconstruction, branch transport and robot-proxy reconstruction gates;
- proved same-device residual determinism and cross-device functional-prior equivalence;
- migrated the raw robot proxy to a write-once state-v3 dataset and cache;
- found that future robot proxy is neither deployably predictable nor materially useful to IDM;
- required future diffusion targets to become cable-only.

### Phase3.14b-r2.5.6–r2.5.7 — cable geometry and objective diagnosis

Established:

- the cable-only contract is valid, but branch support fails;
- the original symmetric segment gate was confounded by lower-tail XY collapse;
- a frozen one-sided upper-segment contract isolates the remaining failure to cable x0 upper expansion;
- Huber upper objectives saturate, while top-k quadratic variants retain a fidelity tradeoff;
- timestep-gated K16 still has no joint geometry/fidelity solution;
- no train-only recommendation or formal configuration was selected.

### Phase3.14b-r2.5.8 — portable direction, reachability and feasibility chain

Established:

- replaced hardware-model identity gates with portable compatibility and same-instance determinism contracts;
- showed a physically valid direct-x0 endpoint is reachable, while balanced local descent is misaligned;
- found grouped-CV direction signal, but no valid state transition from the initial surrogate;
- calibrated constrained direct-x0 integration and a constraint-aware direction surrogate;
- separated structural-zero constraint state and admitted only ULP factor `1.0` under the frozen objective-train policy;
- Stage G preserved direction signal for all nine candidate backbones but produced no feasibility-progress ranker signal;
- Stage G execution verdict is `PASS`, scientific status is `BLOCKED`;
- root cause is `phase314b_r258_stageg_feasibility_progress_not_identifiable`;
- required next path is `AUDIT_CANDIDATE_DESCRIPTOR_IDENTIFIABILITY`;
- `train_only_recommendation` and `selected_configuration` remain `None`.

---

## 4. Current blockers

### Blocker A — feasibility-progress descriptor identifiability

Stage G retained a usable direction signal, but every frozen ranker candidate had zero observable feasible rate, zero utility correlation and no positive Brier improvement. The next audit must determine whether the frozen candidate descriptors contain enough information to rank feasibility progress.

### Blocker B — cable candidate integration

A valid direct-x0 endpoint is reachable, but the learned/surrogate direction has not yet formed a physically valid integrated transition. Geometry, fidelity and feasibility gates have not passed together for any candidate.

### Blocker C — formal candidate oracle

No candidate configuration has been selected. Formal candidate execution and candidate-set headroom remain unmeasured because train-only geometry and feasibility diagnostics are still blocked.

### Blocker D — CPS authorization

CPS development remains blocked until:

- candidate descriptors support identifiable feasibility progress;
- a frozen candidate configuration passes cable geometry, fidelity and feasibility gates;
- formal candidate execution demonstrates useful headroom on held-out grouped data;
- inverse-dynamics inputs and action feasibility are validated under the cable-only future contract.

---

## 5. Roadmap

## Phase3.12d-r2.4 — Versioned slack-breakaway environment

Status: **COMPLETED / PASS**

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

Status: **COMPLETED / PASS**. Current work remains in Phase3.14b train-only candidate-generation diagnostics.

Formal task and observation contract:

```text
task: ccda-slack-cable-v2
conditions: free / hidden_slack_breakaway_pin_v2
schema: ccda_state_v2_position_proprio
state_dim: 87
paper_x_dim: 261
state_action_x_dim: 303
```

Legacy production assets are removed from the branch tip only after migration smoke validation. Git history remains intact.

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

Current status: **SCIENTIFICALLY BLOCKED after Phase3.14b-r2.5.8 Stage G**. Stage H candidate-descriptor identifiability code is committed and its audit is in progress; it is not formal candidate execution and has no finalized scientific result yet.

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
- r2.3 observation-contract and privileged-state ablation;
- r2.4 slack-breakaway environment audit;
- Phase3.13-r1 formal state-v2 dataset and grouped-split audit;
- Phase3.14a immutable cache and deterministic learnability audit;
- r2.5.5 state-v3 robot-proxy migration and target-schema attribution;
- r2.5.6 cable-only geometry and one-sided upper-segment contract audits;
- r2.5.7 upper-objective and timestep-gated K16 diagnostics;
- r2.5.8 portable direct-x0 reachability, constrained integration, ULP admission and Stage G joint direction-feasibility evidence.

### Next formal artifact

```text
Phase3.14b-r2.5.8 Stage H candidate-descriptor identifiability evidence (implementation commit `0203b66`, audit in progress)
```

---

## 9. Current decision

```text
Environment redesign: completed / PASS
Phase3.13 state-v2 activation: completed / PASS
Formal dataset: 448 visible seeds / 896 episodes / 4256 windows
Phase3.13-r1: COMPLETED / PASS
Phase3.14 provenance: PASS / phase314_data_provenance_supported
Phase3.14a: COMPLETED / PASS
Gate: phase314a_state_v2_future_learnability_supported
Phase3.14b: FAIL
Root cause: phase314b_candidate_physical_validity_failed
Phase3.14b-r1.1: COMPLETED / PASS
Root cause: phase314b_r1_cosine_epsilon_terminal_snr_instability_supported
Phase3.14b-r2: FAIL
Root cause: phase314b_r2_no_stable_configuration
Phase3.14b-r2.1: COMPLETED / PASS
Root cause: phase314b_r21_contract_miscalibration_and_ordered_geometry_failure_supported
Phase3.14b-r2.2: FAIL
Root cause: phase314b_r22_no_pilot_geometry_repair

Phase3.14b-r2.3: COMPLETED / PASS
Meaning: validation-only diagnosis completed; no model repair
Root cause: phase314b_r23_tiny_overfit_capacity_or_implementation_failure
Selected configuration: none
Next: debug model/optimizer on train-only tiny set

Phase3.14b-r2.3.1: COMPLETED / PASS
Meaning: Corrected train-only tiny-control contract; no model repair and no candidate selection
Root cause: phase314b_r231_random_noise_single_row_optimization_failure
Selected configuration: None
Next: debug timestep/noise coverage and denoiser conditioning on one train row

Phase3.14b-r2.3.2: COMPLETED / PASS
Meaning: Train-only single-row random-noise denoiser isolation completed; no model repair and no candidate selection
Root cause: phase314b_r232_noisy_input_skip_path_deficiency_supported
Selected configuration: None
Next: design a train-only residual/noisy-skip denoiser pilot; formal validation remains blocked

Phase3.14b-r2.4: COMPLETED / PASS
Meaning: train-only timestep-conditioned noisy-skip pilot completed; no formal model repair and no candidate selection
Root cause: phase314b_r24_conditioned_multirow_generalization_failed
Analytic formula oracle: PASS
One-row gate: analytic_x0_skip_residual_v PASS; all other variants FAIL
Unique-free-16 gate: analytic_x0_skip_residual_v FAIL
Paired-16 gate: not started after the unique-free-16 failure
Selected configuration: None
Next: debug condition capacity and source-row batching on unique-free train rows
Formal test: BLOCKED / unread
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.4.1: COMPLETED / PASS
Meaning: train-only conditioned multirow and source-batching audit completed; no model repair and no candidate selection
Root cause: phase314b_r241_condition_encoder_width_limit_supported
Train-only debug recommendation: condition_width_1024
Selected configuration: None
Next: run an additive train-only wider condition-encoder pilot
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.4.2: COMPLETED / PASS
Meaning: train-only factorized frozen-prior/width pilot completed after three implementation blocks; PASS does not imply formal model repair
Root cause: phase314b_r242_prior_gradient_isolation_supported
Secondary mechanism: None
Train-only recommendation: frozen_p512_r512
Direct-prior stability: width 512 = 1/3, width 1024 = 1/3; neither reached the 2-of-3 stability gate
Unique-free advancing variants: frozen_p512_r512, frozen_p1024_r512, frozen_p512_r1024, frozen_p1024_r1024, decoupled_p1024_r512
Paired low/mid passing variants: frozen_p512_r512, frozen_p1024_r512, decoupled_p1024_r512
Selected configuration: None
Next: run a separate train-only ordered-geometry pilot using the factorized frozen/decoupled prior recipe; formal validation remains blocked
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5: COMPLETED / PASS
Meaning: train-only frozen-prior ordered-geometry diagnostic completed; PASS does not imply formal model repair, validation candidate support, or formal selection
Root cause: phase314b_r25_geometry_gradient_scaling_failed
Train-only recommendation: None
Active-objective gradient ratios: ordered_mean_raw = 30.1933, ordered_cvar_raw = 67.4221, ordered_cvar_contract = 28.8141; all exceeded the frozen upper gate of 5.0
Unique-free advancing variants: v_only_frozen_control only; the control is not selectable
Paired result: v_only_frozen_control one-step gate failed; full reverse remained diagnostic-only
Selected configuration: None
Next: repair train-only geometry scaling before another specified pilot
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.1 Resume1: BLOCKED
Meaning: the complete train-only actual-gradient geometry calibration pilot reran after correcting the paired reverse nearest-index batch contract, but finalization rejected the serialized unique objective matrix/order
Execution block: unique objective matrix/order changed
Stage: finalize
Resume generation: 1
Paired reverse batching correction: PASS; singleton metric rank failure did not recur
Train-only recommendation: None
Selected configuration: None
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.1 Resume2: COMPLETED / PASS
Meaning: the train-only actual-gradient calibration chain completed after correcting the JSON objective-order serialization contract; PASS does not mean ordered geometry or formal diffusion was repaired
Root cause: phase314b_r251_paired_low_mid_geometry_transport_failed
Train-only recommendation: None
Selected configuration: None
Next: diagnose paired low/mid transport under calibrated geometry gradients
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.2: COMPLETED / PASS
Meaning: train-only paired one-step/reverse-trajectory attribution completed; PASS means the diagnostic chain completed only and does not mean ordered geometry or formal diffusion was repaired
Root cause: phase314b_r252_composite_one_step_gate_conflation_supported
Supported mechanisms: composite_gate_conflation, per_timestep_gradient_miscalibration
Train-only recommendation: None
Selected configuration: None
Next: separate exact reconstruction and branch-transport gates without changing training
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.3: BLOCKED
Meaning: the train-only reconstruction/branch-transport gate-separation pilot completed, but finalization rejected the v-only one-step reproduction contract
Execution block: r2.5.2 one-step reproduction failed for v_only_frozen_control
Stage: finalization
Pilot summary preserved: a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86
Final root cause: not finalized
Supported mechanisms: not finalized
Train-only recommendation: None
Selected configuration: None
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.3 Resume2: BLOCKED
Meaning: the repository-root import contract was installed and statically validated, but the finalizer-only wrapper stopped before preflight because the parent shell selected a Python runtime without NumPy
Execution block: repository import contract could not import NumPy through the selected Python runtime
Stage: wrapper repository import contract
Resume generation: 2
Parent-shell PYTHONPATH root-first: PASS
CUDA_VISIBLE_DEVICES: empty
GPU pilot rerun: False
Training rerun: False
Reverse rerun: False
Pilot summary preserved: a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86
Final root cause: not finalized
Supported mechanisms: not finalized
Train-only recommendation: None
Selected configuration: None
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.3 Resume3: PASS
Meaning: the committed train-only gate-separation pilot was finalized without GPU, training, one-step, or reverse rerun after correcting the oracle-adapter, repository-import, and explicit Python-interpreter contracts
Root cause: phase314b_r253_robot_proxy_reconstruction_conflation_supported
Supported mechanisms: exact_reconstruction_vs_branch_identity_separation, intermediate_horizon_fidelity_gap, exact_cable_fidelity_tail_gap, all_source_tail_failure, robot_proxy_reconstruction_conflation
Next: audit a separate robot-proxy trajectory-fidelity objective under the fixed cable branch-transport contract
GPU pilot rerun: False
Training rerun: False
One-step rerun: False
Reverse rerun: False
Pilot summary SHA256: a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86
Python executable: /miniforge3/envs/coord_bimanual/bin/python3.9
Train-only recommendation: None
Selected configuration: None
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.4: BLOCKED
Meaning: the robot-proxy attribution preflight passed, but the pilot stopped before diagnostic-model attribution because the shared paired-prior SHA did not reproduce
Execution block: shared paired-prior SHA did not reproduce
Stage: pilot shared-prior reproduction
Pilot summary: not created
Final root cause: not finalized
Supported mechanisms: not finalized
Cable branch contract: NOT INTERPRETED
Ordered topology contract: NOT INTERPRETED
Reverse sampling rerun: False
Train-only recommendation: None
Selected configuration: None
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.4 Resume1: PASS
Meaning: Train-only shared-prior determinism and functional-equivalence audit completed. PASS means only the diagnostic chain completed.
Root cause: phase314b_r254_resume1_cross_device_bitwise_sha_contract_overstrict
Historical prior SHA: 083278d6b0710ddc3863d822978973f29842bc757edf73ff03a93bb154f5665d
Observed prior SHA values: 8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904 x3
Same-device exact SHA: True
Same-device functional equivalence: True
Historical functional fingerprint: True
Functional prior contract supported: True
Next: phase3_14b_r254_resume2_robot_proxy_attribution_functional_prior_contract
Robot-proxy attribution: NOT RUN / NOT INTERPRETED
Reverse sampling rerun: False
Train-only recommendation: None
Selected configuration: None
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.4 Resume2: BLOCKED
Meaning: Train-only robot-proxy attribution was attempted under the audited cross-device functional-prior contract, but the fresh-prior prediction gate could not be evaluated. No robot-proxy metrics are interpretable.
Execution block: fresh prior snapshot lacks the required in-memory `_prior_prediction_z` field
Stage: fresh functional-prior validation before residual training
Preflight: PASS
Pilot summary: not created
Root cause: phase314b_r254_resume2_execution_failed
Supported mechanisms: not finalized
Next: correct the in-memory prior-prediction exposure contract before retrying Resume2
Historical RTX-4080 exact prior SHA required: False
RTX-4090 fresh state SHA: 8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904
RTX-4090 fresh state SHA exact: True
RTX-4090 fresh prediction SHA: NOT VALIDATED
Historical functional fingerprint: True
Cable branch contract: NOT RUN / NOT INTERPRETED
Ordered topology contract: NOT RUN / NOT INTERPRETED
Robot-proxy attribution: NOT RUN / NOT INTERPRETED
Reverse sampling rerun: False
Train-only recommendation: None
Selected configuration: None
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.4 Resume3: BLOCKED
Meaning: Train-only robot-proxy attribution completed its pilot under the audited functional-prior contract after reconstructing the prior prediction fingerprint from retained prior state in memory, but finalization rejected r2.5.3 contract reproduction. Robot-proxy metrics are not interpretable.
Execution block: r2.5.3 contract reproduction failed for ordered_cvar_contract_g010; pilot evidence shows reproduction false for all three diagnostic models
Stage: finalization after pilot completion
Pilot SHA256: d28c4b40c16ba852a2dc3cc42cfee548defeb6aa7ed3717203690282b984f36e
Root cause: not finalized; pilot diagnosis is phase314b_r254_r253_contract_reproduction_failed
Supported mechanisms: not finalized
Next: repair r2.5.3 contract reproduction
Fresh prior state SHA exact: True
Fresh prior prediction SHA exact: True
Fresh prior z-MSE pass: True
Prediction reconstructed from retained state: True
Stochastic state restored: True
Prediction tensor persisted: False
Historical snapshot API changed: False
Cable branch contract: PASS observed in pilot / NOT FINALIZED
Ordered topology contract: PASS observed in pilot / NOT FINALIZED
Robot-proxy attribution: RUN / NOT INTERPRETABLE
Reverse sampling rerun: False
Train-only recommendation: None
Selected configuration: None
Validation targets: UNUSED / BLOCKED
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED

Phase3.14b-r2.5.4 Resume4/Resume5: COMPLETED / PASS
Meaning: reproduction identity was decomposed, then same-device residual determinism and cable functional non-regression were established
Root cause: phase314b_r254_resume5_same_device_residual_determinism_and_cable_functional_nonregression_supported
Cross-device rule: compare frozen scientific/functional contracts, not hardware-specific byte SHA
Train-only recommendation: None
Selected configuration: None
Next: migrate the robot-proxy schema and regenerate a write-once cache

Phase3.14b-r2.5.5 Stage A/B/C Resume1: COMPLETED / PASS
Meaning: the legacy robot proxy was deterministically migrated to state-v3, the write-once dataset/cache was reproduced, and robot-future attribution completed
Root cause: phase314b_r255_stagec_robot_future_not_deployably_predictable_or_idm_material
Decision: remove robot future from the diffusion target and redefine IDM input around cable future plus causal history/action information
Train-only recommendation: None
Selected configuration: None

Phase3.14b-r2.5.6 Stage A through Stage D.3 Resume1: COMPLETED / PASS
Meaning: the cable-only contract and IDM identifiability audit passed, but cable branch transport failed; gate audits isolated lower-tail XY collapse and froze a one-sided upper-segment contract
Root cause: phase314b_r256_staged3_cable_x0_upper_segment_expansion_failed
Primary failure locus: x0_upper_expansion
Train-only recommendation: None
Selected configuration: None
Next: add and calibrate an ordered upper-segment expansion objective on train-only data

Phase3.14b-r2.5.7 Stage A/B/C/Stage D Resume4: COMPLETED / PASS
Meaning: upper-objective, mechanism, top-k quadratic and timestep-gated K16 diagnostics completed under the corrected GPU replay contract
Root cause: phase314b_r257_staged_timestep_gated_k16_no_geometry_or_fidelity_solution
Primary failure locus: joint_tradeoff
Train-only recommendation: None
Selected configuration: None
Next: calibrate a portable conflict-projected low-noise K16 objective

Phase3.14b-r2.5.8 Stage A Resume1 through Stage F V8: COMPLETED / PASS
Meaning: portable compatibility, direct-x0 reachability, balanced geometry, direction-surrogate, constrained-integrator and constraint-aware ULP-policy diagnostics completed
Root cause: phase314b_r258_stagef_constraint_aware_direction_still_not_integrable
Selected ULP factor: 1.0
Eligible ULP factors: [1.0]
Train-only recommendation: None
Selected configuration: None
Next: calibrate a joint direction and feasibility surrogate

Phase3.14b-r2.5.8 Stage G: EXECUTION PASS / SCIENTIFIC BLOCKED
Meaning: all nine frozen candidate backbones retained direction signal, but none produced identifiable feasibility progress or passed the integrated state/observable gates
Root cause: phase314b_r258_stageg_feasibility_progress_not_identifiable
Primary failure locus: feasibility_progress_ranker
Eligible candidates: none
Train-only recommendation: None
Selected configuration: None
Required next path: AUDIT_CANDIDATE_DESCRIPTOR_IDENTIFIABILITY
Formal test: UNREAD / BLOCKED
Formal training: BLOCKED
Phase3.14c IDM: BLOCKED
Candidate execution: BLOCKED
Phase4/CPS: BLOCKED
```

Phase3.13-r1 regenerated, audited and atomically promoted the formal state-v2 dataset. Phase3.14a established the immutable cache and deterministic future learnability. Phase3.14b through r2.5.4 then localized failures from scheduler instability and ordered cable geometry to branch transport and robot-proxy reconstruction conflation. Phase3.14b-r2.5.5 migrated state-v3 and established that robot future should not be a diffusion target. Phase3.14b-r2.5.6 froze a cable-only, one-sided upper-segment contract; r2.5.7 showed that existing upper objectives and timestep-gated K16 could not jointly satisfy geometry and fidelity. Phase3.14b-r2.5.8 established portable hardware compatibility, direct-x0 reachability, constrained integration and an objective-train ULP admission factor of `1.0`. Stage G preserved direction signal but found feasibility progress unidentifiable for all nine ranker candidates. Stage H candidate-descriptor identifiability is implemented at `0203b66` and currently running; no Stage H evidence has been finalized. Formal test remains unread; no configuration is selected; formal training, Phase3.14c IDM, candidate execution and Phase4/CPS remain blocked.
