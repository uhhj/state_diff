# Phase3.14b-r2.5.7 Stage A Upper-Expansion Objective Calibration

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r257_stagea_no_train_only_upper_objective_configuration`
- Required next path: `CALIBRATE_STRONGER_OR_ALTERNATIVE_UPPER_OBJECTIVE_ON_TRAIN_ONLY_SPLIT`

## Objective contract

- Contract SHA256: `58f9e3e21e3580a317ae2f1ada1e7d99be672e2da1259bc1a6d4800909bf199f`
- Frozen upper-gate SHA256: `24f5ce7aa5c6d3c45ab6d4e659fa1c4719ffb486b61022db453b289e990b2781`
- Frozen upper threshold: `3.937946394`
- Candidate lambdas: `0, 0.001, 0.01, 0.1`
- Lower XY score used for validity: `false`

## Leakage-controlled selection

- Objective-train rows / groups: `638 / 126`
- Selection-holdout rows / groups: `236 / 42`
- Selection SHA256: `ded47097a184fd6f2ed5fb36bb4e1085a255f75f1fa1d9524b193f1c2aba0169`
- Frozen probe accessed during selection: `false`
- Selected lambda: `none`

## Pilot candidates

| Lambda | Feasible | Train-control NMSE | t10 upper | t25 upper | t50 upper |
|---:|---:|---:|---:|---:|---:|
| 0 | false | 0.00317357946 | 0 | 0 | 0 |
| 0.001 | false | 0.0035906008 | 0 | 0 | 0 |
| 0.01 | false | 0.00459918706 | 0 | 0 | 0 |
| 0.1 | false | 0.0129469326 | 0 | 0 | 0 |

## Final frozen-probe audit

- Performed: `false`

## Classification

- Primary failure locus: `train_only_objective_selection`

## Boundary

- r2.5.6 Stage D.3 Resume1 evidence was immutable.
- The model continues to predict direct normalized x0.
- Hyperparameter selection used only a grouped holdout inside the frozen Stage-B training partition.
- Frozen probe targets were not accessed unless a nonzero configuration passed train-only selection.
- The upper gate, scheduler, K, formal split and DeformableRavens were unchanged.
- No checkpoint, tensor, NPZ, cache, image or video was persisted.
- No formal pilot, formal diffusion/reverse, IDM, data collection, candidate execution, Phase4 or CPS was run.
