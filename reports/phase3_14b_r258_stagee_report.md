# Phase3.14b-r2.5.8 Stage E

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r258_stagee_oracle_valid_but_surrogate_direction_not_integrable`
- Required next path: `CALIBRATE_CONSTRAINT_AWARE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY`

## Immutable Stage-D Resume2 binding

- Base evidence commit: `61be1c377eeda6da4c25e21399e160beb9a33249`
- Base worker SHA256: `0d9cd32cb8724d9e90c7c29a826e2e12e66937efccb7123320b5fbf037430c4f`
- Base contract SHA256: `3bfb7f19a1f2d37ee8e7f2f202c2a304396ee1b611932ca2012d841b4540f6e4`
- Base selection SHA256: `3f78083f8038b283648b03717a95ee00b5851abf69ab3ee256959b180b0f8dba`

## Portable environment and frozen control

- Compatibility pass: `true`
- Compatibility SHA256: `03fef5cfcfac7933bca0dfc72e67156bb260d50ae7f2f1f6a6375a34c90ad7ec`
- Observation SHA256: `d66c7a0cf73112dfdbe2c2d82f8e5bd5deb8278b38d650484c141c18edafd005`
- GPU: `NVIDIA GeForce RTX 3090`
- Compute capability: `8.6`
- Required-operation dry run: `true`
- Cold CUDA context before control replay: `true`

## Split and leakage boundary

- Objective-train rows / groups: `638` / `126`
- Selection-holdout rows: `236`
- Frozen-probe rows: `126`
- Candidate generation uses target: `false`
- Holdout evaluated: `false`
- Frozen probe accessed: `false`

## Candidate matrix

| Candidate | Role | Direction source | t10 | t25 | t50 | Eligible |
|---|---|---|---|---|---|---:|
| linear_centered_a10_m2 | negative_control | centered_ridge_a10 | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0000; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | false |
| centered_a10_hist_m1_r25 | diagnostic_control | centered_ridge_a10 | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0000; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | false |
| centered_a10_z4_m1_r25 | selectable | centered_ridge_a10 | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0000; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | false |
| centered_a10_z3_m1_r50 | selectable | centered_ridge_a10 | feasible=0.5690; fallback=0.4310; retention=0.3672; upper=0.5690; physical=0.5690; nmse=1.2724; movement=0.2472; reduction=-0.1171; pass=false | feasible=0.8652; fallback=0.1348; retention=0.6449; upper=0.8652; physical=0.8652; nmse=1.0308; movement=0.4276; reduction=-0.0055; pass=false | feasible=0.8574; fallback=0.1426; retention=0.6229; upper=0.8574; physical=0.8574; nmse=1.0626; movement=0.3621; reduction=-0.0306; pass=false | false |
| centered_a10_z4_m2_r25 | selectable | centered_ridge_a10 | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0000; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | false |
| centered_rr32_z4_m1_r25 | selectable | centered_rridge_k32_a10 | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0000; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | false |
| centered_rr32_z3_m1_r50 | selectable | centered_rridge_k32_a10 | feasible=0.5408; fallback=0.4592; retention=0.3469; upper=0.5408; physical=0.5408; nmse=1.2427; movement=0.2346; reduction=-0.1052; pass=false | feasible=0.8354; fallback=0.1646; retention=0.6186; upper=0.8354; physical=0.8354; nmse=1.0385; movement=0.4059; reduction=-0.0060; pass=false | feasible=0.8072; fallback=0.1928; retention=0.5827; upper=0.8072; physical=0.8072; nmse=1.0677; movement=0.3357; reduction=-0.0309; pass=false | false |
| centered_rr32_z4_m2_r25 | selectable | centered_rridge_k32_a10 | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0000; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | false |
| segment_rr32_z4_m1_r25 | selectable | segment_rridge_k32_a10 | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0000; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | false |
| segment_rr32_z3_m1_r50 | selectable | segment_rridge_k32_a10 | feasible=0.6787; fallback=0.3213; retention=0.4499; upper=0.6787; physical=0.6787; nmse=1.3422; movement=0.2922; reduction=-0.1447; pass=false | feasible=0.9216; fallback=0.0784; retention=0.7125; upper=0.9216; physical=0.9216; nmse=1.0367; movement=0.4616; reduction=-0.0096; pass=false | feasible=0.9107; fallback=0.0893; retention=0.6956; upper=0.9107; physical=0.9107; nmse=32.2885; movement=0.3935; reduction=-0.2150; pass=false | false |
| segment_rr32_z4_m2_r25 | selectable | segment_rridge_k32_a10 | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0000; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | false |
| zero_z4_projection_only | diagnostic_control | zero | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0016; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | feasible=0.0000; fallback=1.0000; retention=0.0000; upper=0.0000; physical=0.0000; nmse=1.0000; movement=0.0000; reduction=0.0000; pass=false | false |
| oracle_z4_m2_r25 | oracle_control | oracle | feasible=0.9875; fallback=0.0125; retention=0.9682; upper=0.9875; physical=0.9875; nmse=0.0741; movement=0.9613; reduction=0.8283; pass=true | feasible=0.9875; fallback=0.0125; retention=0.9724; upper=0.9875; physical=0.9875; nmse=0.0544; movement=0.9673; reduction=0.8512; pass=true | feasible=0.9875; fallback=0.0125; retention=0.9756; upper=0.9875; physical=0.9875; nmse=0.0384; movement=0.9716; reduction=0.8699; pass=true | false |

## Selection and controls

- Objective-train selected configuration: `none`
- Validated train-only recommendation: `none`
- Permutation control: `not-run`
- Projection-only control: `not-run`
- Locked holdout: `not-run`
- Oracle all timesteps: `true`
- Eligible candidates: `none`
- Primary failure locus: `surrogate_precision_under_constraints`
- Contract SHA256: `dfa66bc374871521195d6ffca57e047316b0294d37209b57c1f9493b826f8560`
- Selection SHA256: `068ebecd78e5ac0231d5e3b407855982867432625ef6f5d8f23a37bcd263b94f`

## Boundary

- The integrator selected row scales using only control/candidate values and frozen geometry contracts.
- Ground-truth collapse, stretch, target direction, and target-distance reduction were evaluation-only.
- Topology was evaluated only after vectorized observable gates and retried at smaller scales after rejection.
- No diffusion-model candidate was trained and no architecture was changed.
- The frozen probe was not accessed.
- No reverse sampling, formal training, IDM, candidate execution, DeformableRavens, Phase4, or CPS was run.
- No checkpoint, weights, surrogate weights, prediction tensor, candidate tensor, NPZ, cache, image, or video was persisted.
