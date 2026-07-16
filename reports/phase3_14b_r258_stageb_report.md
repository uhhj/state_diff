# Phase3.14b-r2.5.8 Stage B

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r258_stageb_upper_only_direct_x0_oracle_uses_nonphysical_or_fidelity_shortcut`
- Required next path: `REDESIGN_BALANCED_LOWER_UPPER_SEGMENT_GEOMETRY_OBJECTIVE_BEFORE_MODEL_REPAIR`

## Immutable Stage-A Resume1 binding

- Base evidence commit: `4d5773d070b99f0eccdb97c1bc9fa7c5cb3bc583`
- Base worker SHA256: `c71590b7557da309cc02b3fc9e6c4387f815eb7800d4307f76ce84d238c13149`
- Base contract SHA256: `73cb8412498aa5c65c01fc123c35ee72eb8da9c6c543d31227118dd8bffc693b`
- Base selection SHA256: `62e2800808620dc48858dec5105f97fca71ac25a5ca987242ae5dc2a9cc71a4f`

## Portable compute environment

- Compatibility pass: `true`
- Compatibility SHA256: `03fef5cfcfac7933bca0dfc72e67156bb260d50ae7f2f1f6a6375a34c90ad7ec`
- Observation SHA256: `26e55eab01957ee3711cabaf1499774f140237e4ec24241d723db6a36f949ab0`
- GPU: `NVIDIA GeForce RTX 4080 SUPER`
- Compute capability: `8.9`
- Driver: `580.119.02`
- Required-operation dry run: `true`
- Cold CUDA context before control replay: `true`

## Frozen Stage-C control

- Reference equivalence pass: `true`
- Calibration reference/observed SHA256: `fb91d29c84cedccf8f72458f08c2f213bbbc542663593094c0ab651701a052f0` / `fb91d29c84cedccf8f72458f08c2f213bbbc542663593094c0ab651701a052f0`
- Control reference/observed SHA256: `dee61b8e31455eeb0e7e0eceae5c1500e3ce57ca8bd42989378d9f76fd779ae7` / `dee61b8e31455eeb0e7e0eceae5c1500e3ce57ca8bd42989378d9f76fd779ae7`
- Stage-C nonzero candidates trained: `0`
- Retained-control wrapper restored: `true`

## Source audit

- Direct-x0 residual parameterization: `true`
- Upper objective one-sided: `true`
- Explicit lower constraint present: `false`
- Identified risk: `one-sided upper-only geometry descent can shorten segments without an explicit lower/collapse constraint`

## Target-line reachability

| t | Control upper | Target upper | Line reachable | First valid alpha | Target physical | Monotonic violation |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0.991525 | true | 1.0000 | 0.95339 | 0 |
| 25 | 0 | 0.991525 | true | 1.0000 | 0.95339 | 0 |
| 50 | 0 | 0.991525 | true | 1.0000 | 0.95339 | 0 |

## Direct-x0 upper-only oracle

| t | Geometry-descent target cosine | Upper+fidelity reachable | Cable-valid reachable | Best upper+fidelity | Best cable-valid |
|---:|---:|---:|---:|---|---|
| 10 | 0.15736 | true | false | r=0.050; upper=0.9788; nmse-ratio=1.0028; lower=0.8517; physical=0.1356; collapse=0.0763 | none |
| 25 | 0.122413 | true | false | r=0.050; upper=0.9576; nmse-ratio=0.9785; lower=0.8305; physical=0.1144; collapse=0.0803 | none |
| 50 | 0.108205 | true | false | r=0.025; upper=0.5720; nmse-ratio=0.9755; lower=0.8856; physical=0.2966; collapse=0.0502 | none |

## Direct-x0 model tangent

| t | Rows | 8-step Krylov explained lower bound | Global response stability | Output-head explained | Output-head relative update |
|---:|---:|---:|---:|---:|---:|
| 10 | 32 | 0.891223 | 0.999989 | 0.086054 | 0.135803 |
| 25 | 32 | 0.933721 | 1 | 0.111616 | 0.105523 |
| 50 | 32 | 0.968308 | 1 | 0.130911 | 0.0935915 |

## Classification

- Primary failure locus: `one_sided_objective_shortcut`
- Target anchor all pass: `true`
- Target line all reachable: `true`
- Oracle upper+fidelity count: `3`
- Oracle cable-valid count: `0`
- One-sided nonphysical shortcut count: `3`
- Geometry-descent mean target cosine: `0.1293260189750445`
- Contract SHA256: `2322d276e57deae59135e0170920d6c9b2df3aeab040dbe58daf73db2adde684`
- Selection SHA256: `ed716f7fca55c2b222f6363908f50f89a981a821751f2bddbb7f24ea9df6a443`

## Boundary

- No new model candidate was trained.
- Direct-x0 oracle optimization modified only temporary output tensors.
- Central finite differences restored the model byte-exact after every parameter perturbation.
- Ground truth was used only for train-only diagnostic reachability, never for candidate selection.
- The frozen probe was not accessed.
- No full repaired model, reverse sampling, formal training, IDM, candidate execution, DeformableRavens, Phase4, or CPS was run.
- No checkpoint, weights, prediction tensor, oracle tensor, candidate tensor, NPZ, cache, image, or video was persisted.
