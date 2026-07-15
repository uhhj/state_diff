# Phase3.14b-r2.5.7 Stage B Upper-Objective Mechanism Audit

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r257_stageb_upper_objective_huber_linear_saturation`
- Required next path: `CALIBRATE_TOPK_QUADRATIC_OR_SOFTPLUS_UPPER_OBJECTIVE_ON_TRAIN_ONLY_SPLIT`

## Immutable Stage-A binding

- Stage-A worker identity SHA256: `ab6a626a284a1da6c1cf88b75f26388f9c31413cd4a112f11def3c610f7fde3c`
- Stage-A objective contract SHA256: `58f9e3e21e3580a317ae2f1ada1e7d99be672e2da1259bc1a6d4800909bf199f`
- Stage-A selection SHA256: `ded47097a184fd6f2ed5fb36bb4e1085a255f75f1fa1d9524b193f1c2aba0169`

## Mechanism contract

- Contract SHA256: `829082531e2b52298bf2bd2a5b5ccb32509c5d4b8bd430c837aa610a95a4ec22`
- Training checkpoints: `1, 100, 500, 2000, 8000`
- Fixed diagnostic timesteps: `10, 25, 50, 75`
- Frozen probe accessed: `false`
- Reverse sampling run: `false`

## Candidate response

| Lambda | Exact replay | t10 mean row-best excess | t10 p95 excess | Positive segments | Top position |
|---:|---:|---:|---:|---:|---:|
| 0 | true | 0.750798598 | 1.06903095 | 44.9618644 | 0.063559322 |
| 0.001 | true | 0.947523154 | 1.24988322 | 46.190678 | 0.118644068 |
| 0.01 | true | 1.05550623 | 1.38342945 | 48.8728814 | 0.0847457627 |
| 0.1 | true | 1.09981685 | 1.50875064 | 46.6525424 | 0.334745763 |

## Gradient mechanism

- Best lambda by continuous t10 response: `0.001`
- Best t10 mean-excess reduction: `-0.262020409`
- Strongest-lambda scaled geometry/diffusion gradient ratio: `4.2364382`
- Strongest-lambda gradient cosine: `0.35547269`
- Strongest-lambda active-element conflict fraction: `0.420751112`
- Strongest-lambda zero geometry-gradient fraction: `0.00202356387`
- Strongest-lambda Huber linear fraction: `0.968480814`
- Strongest-lambda row-max top-position fraction: `0.125`

## Classification

- Primary failure locus: `huber_saturation`
- Mechanism recommendation: `CALIBRATE_TOPK_QUADRATIC_OR_SOFTPLUS_UPPER_OBJECTIVE_ON_TRAIN_ONLY_SPLIT`

## Boundary

- The four Stage-A pilots were replayed exactly; no new lambda or objective variant was run.
- Gradient diagnostics were read-only and the final model/optimizer/loss/gradient/exposure identities remained exact.
- Selection holdout only was used for prediction-response attribution.
- The frozen 126-row probe was not accessed.
- Reverse sampling, formal pilot, IDM, data collection, candidate execution, DeformableRavens, Phase4 and CPS were not run.
- No checkpoint, weights, prediction tensor, candidate tensor, NPZ, cache, image or video was persisted.
- `selected_configuration=None` and `train_only_recommendation=None`.
