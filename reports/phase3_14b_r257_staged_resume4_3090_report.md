# Phase3.14b-r2.5.7 Stage D Resume4 RTX 3090

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r257_staged_timestep_gated_k16_no_geometry_or_fidelity_solution`
- Required next path: `CALIBRATE_CONFLICT_PROJECTED_LOW_NOISE_K16_OBJECTIVE_ON_TRAIN_ONLY_SPLIT`

## Resume3 failure provenance

- Resume3 implementation commit: `ef5eebb964ac7a8198506962646e048ccf0f6605`
- Resume3 blocked SHA256: `c769fa1f0af69dfe24612cf9bdb7f43c8ce03fb93f039c069b84b07d088a89cd`
- Resume3 test-gate SHA256: `c1cf8558bc0d5a44018812b034e4ce5d3498c290c68c20d0ef8a249938b05adc`
- Resume3 files/reports modified: `false`

## RTX-3090 replay environment

- GPU: `NVIDIA GeForce RTX 3090`
- Compute capability: `8.6`
- GPU UUID: `GPU-9f63aa95-efea-9318-db68-cd70bd02e9a4`
- PCI bus: `00000000:06:00.0`
- Driver: `580.119.02`
- Python: `3.9.15`
- NumPy: `1.23.3`
- PyTorch: `1.12.1.post200`
- Torch CUDA: `11.2`
- Environment SHA256: `7b51113fb013a7f24716049fe4e11464261af62b0b3c4950cdf454fc21aad260`
- Main worker CUDA context cold before Stage-C: `true`

## Manual calibration diagnostic

- Expected calibration SHA256: `fb91d29c84cedccf8f72458f08c2f213bbbc542663593094c0ab651701a052f0`
- Observed calibration SHA256: `fb91d29c84cedccf8f72458f08c2f213bbbc542663593094c0ab651701a052f0`
- Total differences: `0`
- Primary difference locus: `none`

## Frozen Stage-C entrypoint

- Calibration exact: `true`
- Calibration SHA256: `fb91d29c84cedccf8f72458f08c2f213bbbc542663593094c0ab651701a052f0`
- Control record exact: `true`
- Control record SHA256: `dee61b8e31455eeb0e7e0eceae5c1500e3ce57ca8bd42989378d9f76fd779ae7`
- Stage-C nonzero candidates trained: `0`

## Resume4 contract

- Contract SHA256: `cc2cb6f87947cceb070104a7f50a04e1f40d9074612007f4790688bb77111c50`
- Selection SHA256: `136d4575add0452759106f4f141837dc78ff72f042eb0d08ea37cb0c96334fb6`
- Selected configuration: `none`

## Gated candidates

| Candidate | Cutoff | Target ratio | Lambda | t10 reduction | Train ratio | t10 upper | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| k16_t10_r0p50 | 10 | 0.5 | 0.0228672124541 | -1.1657058 | 3.81159319 | 0 | false |
| k16_t10_r1p00 | 10 | 1 | 0.0457344249082 | -1.2159244 | 20.8863386 | 0 | false |
| k16_t25_r0p50 | 25 | 0.5 | 0.0289924766638 | -0.311035544 | 3.26123656 | 0 | false |
| k16_t25_r1p00 | 25 | 1 | 0.0579849533275 | -0.165211558 | 4.68517566 | 0 | false |
| k16_t50_r0p50 | 50 | 0.5 | 0.0370158194844 | 0.0534388269 | 3.19053499 | 0 | false |
| k16_t50_r1p00 | 50 | 1 | 0.0740316389689 | 0.0569411053 | 4.75405424 | 0 | false |

## Classification

- Primary failure locus: `joint_tradeoff`

## Boundary

- Resume4 used a new write-once RTX-3090 namespace; no Resume3 artifact was deleted, overwritten, or rerun.
- Environment and manual-diff probes ran in disposable child processes.
- The main worker had no initialized CUDA context before the frozen Stage-C entry point.
- No calibration/control SHA or scientific gate was relaxed.
- The frozen probe was not accessed.
- No full-874-row repaired model, reverse sampling, formal pilot, IDM, data collection, candidate execution, DeformableRavens, Phase4, or CPS was run.
- No checkpoint, weights, tensor, NPZ, cache, image, or video was persisted.
