# Phase3.14b-r2.5.8 Stage D Resume2

## Temporal frozen-test recovery

- Stage-D implementation commit: `539521776dd515fac11a84396d3664ce44f31017`
- Stage-D first blocked provenance: `4888d9fa85e7b9411899152015acc6dad4f15223`
- Resume1 implementation commit: `8b692966d9f332ad2557c755c1b3657fc4a94943`
- Resume1 blocked provenance: `790d48f202302909b12ea50d710b6e10a2e9437d`
- Resume2 implementation commit: `9184e2921f5f120877c9d8bc84d8c5fa5596af7e`
- Failure class: `one_frozen_test_requires_its_historical_closed_world_repository_view`
- Base regular view: `6866507c42b9bc9d2d271becd1a9423f61710405`
- Historical closed-world view: `3b30ad610bf35244839c2f78d5fed1921d98c6bc`
- Reconstructed frozen population: `51 files / 1064 passed`
- Test implementation modified or filtered: `false`
- Scientific calibration started before Resume2: `false`

## Resume2 scientific result
# Phase3.14b-r2.5.8 Stage D

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r258_staged_grouped_cv_direction_signal_does_not_form_valid_state_transition`
- Required next path: `CALIBRATE_SURROGATE_GUIDED_CONSTRAINED_DIRECT_X0_INTEGRATOR`

## Immutable Stage-C Resume1 binding

- Base evidence commit: `6866507c42b9bc9d2d271becd1a9423f61710405`
- Base worker SHA256: `8af4981760c2959667379221215c067fcd838b3e73349ab2af5517e197806913`
- Base contract SHA256: `355e3a12e13533b4a6f283344cfda1be20c4d6a4bef2fdd28f2ba03b3397b696`
- Base selection SHA256: `613dd19b4611fddd24e5a5a71c29054b8a668a868d40198caccb42c82f4b3f50`

## Portable environment and frozen control

- Compatibility pass: `true`
- Compatibility SHA256: `03fef5cfcfac7933bca0dfc72e67156bb260d50ae7f2f1f6a6375a34c90ad7ec`
- Observation SHA256: `2c818640e5dca3f81ef7a32fe748396c48b25cf0868acc73443b75db61c74d35`
- GPU: `NVIDIA GeForce RTX 4080 SUPER`
- Compute capability: `8.9`
- Required-operation dry run: `true`
- Cold CUDA context before control replay: `true`
- Stage-C reference equivalence: `true`

## Split and leakage boundary

- Objective-train rows / groups: `638` / `126`
- Selection-holdout rows: `236`
- Frozen-probe rows: `126`
- Candidate selection uses holdout: `false`
- Holdout evaluated: `false`
- Frozen probe accessed: `false`
- Selectable features translation invariant: `true`
- Absolute-coordinate diagnostic selectable: `false`

## Baselines

| Baseline | t10 cosine | t25 cosine | t50 cosine |
|---|---:|---:|---:|
| global_mean | 0.291601 | 0.302561 | 0.224413 |
| condition_mean | 0.292504 | 0.303536 | 0.225853 |

## Objective-train grouped OOF candidates

| Candidate | Role | t10 cosine | t25 cosine | t50 cosine | Direction all | State all | Eligible |
|---|---|---:|---:|---:|---:|---:|---:|
| global_mean | negative_control | 0.291601 | 0.302561 | 0.224413 | false | false | false |
| condition_mean | negative_control | 0.292504 | 0.303536 | 0.225853 | false | false | false |
| absolute_rridge_k32_a10 | diagnostic_control | 0.570222 | 0.584792 | 0.464542 | true | false | false |
| centered_ridge_a1 | selectable | 0.509292 | 0.52922 | 0.414589 | true | false | false |
| centered_ridge_a10 | selectable | 0.530723 | 0.554522 | 0.44436 | true | false | false |
| centered_rridge_k16_a10 | selectable | 0.508536 | 0.526129 | 0.410595 | true | false | false |
| centered_rridge_k32_a10 | selectable | 0.523057 | 0.545886 | 0.433665 | true | false | false |
| segment_rridge_k16_a10 | selectable | 0.509274 | 0.528475 | 0.415669 | true | false | false |
| segment_rridge_k32_a10 | selectable | 0.523073 | 0.548615 | 0.440667 | true | false | false |

## OOF scientific witnesses

### `centered_ridge_a1`

| t | Direction gate | Scientific witness |
|---:|---:|---|
| 10 | true | none |
| 25 | true | none |
| 50 | true | none |

### `centered_ridge_a10`

| t | Direction gate | Scientific witness |
|---:|---:|---|
| 10 | true | none |
| 25 | true | none |
| 50 | true | none |

### `centered_rridge_k16_a10`

| t | Direction gate | Scientific witness |
|---:|---:|---|
| 10 | true | none |
| 25 | true | none |
| 50 | true | none |

### `centered_rridge_k32_a10`

| t | Direction gate | Scientific witness |
|---:|---:|---|
| 10 | true | none |
| 25 | true | none |
| 50 | true | none |

### `segment_rridge_k16_a10`

| t | Direction gate | Scientific witness |
|---:|---:|---|
| 10 | true | none |
| 25 | true | none |
| 50 | true | none |

### `segment_rridge_k32_a10`

| t | Direction gate | Scientific witness |
|---:|---:|---|
| 10 | true | none |
| 25 | true | none |
| 50 | true | none |

## Locked selection

- Objective-train selected configuration: `none`
- Validated train-only recommendation: `none`
- Permutation control pass: `not-run`
- Locked holdout scientific pass: `not-run`
- Eligible candidates: `none`
- Primary failure locus: `direction_without_state_reachability`
- Contract SHA256: `3bfb7f19a1f2d37ee8e7f2f202c2a304396ee1b611932ca2012d841b4540f6e4`
- Selection SHA256: `3f78083f8038b283648b03717a95ee00b5851abf69ab3ee256959b180b0f8dba`

## Boundary

- Surrogate candidates were fitted only on the 638-row objective-training population.
- Candidate and scale selection used only six-fold grouped OOF objective-train predictions.
- The 236-row selection holdout could not change candidate, features, regularization, output rank, scale, or thresholds.
- Group IDs, pair keys, visible seeds, and window indices were excluded from all features.
- Absolute-coordinate features were diagnostic-only and could not be selected.
- No diffusion-model candidate was trained and no model architecture was changed.
- The frozen probe was not accessed.
- No reverse sampling, formal training, IDM, candidate execution, DeformableRavens, Phase4, or CPS was run.
- No checkpoint, weights, surrogate weights, prediction tensor, oracle tensor, candidate tensor, NPZ, cache, image, or video was persisted.
